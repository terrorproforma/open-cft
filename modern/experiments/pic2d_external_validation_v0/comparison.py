"""The ASME V&V 20 comparison spec of external validation v0 (predeclared; evaluated only after the run).

For every compared quantity (ASME V&V 20-2009, reaffirmed 2021):

    E      = S - D                         S = our simulation result, D = the reference's published value
    u_val  = sqrt(u_num^2 + u_input^2 + u_D^2)
    u_num  = our numerical uncertainty (predicted here from the accepted 50 um convergence pair: seed-b <= 1.1 %, W x 0.7 5.7 % I_d / 4.6 % S / 11.9 % peak n_e /
             9.3 % T_e,peak -> the particle-resolution band; the grid band is the ss-v4 caveat and is NOT in u_num until the 15 um follow-up runs)
    u_input = the propagated input uncertainty (declared per row; v0 propagates NONE - the B-scale, neutral-profile and effective-source uncertainties are
             recorded as unpropagated inputs and make every row CONDITIONAL, ASME V&V 20 s.1-2)
    u_D    = the reference's uncertainty: stated precision + figure digitisation + the reference's own run-to-run variability (paper vs thesis)

Predeclared statements per row (evaluated with the RUN's S; k = 2 coverage):

    agreement_within_u_val      |E| <= 2 u_val
    agreement_within_tolerance  2 u_val < |E| <= tolerance     (tolerance = the literature's scalar-agreement norm for PIC-vs-PIC / PIC-vs-experiment: 20 % on currents
                                                                and fractions, +-5 V on potential steps, 0.3 dex on densities, 20 % on wall energies / fluxes)
    discrepancy                 |E| > tolerance                 -> attributed FIRST to the declared closure differences (row 'closure_differences'), never to the
                                                                PIC kernel, until the sensitivity variants (Bohm alpha 0.4; plume box) have run

The reference is a published MODEL output (``EvidenceKind.PUBLISHED_EXTERNAL`` -> ``SourceAuthority.PUBLISHED_MODEL_OUTPUT``), so the claim ceiling of the
whole exercise is ``ClaimLevel.CROSS_MODEL_AGREEMENT``: it opens no physics level (GATE-L3 stays closed) and validates nothing against hardware.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from cft_revival.validation.contracts import ClaimLevel, EvidenceKind, SourceAuthority

from . import reference

SCHEMA = "cft.pic2d.external-validation-v0.comparison-spec/0.1.0-draft"
COVERAGE_FACTOR = 2.0
CLAIM_CEILING = ClaimLevel.CROSS_MODEL_AGREEMENT.name
EVIDENCE_KIND = EvidenceKind.PUBLISHED_EXTERNAL.value
SOURCE_AUTHORITY = SourceAuthority.PUBLISHED_MODEL_OUTPUT.value

# our predicted numerical uncertainty (relative, standard) from the accepted 50 um convergence pair (steady-state v2 base vs seed-b / W x 0.7), used as the
# particle-resolution band of a quantity class; the W x 0.7 number is the larger of the pair and is taken as ONE standard uncertainty (conservative)
U_NUM_RELATIVE = {
    "current": 0.057,          # I_d: W x 0.7 +5.7 % (seed-b -0.1 / -0.9 %)
    "ionisation_rate": 0.046,  # S: W x 0.7 -4.6 % (seed-b -0.8 %)
    "beam_current": 0.06,      # I_beam: no pair number; declared = the I_d band rounded up
    "peak_density": 0.119,     # peak n_e: W x 0.7 -11.9 % (seed-b -8.2 %)
    "temperature": 0.093,      # T_e,peak: W x 0.7 -9.3 % (seed-b -1.1 %)
    "wall_flux": 0.12,         # declared = the peak-density band (wall fluxes are density-limited); no pair number
    "wall_energy": 0.093,      # declared = the temperature band
}
U_NUM_ABSOLUTE = {
    "potential_step_v": 2.0,   # DECLARED: no replicate exists for the cusp steps; 2 V = the accepted plateau's staircase read-off resolution (bin of the n_e-weighted cell potential)
}
GRID_CAVEAT = ("the grid band is NOT in u_num: the run is at the published 20 um grid (Delta/lambda_D 2.7 at the published density / 10 eV, between the v2.0.3 soft 2.5 and "
               "hard pi levels); a 15 um sibling is the declared follow-up that turns the grid caveat into a u_num component")

CLOSURE_DIFFERENCES = [
    {"item": "anomalous cross-field transport", "reference": "Bohm-type D_perp = 0.4 k T_e / (e B) by perpendicular-velocity rotation", "ours": "NONE in the primary run (accepted v1.3 closure); "
     "the sealed variant 'bohm-0.4' switches the v1.4 isotropic Bohm-scattering hook on at alpha = 0.4 (nu_an = alpha omega_ce -> D_perp ~ alpha k T_e / e B for alpha << 1; at 0.4 the exact "
     "factor is alpha / (1 + alpha^2) = 0.345) - a DIFFERENT model of the same coefficient (isotropic redirect changes the parallel speed too)",
     "expected_direction": "less cross-field loss -> higher confinement, higher density / T_e, LARGER cusp potential steps and anode current in our primary run"},
    {"item": "secondary electron emission", "reference": "50 % of impacting electrons re-emitted at 90 % of the incident energy", "ours": "none (the SEE scaffold fails closed)",
     "expected_direction": "our dielectric floats more negative -> larger sheath drop, fewer wall electron losses"},
    {"item": "neutral background", "reference": "static DSMC profile ~6e20 -> ~1e20 along the channel (mean ~2e20), 500 K, diffuse walls", "ours": "static UNIFORM 2e20 at 500 K (no inventory)",
     "expected_direction": "our ionisation is redistributed toward the exit relative to the reference (their n_g is 3x higher near the anode)"},
    {"item": "electron source", "reference": "uniform 1 eV volume source of 17.55 mA on the outer rim of a 6.48 mm plume box, of which ~1.8-1.95 mA reach the channel", "ours": "channel-only: 1.8 mA at 1 eV "
     "injected at the exit plane (the reference's continuity-derived effective source, TAKEN FROM ITS RESULTS: I_anode - I_beam)", "expected_direction": "our anode current is conditioned on the "
     "reference's outcome by construction: I_a - I_inj (= the ionisation-borne current) is the independent part"},
    {"item": "exit boundary", "reference": "plume box to 20.48 mm with 0 V far boundaries; the main potential drop forms a 'bulge' beyond the exit", "ours": "Dirichlet 0 V exit plane at 14 mm (channel-only)",
     "expected_direction": "the whole 400 V drop is forced inside our channel: the exit cell potential, the ion exit energy and everything downstream are not comparable (comparable_under)"},
    {"item": "self-similarity scaling of the reference", "reference": "factor 4 (paper) / 8 (thesis): lambda_D / L larger by 2.0 / 2.83 in the reference plasma; sheaths relatively thicker",
     "ours": "device scale", "expected_direction": "the reference's cusp potential dips and sheath structure are those of a relatively larger-Debye plasma; our dips may be sharper"},
    {"item": "collision set", "reference": "e-n elastic / excitation / ionisation (LLNL EEDL), Coulomb, CEX, Xe2+ (thesis)", "ours": "e-n elastic / one lumped excitation / single ionisation (Biagi v7.1); no Coulomb, no CEX, no Xe2+",
     "expected_direction": "second-order for the channel scalars; CEX matters for the plume"},
    {"item": "dielectric in the Poisson solve", "reference": "1 mm Al2O3 (epsilon_r 9) between the plasma and the grounded body, surface charge included", "ours": "surface charge on a perfect-insulator "
     "backing (no field inside the dielectric)", "expected_direction": "steady-state wall potential is set by the plasma either way; the transient charging and the non-local coupling differ"},
    {"item": "time to steady state", "reference": "76 us from a seeded plasma column, averaged over 3.2 us", "ours": "plateau rule: >= 3 transits (4.2 us) with trailing-20 % drifts < 5 % of I_d and N_e",
     "expected_direction": "with static neutrals no slow variable remains; a plateau in a few transits is plausible but the reference's 76 us is the longer average"},
]


def _quantity(quantity_id: str, *, estimand: str, comparable_under: tuple[str, ...], u_num_class: str, tolerance: dict[str, Any], u_input: list[dict[str, Any]],
              statement_if_discrepant: str, notes: str | None = None) -> dict[str, Any]:
    d, u_d, components = reference.reported_value_and_u_d(quantity_id)
    row = reference.REPORTED[quantity_id]
    if u_num_class in U_NUM_RELATIVE:
        u_num = {"kind": "relative", "standard": U_NUM_RELATIVE[u_num_class], "source": f"accepted 50 um convergence pair, class '{u_num_class}'", "absolute_at_d": U_NUM_RELATIVE[u_num_class] * abs(d)}
    else:
        u_num = {"kind": "absolute", "standard": U_NUM_ABSOLUTE[u_num_class], "source": f"declared, class '{u_num_class}'", "absolute_at_d": U_NUM_ABSOLUTE[u_num_class]}
    if row.get("log_scale"):
        u_num = {"kind": "log10", "standard": math.log10(1.0 + U_NUM_RELATIVE[u_num_class]), "source": f"accepted 50 um convergence pair, class '{u_num_class}' (as dex)", "absolute_at_d": None}
    return {
        "quantity_id": quantity_id, "description": row["source"], "unit": row["unit"], "kind": row["kind"], "log_scale": bool(row.get("log_scale", False)),
        "D": d, "u_D": {"standard": u_d, "components": components, "coverage_factor": COVERAGE_FACTOR},
        "D_experiment": row.get("experiment"), "D_second_run": row.get("second_run"),
        "S_estimand": estimand, "comparable_under": list(comparable_under),
        "u_num_predicted": {**u_num, "grid_caveat": GRID_CAVEAT},
        "u_input": {"propagated": False, "components": u_input, "note": "NOT propagated in v0 (every row is conditional on these inputs); a B-scale +-8 % run and the uniform-vs-profile neutral "
                                                                        "sensitivity are the declared follow-ups"},
        "tolerance": tolerance,
        "statements": {"agreement_within_u_val": f"|E| <= {COVERAGE_FACTOR:g} u_val: the two codes agree within the validation uncertainty on this quantity (cross-model agreement, no hardware statement)",
                       "agreement_within_tolerance": "2 u_val < |E| <= tolerance: agreement at the literature's scalar norm; the excess over u_val is attributed to the declared closure differences",
                       "discrepancy": statement_if_discrepant},
        "notes": notes,
    }


def comparison_rows() -> list[dict[str, Any]]:
    b_scale = {"name": "magnet remanence scale", "standard_relative": 0.083, "method": "the axis anchor 'about 0.6 T' read as +-0.05 T -> +-8.3 % on |B| everywhere (linear field)"}
    n_g = {"name": "neutral background", "standard_relative": 0.5, "method": "uniform 2e20 vs the reference's 1e20-6e20 profile: factor ~2 local deviations"}
    source = {"name": "effective electron source", "standard_relative": 0.08, "method": "1.8 mA (continuity) vs 1.95 mA (paper's own estimate): +-0.15 mA"}
    both = ("channel", "plume-brandt")
    return [
        _quantity("anode_electron_current_a", estimand="summary.window_currents_a.discharge_a of the trailing window (anode electron current = discharge current)", comparable_under=both,
                  u_num_class="current", tolerance={"kind": "relative", "value": 0.20, "basis": "literature norm for PIC discharge currents (Szabo 2014 16 %, Brandt-vs-experiment 5 %)"},
                  u_input=[b_scale, n_g, source], statement_if_discrepant="|E| > 20 %: the anode current disagrees beyond the closure-conditioned band; first suspects the absent Bohm transport "
                  "(sealed variant bohm-0.4) and the uniform neutral background; NO statement about the PIC kernel",
                  notes="conditioned on the injected 1.8 mA: I_a - I_inj (the ionisation-borne part, reference 2.5 mA) is the independent comparison (row beam_fraction / ion_beam_current)"),
        _quantity("net_ionisation_fraction", estimand="I_a / (e Q_in) with the reference's Q_in = 1.1e17 /s (Brandt's own definition; NOT the v1.3 gross utilisation)", comparable_under=both,
                  u_num_class="current", tolerance={"kind": "absolute", "value": 0.05, "basis": "5 percentage points (the reference's own 24 vs 25 % agreement with experiment)"},
                  u_input=[b_scale, n_g, source], statement_if_discrepant="a consistency row (same number as the anode current divided by e Q_in): disagreement here IS the anode-current disagreement",
                  notes="not independent of anode_electron_current_a"),
        _quantity("ion_beam_current_a", estimand="summary.window_currents_a.exit_ion_beam_a (ion current through the exit plane); under the plume option the ion current through the outer boundary",
                  comparable_under=both, u_num_class="beam_current", tolerance={"kind": "relative", "value": 0.20, "basis": "literature norm; the reference itself is 20 % below its experiment"},
                  u_input=[b_scale, n_g, source], statement_if_discrepant="|E| > 20 %: the ion production that leaves the channel disagrees; channel-only counts every exit-plane ion as beam (the "
                  "reference loses ions to the front face and the body) - the plume option is the discriminating follow-up",
                  notes="channel-only over-counts the beam relative to a plume-boundary count; the sign of that bias is known (ours >= comparable)"),
        _quantity("beam_fraction_of_feed", estimand="exit_ion_beam_a / (e Q_in)", comparable_under=both, u_num_class="beam_current",
                  tolerance={"kind": "absolute", "value": 0.03, "basis": "3 percentage points = 20 % of the reference's 14 %"}, u_input=[b_scale, n_g, source],
                  statement_if_discrepant="as ion_beam_current_a"),
        _quantity("plasma_potential_near_anode_above_anode_v", estimand="n_e-weighted mean of phi - U_anode over the anode-side cell (0 < z < first cusp plane, r < r_w - 0.5 mm) of the trailing-window map",
                  comparable_under=both, u_num_class="potential_step_v", tolerance={"kind": "absolute", "value": 5.0, "basis": "+-5 V = twice the reference's rounding unit"},
                  u_input=[b_scale, n_g], statement_if_discrepant="|E| > 5 V: the anode-cell plasma potential disagrees (T_e and the anode sheath differ)"),
        _quantity("potential_drop_first_cusp_v", estimand="difference of the n_e-weighted cell potentials across the anode-side interior cusp plane (cell 1 - cell 2, anode -> exit), trailing-window map",
                  comparable_under=both, u_num_class="potential_step_v", tolerance={"kind": "absolute", "value": 5.0, "basis": "+-5 V"}, u_input=[b_scale, n_g],
                  statement_if_discrepant="|E| > 5 V: the cusp potential step disagrees; the step is set by the cross-field electron mobility across the cusp - the absent Bohm transport is the first "
                  "suspect (variant bohm-0.4), then the linear-iron over-focusing of the cusp field (A4)"),
        _quantity("potential_drop_second_cusp_v", estimand="difference of the n_e-weighted cell potentials across the exit-side interior cusp plane (cell 2 - cell 3), trailing-window map",
                  comparable_under=both, u_num_class="potential_step_v", tolerance={"kind": "absolute", "value": 5.0, "basis": "+-5 V"}, u_input=[b_scale, n_g],
                  statement_if_discrepant="as potential_drop_first_cusp_v; under channel-only the exit cell is also pulled by the 0 V exit plane (A9)",
                  notes="channel-only: the exit-side cell is the one distorted by the Dirichlet exit plane - comparable with that caveat"),
        _quantity("ion_density_typical_per_m3", estimand="log10 of the trailing-window peak n_i (densest node with >= 32 macro-particles) inside the channel; ALSO the channel-volume mean n_i reported",
                  comparable_under=both, u_num_class="peak_density", tolerance={"kind": "log10", "value": 0.3, "basis": "factor 2 (colour-scale read-off plus the reference's own internal 1e19 vs 4e18)"},
                  u_input=[b_scale, n_g], statement_if_discrepant="|E| > 0.3 dex: the channel density disagrees by more than a factor 2; the neutral background (uniform vs profile) and the transport "
                  "closure are the first suspects"),
        _quantity("wall_ion_energy_max_ev", estimand="max over the channel wall of the trailing-window wall ion mean energy map (maps.npz wall_ion_mean_energy_ev), and its z",
                  comparable_under=both, u_num_class="wall_energy", tolerance={"kind": "relative", "value": 0.20, "basis": "20 %"}, u_input=[b_scale],
                  statement_if_discrepant="|E| > 20 %: the sheath drop at the exit-side cusp disagrees (SEE absent in ours -> our wall floats more negative -> HIGHER impact energy expected)"),
        _quantity("wall_ion_current_density_max_a_per_m2", estimand="max over the channel wall of the trailing-window wall ion flux map times e (A/m^2), and its z",
                  comparable_under=both, u_num_class="wall_flux", tolerance={"kind": "relative", "value": 0.20, "basis": "20 %"}, u_input=[b_scale, n_g],
                  statement_if_discrepant="|E| > 20 %: the cusp wall ion flux disagrees; density and cusp field (A4) are the suspects"),
        _quantity("plume_peak_angle_deg", estimand="angle of the maximum of the far-field ion current per unit solid angle (plume option only: plume_ion_current_per_sr_a about the exit centre)",
                  comparable_under=("plume-brandt",), u_num_class="current", tolerance={"kind": "absolute", "value": 10.0, "basis": "two 5 deg bins = the reference's own paper-vs-thesis spread"},
                  u_input=[b_scale], statement_if_discrepant="|E| > 10 deg: the hollow-cone angle disagrees (both codes flag their far boundary as too close)",
                  notes="NOT comparable under the primary channel-only protocol"),
        _quantity("electron_energy_near_exit_cusp_ev", estimand="max of the trailing-window mean electron energy map in the exit-cusp region (14 < z < 18 mm; plume option only)",
                  comparable_under=("plume-brandt",), u_num_class="temperature", tolerance={"kind": "relative", "value": 0.25, "basis": "the reference's own 'around 200 eV'"},
                  u_input=[b_scale], statement_if_discrepant="|E| > 25 %: the exit-cusp electron energy disagrees", notes="NOT comparable under the primary channel-only protocol (the exit cusp at ~16 mm lies outside the channel box)"),
    ]


def qualitative_rows() -> list[dict[str, Any]]:
    return [{"id": k, **v, "our_observable": obs, "comparable_under": under} for (k, v), (obs, under) in zip(reference.QUALITATIVE.items(), (
        ("axial n_e-weighted potential profile: flat to within 5 % of U_anode over the interior cells; the exit-cell drop is forced by the 0 V plane under channel-only", ["channel", "plume-brandt"]),
        ("phi(axis) - phi(wall - 3 dr) at each cusp plane (closure.extract_targets sheath_drop_v): 'several 10 V' dips", ["channel", "plume-brandt"]),
        ("ionisation-rate map: local maxima on the axis, at the cusp planes and upstream of each cusp (renderer v0.2 windowed panel)", ["channel", "plume-brandt"]),
        ("wall ion flux and mean-energy maps: maxima within +-0.5 mm of the cusp planes, < 10 % of the maximum elsewhere", ["channel", "plume-brandt"]),
        ("radial phi profile at the cusp planes: sheath width 5-10 lambda_D (lambda_D from the local trailing-window n_e, T_e)", ["channel", "plume-brandt"]),
        ("radial n_e profile between the cusps: half-maximum radius / r_w compared with the reference's 'more uniform fill'", ["channel", "plume-brandt"]),
    ))]


def validation_metric(quantity: Mapping[str, Any], s_value: float, *, u_num_measured: float | None = None) -> dict[str, Any]:
    """E = S - D, u_val and the predeclared statement for one row.  ``u_num_measured`` (standard, in the row's unit or dex) replaces the predicted u_num when a run supplies it."""

    log_scale = bool(quantity.get("log_scale", False))
    d = math.log10(float(quantity["D"])) if log_scale else float(quantity["D"])
    s = math.log10(float(s_value)) if log_scale else float(s_value)
    error = s - d
    u_d = float(quantity["u_D"]["standard"])
    predicted = quantity["u_num_predicted"]
    if u_num_measured is not None:
        u_num = float(u_num_measured)
    elif predicted["kind"] == "relative":
        u_num = float(predicted["standard"]) * abs(d)
    else:
        u_num = float(predicted["standard"])
    u_input = 0.0            # not propagated in v0 (declared)
    u_val = math.sqrt(u_num**2 + u_input**2 + u_d**2)
    tolerance = quantity["tolerance"]
    if tolerance["kind"] == "relative":
        tol = float(tolerance["value"]) * abs(d)
    else:
        tol = float(tolerance["value"])
    if abs(error) <= COVERAGE_FACTOR * u_val:
        verdict = "agreement_within_u_val"
    elif abs(error) <= tol:
        verdict = "agreement_within_tolerance"
    else:
        verdict = "discrepancy"
    return {"quantity_id": quantity["quantity_id"], "D": d, "S": s, "E": error, "u_num": u_num, "u_input": u_input, "u_input_propagated": False, "u_D": u_d, "u_val": u_val,
            "coverage_factor": COVERAGE_FACTOR, "expanded_u_val": COVERAGE_FACTOR * u_val, "tolerance": tol, "verdict": verdict, "statement": quantity["statements"][verdict],
            "scale": "log10" if log_scale else "linear", "conditional_on_unpropagated_inputs": [c["name"] for c in quantity["u_input"]["components"]],
            # when the expanded validation uncertainty already exceeds the tolerance the row cannot discriminate at the tolerance level: the reference's own
            # precision (u_D) or our particle band (u_num) dominates - reported, never hidden (comparison_document inconclusive_conditions)
            "tolerance_below_expanded_u_val": bool(tol < COVERAGE_FACTOR * u_val),
            "u_val_dominated_by": "u_D" if u_d >= u_num else "u_num"}


def comparison_document() -> dict[str, Any]:
    return {
        "schema_version": SCHEMA,
        "status": "DRAFT predeclared comparison spec; NOT preregistered; evaluated only after the run with `run.py compare`",
        "form": "ASME V&V 20-2009 (R2021): E = S - D; u_val = sqrt(u_num^2 + u_input^2 + u_D^2); statements at k = 2",
        "reference": {"citation": reference.CITATION, "doi": reference.DOI, "thesis_urn": reference.THESIS["urn"], "evidence_kind": EVIDENCE_KIND, "source_authority": SOURCE_AUTHORITY,
                      "claim_ceiling": CLAIM_CEILING, "claim_note": "a published model output: the ceiling is cross-model agreement; opens no physics level; GATE-L3 stays closed"},
        "u_num_policy": {"relative_classes": U_NUM_RELATIVE, "absolute_classes": U_NUM_ABSOLUTE, "grid_caveat": GRID_CAVEAT,
                         "measured_replacement": "if a seed replicate of THIS case runs, its half-spread replaces the predicted class value (validation_metric u_num_measured)"},
        "u_input_policy": "not propagated in v0; every row lists its unpropagated inputs and is reported as conditional on them",
        "coverage_factor": COVERAGE_FACTOR,
        "quantities": comparison_rows(),
        "qualitative": qualitative_rows(),
        "closure_differences": CLOSURE_DIFFERENCES,
        "inconclusive_conditions": inconclusive_conditions(),
    }


def inconclusive_conditions() -> list[dict[str, str]]:
    """What would make v0 inconclusive - stated before the run."""

    return [
        {"condition": "no plateau within the wall budget", "consequence": "no S exists; the trailing-window values are reported as 'transient at t = ...'; no E is formed"},
        {"condition": "hard peak-Debye gate (pi) or omega_pe dt gate stop", "consequence": "the discharge densified beyond the 20 um / 0.7 ps envelope (interval-averaged peak > 1.36e19 x T_e / 10 eV for the "
                                                                             "Debye gate; > 2.6e19 for omega_pe dt): resolution-limited at the published grid; the 15 um / 0.5 ps sibling (hard level "
                                                                             "2.4e19 at 10 eV) is the only route; no E is formed"},
        {"condition": "plateau with the peak between the soft 2.5 and hard pi levels", "consequence": "E is formed but every row carries 'resolution margin not met'; the grid caveat becomes a discrepancy suspect"},
        {"condition": "windowed residual power >= 2 % at the plateau", "consequence": "grid heating: E is reported but not quotable (the ss-v4 acceptance (b) rule)"},
        {"condition": "no ignition at 2e20 with 1.8 mA / 1 eV injection and the 5e16 seed", "consequence": "recorded outcome; the seed / injection are the frozen inputs; no relaunch with adjusted inputs under this spec"},
        {"condition": "a field gate G1-G7 fails", "consequence": "the reconstruction, not the plasma model, is under test: the run is not launched (preflight refuses)"},
        {"condition": "every current / potential row discrepant in the direction the closure table predicts (more confinement in ours)", "consequence": "the exercise measured the closure difference, "
                                                                                                                                             "not the kernels; the bohm-0.4 variant is the discriminating run"},
        {"condition": "the reference's own two runs (paper 4.3 mA / 50 deg; thesis 4.7 mA / 60 deg) already span the tolerance", "consequence": "the row cannot discriminate (u_D dominates u_val): reported as such"},
        {"condition": "channel-only rows only", "consequence": "the plume angle and the exit-cusp electron energy are NOT compared in v0 (comparable_under); a 'validation of the plume' claim is not available"},
    ]


def validate_comparison_spec(document: Mapping[str, Any]) -> list[str]:
    """Schema check of a comparison spec; returns the list of problems (empty = valid)."""

    problems: list[str] = []
    if document.get("schema_version") != SCHEMA:
        problems.append("schema_version")
    ref = document.get("reference") or {}
    for key in ("doi", "claim_ceiling", "evidence_kind", "source_authority"):
        if not ref.get(key):
            problems.append(f"reference.{key}")
    if ref.get("claim_ceiling") != CLAIM_CEILING:
        problems.append("reference.claim_ceiling must be CROSS_MODEL_AGREEMENT for a published model output")
    ids = set()
    for row in document.get("quantities") or []:
        qid = row.get("quantity_id", "?")
        if qid in ids:
            problems.append(f"{qid}: duplicate")
        ids.add(qid)
        for key in ("D", "u_D", "S_estimand", "comparable_under", "u_num_predicted", "u_input", "tolerance", "statements", "unit"):
            if key not in row:
                problems.append(f"{qid}: missing {key}")
        if not isinstance(row.get("D"), (int, float)) or not math.isfinite(float(row.get("D", float("nan")))):
            problems.append(f"{qid}: D must be finite")
        u_d = row.get("u_D") or {}
        if not u_d.get("components") or u_d.get("standard", -1) < 0:
            problems.append(f"{qid}: u_D needs components and a non-negative standard value")
        else:
            rss = math.sqrt(sum(float(c["standard"]) ** 2 for c in u_d["components"]))
            if abs(rss - float(u_d["standard"])) > 1e-12 * max(1.0, rss):
                problems.append(f"{qid}: u_D.standard is not the root-sum-square of its components")
        if not set(row.get("comparable_under") or ()) <= {"channel", "plume-brandt"} or not row.get("comparable_under"):
            problems.append(f"{qid}: comparable_under must be a non-empty subset of the domain options")
        tol = row.get("tolerance") or {}
        if tol.get("kind") not in ("relative", "absolute", "log10") or not tol.get("basis") or not (float(tol.get("value", -1)) > 0):
            problems.append(f"{qid}: tolerance needs kind, positive value and basis")
        if set(row.get("statements") or {}) != {"agreement_within_u_val", "agreement_within_tolerance", "discrepancy"}:
            problems.append(f"{qid}: the three predeclared statements are required")
        u_in = row.get("u_input") or {}
        if u_in.get("propagated") is not False or not isinstance(u_in.get("components"), list):
            problems.append(f"{qid}: u_input must declare propagated=False with a component list in v0")
    if not document.get("closure_differences"):
        problems.append("closure_differences")
    if not document.get("inconclusive_conditions"):
        problems.append("inconclusive_conditions")
    return problems


__all__ = ["CLAIM_CEILING", "CLOSURE_DIFFERENCES", "COVERAGE_FACTOR", "GRID_CAVEAT", "SCHEMA", "U_NUM_ABSOLUTE", "U_NUM_RELATIVE", "comparison_document", "comparison_rows",
           "inconclusive_conditions", "qualitative_rows", "validate_comparison_spec", "validation_metric"]
