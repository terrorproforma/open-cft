"""The published reference case of external validation v0 and everything extracted from it.

Reference (verified 2026-09-04 against the Crossref work record ``api.crossref.org/works/10.2322/tastj.14.Pb_235`` and the
open-access full text on J-STAGE):

    Brandt, T., Schneider, R., Duras, J., Kahnfeld, D., Hey, F. G., Kersten, H., Jansen, F., Braxmaier, C. (2016).
    Particle-in-Cell Simulation of a Down-Scaled HEMP Thruster.  Transactions of the Japan Society for Aeronautical and
    Space Sciences, Aerospace Technology Japan 14 (ists30), Pb_235-Pb_242.  doi:10.2322/tastj.14.Pb_235

The paper publishes the discharge-channel envelope, the operating point, the numerical parameters and the scalar results, but
NOT the magnet stack.  The stack (three SmCo rings, five soft-iron distance rings, their dimensions and materials) is published
in the same author's doctoral thesis, read in full text on 2026-09-04:

    Brandt, T. (2017).  Computer modeling for improvement of a High Efficiency Multistage Plasma Thruster.  Dr. rer. nat.
    thesis, Christian-Albrechts-Universitaet zu Kiel.  URN urn:nbn:de:gbv:8-diss-224024
    (https://nbn-resolving.org/urn:nbn:de:gbv:8-diss-224024), chapters 6-7.

Every number below carries its source (paper page or thesis chapter), its kind (``text`` = stated in the running text or a
table; ``figure`` = read off a figure, digitisation needed; ``inferred`` = derived from stated numbers by the rule given) and,
for the quantities the comparison spec uses, its uncertainty budget.  Nothing here is our result; the comparison spec
(``comparison.py``) turns the reported quantities into ASME V&V 20 rows.
"""

from __future__ import annotations

from typing import Any

from cft_revival.validation.contracts import validate_doi

ELEMENTARY_CHARGE_C = 1.602176634e-19

DOI = validate_doi("10.2322/tastj.14.Pb_235")
CITATION = ("Brandt, T., Schneider, R., Duras, J., Kahnfeld, D., Hey, F. G., Kersten, H., Jansen, F., Braxmaier, C. (2016). "
            "Particle-in-Cell Simulation of a Down-Scaled HEMP Thruster. Trans. JSASS Aerospace Tech. Japan 14(ists30), Pb_235-Pb_242.")
OPEN_ACCESS_PDF = "https://www.jstage.jst.go.jp/article/tastj/14/ists30/14_Pb_235/_pdf"
THESIS = {
    "citation": "Brandt, T. (2017). Computer modeling for improvement of a High Efficiency Multistage Plasma Thruster. Dr. rer. nat. thesis, "
                "Christian-Albrechts-Universitaet zu Kiel.",
    "urn": "urn:nbn:de:gbv:8-diss-224024",
    "url": "https://nbn-resolving.org/urn:nbn:de:gbv:8-diss-224024",
    "role": "supplies the magnet stack (chapter 6 'the same periodic arrangement of magnets as for the micro-HEMPT'; chapter 7 'Setup including the magnets "
            "assembly') and a second, independently run version of the same case (thesis chapter 7: domain 19.12 mm, scaling factor 8, I_a 4.7 mA, peak "
            "angle 60 deg) whose spread against the paper's numbers is the reference's own reproducibility",
}
VERIFICATION = {
    "doi_verified": "Crossref work record fetched 2026-09-04 (title, authors, container, volume 14, issue ists30, pages Pb_235-Pb_242, year 2016); "
                    "Crossref records the DOI in lower case (10.2322/tastj.14.pb_235); both forms resolve",
    "full_text": "open-access PDF (J-STAGE) downloaded 2026-09-04 and read in full (8 pages)",
    "thesis": "full text read through the URN landing page on 2026-09-04 (chapters 5-8)",
}

# --------------------------------------------------------------------------------------------------------------------------
# Why this reference and not another (all DOIs verified in modern/docs/literature/*.md unless stated)
# --------------------------------------------------------------------------------------------------------------------------

ALTERNATIVES_CONSIDERED = [
    {"reference": "Matyash, Schneider, Mutzke, Kalentev, Taccogna, Koch, Schirra (2010) IEEE Trans. Plasma Sci. 38(9) 2274-2280",
     "doi": "10.1109/TPS.2010.2056936",
     "device": "Thales HEMP-T DM3a (51 mm x 9 mm channel) at ~1 kV",
     "why_not": "9 mm bore at Debye-resolving cells is 8-30x our cell count per transit; the DM3a magnet stack, operating point and scalar results are "
                "not tabulated in the paper (closed access; abstract only in our review)"},
    {"reference": "Matthias, Kahnfeld, Schneider, Yeo, Ogawa (2019) Contrib. Plasma Phys. 59(9) e201900028",
     "doi": "10.1002/ctpp.201900028",
     "device": "optimised downscaled CFT of the Fahey-Muffatti-Ogawa MDO",
     "why_not": "design point, field and numerical parameters not in the abstract (closed access); our lineage's own design - a comparison would test the "
                "MDO chain, not the PIC kernel"},
    {"reference": "Kahnfeld, Heidemann, Duras, Matthias, Bandelow, Luskow, Kemnitz, Matyash, Schneider (2018) Plasma Sources Sci. Technol. 27 124002",
     "doi": "10.1088/1361-6595/aaf29a",
     "device": "DM3a, breathing modes",
     "why_not": "time-dependent (100 kHz breathing) target needing neutral dynamics we do not model; DM3a scale"},
    {"reference": "Lewerentz, Kahnfeld, Schulz, Heidemann, Schneider (2022) Frontiers in Physics 10 833159",
     "doi": "10.3389/fphy.2022.833159",
     "device": "MS4 thruster",
     "why_not": "different device class (multi-stage MS4, larger); not examined beyond the abstract in our reviews"},
    {"reference": "Keller, Koehler, Hey, Berger, Braxmaier, Feili, Weise, Johann (2015) IEEE Trans. Plasma Sci. 43(1) 45-53",
     "doi": "10.1109/TPS.2014.2321095",
     "device": "the SAME micro-HEMPT family, experiment",
     "why_not": "experimental (validation v2's target, not a code-to-code case); geometry given as ranges (chamber 2-5 mm, SmCo OD 10-40 mm, cusp length "
                "1-10 mm) - Brandt 2016 / 2017 pin ONE of these configurations and simulate it"},
]
WHY_BRANDT = (
    "the only published HEMP PIC-MCC case near our scale (14 x 1.5 mm channel vs our 24 x 2 mm; 400 V vs 300 V; mA-class) whose full text is open "
    "access and which states its operating point, grid, time step, super-particle ratio, boundary conditions and SCALAR results in the text; the "
    "same code family (Greifswald PIC-MCC) as every other HEMP PIC paper; the companion thesis publishes the magnet stack so the field is "
    "reconstructable without author contact; and the case has a second published run (thesis chapter 7) that bounds the reference's own "
    "reproducibility. The cost of the case at device scale is affordable (section 'protocol')."
)

# --------------------------------------------------------------------------------------------------------------------------
# Extracted setup (paper unless stated; z measured from the anode surface, r from the axis; SI in the values, mm/sccm in the notes)
# --------------------------------------------------------------------------------------------------------------------------

SETUP: dict[str, dict[str, Any]] = {
    "channel_radius_m": {"value": 1.5e-3, "kind": "text", "source": "paper Pb_235-236 ('R_thr = 1.5 mm'); thesis ch. 7 ('radius of 1.5 mm')"},
    "channel_length_m": {"value": 14.0e-3, "kind": "text", "source": "paper Pb_235-236 ('Z_thr = 14 mm', Z counted from the anode surface); thesis ch. 7"},
    "dielectric": {"value": {"material": "Al2O3", "inner_radius_m": 1.5e-3, "outer_radius_m": 2.5e-3, "thickness_m": 1.0e-3, "relative_permittivity": 9.0,
                             "surface_charge_on": "r = 1.5 mm for 0 <= z <= 14 mm and the top end z = 14 mm for 1.5 <= r <= 2.5 mm"},
                   "kind": "text", "source": "paper Pb_237 ('At r = 1.5 mm over 0 <= z <= 14 mm surface charge accumulation ... A surface with equal properties "
                                             "is at z = 14 mm, 1.5 mm <= r <= 2.5 mm'; 'The dielectrics in this model is quite narrow (1 mm)'); thesis ch. 7 "
                                             "(Al2O3, epsilon_r = 9, inner radius 1.5 mm, thickness 1 mm)"},
    "grounded_body": {"value": {"z_min_m": 0.0, "z_max_m": 14.0e-3, "r_min_m": 2.5e-3, "r_max_m": 5.12e-3},
                      "kind": "text", "source": "paper Pb_237 ('The volume 0 mm <= z <= 14 mm, 2.5 mm <= r <= 5.12 mm is grounded, in order to represent the magnets and their distance rings')"},
    "anode": {"value": {"z_m": 0.0, "r_min_m": 0.0, "r_max_m": 1.5e-3, "potential_v": 400.0},
              "kind": "text", "source": "paper Pb_237 ('The anode lies at z = 0 mm ... 0 mm <= r <= 1.5 mm on a potential of 400 V'); thesis ch. 7 sets the whole "
                                        "z = 0 boundary out to r = 2.5 mm at anode potential and notes the physical anode ends at r = 1.25 mm (neutral gap 1.25-1.5 mm)"},
    "simulation_domain": {"value": {"r_max_m": 5.12e-3, "z_max_m": 20.48e-3, "outer_boundary": "0 V Dirichlet on 14 <= z <= 20.48 mm at r = 5.12 mm and on z = 20.48 mm"},
                          "kind": "text", "source": "paper Pb_236-237; thesis ch. 7 uses Z = 19.12 mm (a second run of the case)"},
    "anode_voltage_v": {"value": 400.0, "kind": "text", "source": "paper Pb_237; thesis ch. 7 (experiment: fixed anode potential 400 V)"},
    "mass_flow": {"value": {"sccm": 0.27, "atoms_per_s": 1.1e17, "propellant": "xenon"}, "kind": "text",
                  "source": "paper Pb_236 ('inflow of 0.27 SCCM xenon'), Pb_240 ('neutral gas influx of 1.1e17 particles per second')"},
    "neutral_background": {"value": {"mean_density_per_m3": 2.0e20, "profile": "DSMC import, static; drops from ~6e20 at the injector to ~1e20 at the exit (thesis Fig. 7.3)",
                                     "temperature_k": 500.0, "wall_reflection": "diffuse", "depletion": "neglected (25 % ionised)"},
                           "kind": "text", "source": "paper Pb_236 ('average neutral gas density inside the discharge channel of about 2e20 m^-3', 'kept as a static "
                                                     "background', 'thermal velocity of the neutrals at 500 K'); thesis ch. 7 (6e20 -> 1e20 along the channel)"},
    "cathode_neutraliser": {"value": {"real": "tungsten filaments at r ~ 40 mm, z = 0 from the exit (outside the domain)",
                                      "model": "uniform volume electron source at 1 eV over 14 <= z <= 20.48 mm, 3.84 <= r <= 5.12 mm and 19.2 <= z <= 20.48 mm, 0.01 <= r <= 3.84 mm",
                                      "source_current_a": 17.55e-3, "lost_at_outer_boundary_a": 11.7e-3, "remaining_a": 5.85e-3,
                                      "estimated_reaching_thruster_a": 1.95e-3, "continuity_effective_a": 1.8e-3,
                                      "ignition_aid": "additional electron source at the channel exit for the first 1.5e6 steps"},
                            "kind": "text / inferred", "source": "paper Pb_237 (source areas, 1 eV, ignition aid) and Pb_240 (17.55 / 11.7 / 5.85 mA, 'a third of them reach the "
                                                                 "thruster'); the continuity value 1.8 mA = I_anode - I_beam = 4.3 - 2.5 mA (electron continuity with "
                                                                 "quasi-neutral wall recombination)"},
    "magnet_stack": {"value": {"magnet_count": 3, "magnet_material": "SmCo (samarium-cobalt; grade / remanence not stated)",
                               "magnet_axial_length_m": 5.0e-3, "magnet_inner_radius_m": 2.5e-3, "magnet_outer_radius_m": 15.0e-3,
                               "distance_ring_count": 5, "distance_ring_material": "'Carbon steel forgings, annealed' (soft iron, FEMM library material)",
                               "distance_ring_axial_length_m": 0.5e-3, "distance_ring_inner_radius_m": 2.5e-3, "distance_ring_outer_radius_m": 8.0e-3,
                               "stack_length_m": 18.0e-3, "anode_position": "halfway along the first magnet",
                               "layout_inferred": "ring | magnet 1 | ring | magnet 2 | ring | magnet 3 | ring (four of the five rings placed); with the anode at the "
                                                  "mid-plane of magnet 1: magnet 1 -2.5..2.5 mm, ring 2.5..3.0, magnet 2 3.0..8.0, ring 8.0..8.5, magnet 3 8.5..13.5, "
                                                  "ring 13.5..14.0 (= the channel exit); the fifth ring's position is not stated; 3 x 5 + 5 x 0.5 = 17.5 mm against the "
                                                  "stated 18 mm (0.5 mm discrepancy recorded)",
                               "return_yoke": "none described (FEMM model lists the magnets and the distance rings; the housing is not part of the magnetic model)",
                               "field_solver": "FEMM, r-z, triangular mesh, domain z -37.5..48.5 mm, r 0..50 mm (thesis ch. 7)"},
                     "kind": "text (thesis)", "source": "thesis ch. 6 ('The axial length of the magnetic rings is 5 mm while for the distance rings this length is 0.5 mm ... "
                                                         "inner radii of these rings from 2.5 mm (1 mm thickness version) ... outer radius ... 15 mm for the magnetic rings and "
                                                         "8 mm for the distance rings'; 'magnetization of samarium-cobalt'; distance rings 'Carbon steel forgings, annealed') "
                                                         "and ch. 7 ('three magnetic rings and five distance rings ... The length of this setup is 18 mm. The anode surface is "
                                                         "placed halfway at the first magnet'); paper Pb_236 (FEMM import) and Fig. 1-2 (three magnets of alternating polarity)"},
    "field_anchors": {"value": {"axis_max_t": 0.6, "axis_max_at": {"r_m": 0.0, "z_m": 11.0e-3}, "near_cusps_t": "about 0.2 T and lower",
                                "exit_cusp_axis_t": 0.05, "exit_cusp_axis_at": {"r_m": 0.0, "z_m": 17.0e-3}, "exit_null_z_m": 16.0e-3,
                                "thesis_max_t": 0.7, "internal_nulls": "on the axis both B_z and B_r drop to zero around the z-positions of the distance rings"},
                      "kind": "text", "source": "paper Pb_236 ('maximum flux density inside the micro HEMPTs discharge channel (e.g. at Z = 11 mm, R = 0 mm ...) is about 0.6 T'; "
                                                "'Near the magnetic cusps the flux is about 0.2 T and lower'; 'At the exit cusp outside the channel the flux is also low (e.g. "
                                                "0.05 T at Z = 17 mm, R = 0 mm)'); thesis ch. 7 ('maximum of about 0.7 T'; zero-field point 'outside the thrusters exit at "
                                                "around z = 16 mm'; nulls 'around the z-position of the distance rings')"},
    "grid": {"value": {"cells": [1024, 256], "cell_m": 20.0e-6, "domain_m": [20.48e-3, 5.12e-3]}, "kind": "text",
             "source": "paper Pb_237 ('1024 x 256 cells', 'Delta r = 2e-5 m'); scaled-back (original-system) units"},
    "time_step_s": {"value": 3.17e-12, "kind": "text", "source": "paper Pb_237 ('Delta t = 3.17e-12 s'); thesis ch. 7: dt_0 = 4e-13 s in the factor-8 scaled system = a fifth "
                                                                  "of the plasma period at the scaled maximum density 8e19"},
    "steps": {"value": {"total": 2.4e7, "ignition_source_off_after": 1.5e6, "averaging_window": 1.0e6, "quasi_steady_time_s": 76.12e-6},
              "kind": "text", "source": "paper Pb_237 ('run over 2.4e7 time steps'), Pb_238 ('quasi steady-state ... after 7.612e-5 s (averaged over 1e6 time steps)')"},
    "macroparticles": {"value": {"super_particle_ratio": 2618, "rule": "no less than six super particles in the axis cells at the maximum assumed plasma density",
                                 "note": "the ratio applies in the SCALED system; in original-system units one macro-particle stands for 2618 x s^2 real particles "
                                         "(s = 4 paper / 8 thesis: 4.2e4 / 1.7e5)"},
                       "kind": "text", "source": "paper Pb_237"},
    "self_similarity_scaling": {"value": {"factor_paper": 4, "factor_thesis": 8, "kept_constant": "system length / gyroradius and system length / mean free path",
                                          "consequence": "lambda_D / L and omega_pe x transit are NOT preserved: at factor s the scaled plasma has lambda_D / L larger "
                                                         "by sqrt(s) (2.0 / 2.83) and relatively thicker sheaths; 'as soon as surface processes and sources get important "
                                                         "the scaling ... deviates from the real solution'"},
                                "kind": "text", "source": "paper Pb_237 ('scaled down by a factor of four'); thesis ch. 7 ('scaled down by factor 8')"},
    "collisions": {"value": "electron-neutral elastic, ionisation, excitation (LLNL EEDL cross sections), Coulomb, charge exchange; thesis ch. 7 adds Xe2+ production",
                   "kind": "text", "source": "paper Pb_237; thesis ch. 7"},
    "anomalous_transport": {"value": {"model": "Bohm-type perpendicular diffusion D_perp = 0.4 k_B T_e / (e B) imposed by rotating the perpendicular velocity (guiding-centre shift; "
                                               "parallel speed unchanged); electrons selected at random at a rate set by the local |B|",
                                      "coefficient": 0.4, "provenance": "'derived from a 3D simulation of a similar thruster model' (Kalentev et al.)"},
                            "kind": "text", "source": "paper Pb_237"},
    "wall_model": {"value": {"dielectric": "surface charge accumulated and included in the Poisson solve with the dielectric constant; no explicit boundary condition",
                             "see": "50 % of impacting electrons re-emitted with 90 % of their incident energy", "ions": "absorbed; 'recycled' as neutrals conceptually (static background)"},
                   "kind": "text", "source": "paper Pb_237, Pb_239"},
    "electron_temperature_estimate_ev": {"value": 10.0, "kind": "text (thesis)", "source": "thesis ch. 7 ('estimated average electron temperature is 10 eV')"},
    "max_density_estimate_per_m3": {"value": 1.0e19, "kind": "text (thesis)", "source": "thesis ch. 7 ('estimated maximum plasma density in steady state is 1e19 m^-3')"},
}

# --------------------------------------------------------------------------------------------------------------------------
# Reported quantities D (the objects of the comparison); every entry: value, unit, kind, source, uncertainty components
# --------------------------------------------------------------------------------------------------------------------------

I_FEED_A = 1.1e17 * ELEMENTARY_CHARGE_C          # 17.6 mA equivalent of the 1.1e17 atoms/s feed

REPORTED: dict[str, dict[str, Any]] = {
    "anode_electron_current_a": {
        "value": 4.3e-3, "unit": "A", "kind": "text", "source": "paper Pb_240 ('the electron current at the anode is 4.3 mA, which is close to the measured value of 4.5 mA')",
        "experiment": 4.5e-3, "second_run": {"value": 4.7e-3, "source": "thesis ch. 7 ('The sum is 4.7 mA and in the experiment it is 4.5 mA')"},
        "u_d": [{"name": "stated_precision", "standard": 0.05e-3, "method": "half the last stated digit (4.3 mA)"},
                {"name": "reference_variability", "standard": 0.2e-3, "method": "half the spread between the paper (4.3 mA) and thesis (4.7 mA) runs of the same case"}],
    },
    "net_ionisation_fraction": {
        "value": 0.24, "unit": "fraction", "kind": "text / inferred", "source": "paper Pb_240 ('2.7e16 particles per second ... neutral gas influx of 1.1e17 ... ionization rate of 24 %'); "
                                                                              "Brandt equates the net ion production with the anode electron current (thesis ch. 7)",
        "experiment": 0.25, "second_run": {"value": 4.7e-3 / I_FEED_A, "source": "thesis ch. 7 anode current 4.7 mA / 17.6 mA"},
        "u_d": [{"name": "stated_precision", "standard": 0.005, "method": "half the last stated digit (24 %)"},
                {"name": "reference_variability", "standard": 0.5 * abs(4.7e-3 / I_FEED_A - 0.24), "method": "half the spread between the paper and thesis anode currents divided by e Q_in"}],
        "dependence": "not independent of anode_electron_current_a (same number divided by e Q_in); compared as a consistency row",
    },
    "ion_beam_current_a": {
        "value": 2.5e-3, "unit": "A", "kind": "text", "source": "paper Pb_241 ('overall ion beam current ... 2.5 mA, which is similar to the value of 3.1 mA derived from the experiment'); "
                                                              "thesis ch. 7 (2.5 mA) - summed over the domain's outer boundary cells",
        "experiment": 3.1e-3, "second_run": {"value": 2.5e-3, "source": "thesis ch. 7"},
        "u_d": [{"name": "stated_precision", "standard": 0.05e-3, "method": "half the last stated digit"},
                {"name": "boundary_sensitivity", "standard": 0.25e-3, "method": "DECLARED 10 %: Brandt states the Dirichlet -> Neumann outer boundary changed the plume current "
                                                                                "ratios and that the domain is 'still too small'; no number given"}],
    },
    "beam_fraction_of_feed": {
        "value": 2.5e-3 / I_FEED_A, "unit": "fraction", "kind": "inferred", "source": "I_beam / (e Q_in) = 2.5 mA / 17.6 mA",
        "u_d": [{"name": "propagated_from_beam_current", "standard": (0.05e-3**2 + 0.25e-3**2) ** 0.5 / I_FEED_A, "method": "beam-current budget divided by e Q_in"}],
    },
    "plasma_potential_near_anode_above_anode_v": {
        "value": 5.0, "unit": "V", "kind": "text", "source": "paper Pb_239-240 ('Near the anode the plasma potential is about 5 V (relative to the anode potential of 400 V)'); thesis ch. 7",
        "u_d": [{"name": "stated_rounding", "standard": 2.5, "method": "'about 5 V': half the rounding unit"}],
    },
    "potential_drop_first_cusp_v": {
        "value": 10.0, "unit": "V", "kind": "text", "source": "paper Pb_240 ('At the first magnetic cusp, the potential undergoes a drop of about 10 V'); thesis ch. 7",
        "u_d": [{"name": "stated_rounding", "standard": 2.5, "method": "'about 10 V': half of the 5 V rounding unit"}],
    },
    "potential_drop_second_cusp_v": {
        "value": 5.0, "unit": "V", "kind": "text", "source": "paper Pb_240 ('At the second cusp inside the channel, the potential drop is significantly lower, with only about 5 V'); thesis ch. 7",
        "u_d": [{"name": "stated_rounding", "standard": 2.5, "method": "'about 5 V': half the rounding unit"}],
    },
    "ion_density_typical_per_m3": {
        "value": 1.0e19, "unit": "m^-3", "kind": "figure / text", "source": "paper Pb_240 ('Ions have mostly a density of about 1e19 m^-3', Fig. 6 colour scale); thesis ch. 7 "
                                                                            "a-priori estimate 1e19; paper Pb_240 also states ~50x more neutrals than ions (-> ~4e18 at 2e20)",
        "u_d": [{"name": "figure_digitisation_log10", "standard": 0.18, "method": "log-scale colour map: factor 1.5 (0.18 dex) read-off"},
                {"name": "internal_inconsistency_log10", "standard": 0.2, "method": "'about 1e19' vs 'about 50 times more neutrals' (2e20 / 50 = 4e18): half the log10 spread"}],
        "log_scale": True,
    },
    "wall_ion_energy_max_ev": {
        "value": 160.0, "unit": "eV", "kind": "figure / text", "source": "paper Pb_240 ('cusp inside the acceleration channel close to the exit of the thruster, with ion energies up to 160 eV', Fig. 11)",
        "u_d": [{"name": "figure_digitisation", "standard": 16.0, "method": "10 % of the stated maximum (colour-scale read-off of Fig. 11)"}],
    },
    "wall_ion_current_density_max_a_per_m2": {
        "value": 640.0, "unit": "A/m^2", "kind": "figure / text", "source": "paper Pb_240 ('At this location one gets a current flux density of 640 A/m^2', Fig. 10)",
        "u_d": [{"name": "figure_digitisation", "standard": 64.0, "method": "10 % of the stated maximum (colour-scale read-off of Fig. 10)"}],
    },
    "plume_peak_angle_deg": {
        "value": 50.0, "unit": "deg", "kind": "figure / text", "source": "paper Pb_241 ('The maximum lies at about 60 degrees in the experiment and 50 degrees in the simulation', Fig. 12, 5 deg bins)",
        "experiment": 60.0, "second_run": {"value": 60.0, "source": "thesis ch. 7 ('the maximum is at higher angles at about 60 degrees in the experiment and in the simulation', Fig. 7.6)"},
        "u_d": [{"name": "bin_half_width", "standard": 2.5, "method": "5 deg angular bins"},
                {"name": "reference_variability", "standard": 5.0, "method": "half the spread between the paper (50 deg) and thesis (60 deg) runs"}],
    },
    "electron_energy_near_exit_cusp_ev": {
        "value": 200.0, "unit": "eV", "kind": "figure / text", "source": "paper Pb_240 ('a high energy region of around 200 eV near the exit cusp', Fig. 8)",
        "u_d": [{"name": "stated_rounding", "standard": 50.0, "method": "'around 200 eV': 25 %"}],
    },
}

QUALITATIVE: dict[str, dict[str, str]] = {
    "flat_interior_potential": {"statement": "potential mostly flat throughout the channel close to the anode potential; main drop at the exit cusp beyond the dielectric 'anchor' (a bulge)",
                                "source": "paper Pb_238-240, Fig. 3 and 5"},
    "cusp_potential_dips": {"statement": "near the cusps, close to the dielectric surface, a potential dip of 'several 10 V'", "source": "thesis ch. 7"},
    "ionisation_structure": {"statement": "ionisation close to the axis and in the cusps, a peak UPSTREAM of each cusp, and at the exit", "source": "paper Pb_240, Fig. 7"},
    "wall_flux_localisation": {"statement": "wall particle flux and mean ion energy maxima at the cusp locations; erosion practically zero elsewhere", "source": "paper Pb_240, Fig. 10-11"},
    "sheath": {"statement": "radial sheath at the dielectric 5-10 Debye lengths wide", "source": "paper Pb_239"},
    "density_fill": {"statement": "electron density fills the channel more uniformly (radially) than in the large HEMPT; ions follow the electrons", "source": "paper Pb_239"},
}


def reported_value_and_u_d(quantity_id: str) -> tuple[float, float, list[dict[str, Any]]]:
    """(D, combined standard u_D, components) of a reported quantity (root-sum-square of the components)."""

    row = REPORTED[quantity_id]
    components = row["u_d"]
    u_d = sum(float(c["standard"]) ** 2 for c in components) ** 0.5
    return float(row["value"]), u_d, components


def reference_document() -> dict[str, Any]:
    """The reference record written into the experiment's protocol.json and README."""

    return {
        "citation": CITATION, "doi": DOI, "open_access_pdf": OPEN_ACCESS_PDF, "thesis": THESIS, "verification": VERIFICATION,
        "why_this_reference": WHY_BRANDT, "alternatives_considered": ALTERNATIVES_CONSIDERED,
        "setup": SETUP, "reported": REPORTED, "qualitative": QUALITATIVE, "feed_current_equivalent_a": I_FEED_A,
    }


__all__ = ["ALTERNATIVES_CONSIDERED", "CITATION", "DOI", "I_FEED_A", "OPEN_ACCESS_PDF", "QUALITATIVE", "REPORTED", "SETUP", "THESIS", "VERIFICATION", "WHY_BRANDT",
           "reference_document", "reported_value_and_u_d"]
