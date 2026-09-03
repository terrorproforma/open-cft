# References and Provenance

## Primary publication

Angus Muffatti and Hideaki Ogawa, “Multi-objective Design Optimisation of a
Small Scale Cusped Field Thruster for Micro-satellite Platforms,”
`ISTS 2017-b-32`, received 16 April 2017.

- Official archive:
  https://archive.ists.ne.jp/upload_pdf/2017-b-32.pdf
- Retrieved for this audit: 1 September 2026
- Retrieved file size: 646,858 bytes
- SHA-256:
  `E28B216E8652416768760A4338CDC4488493B10ED28FCBED6F6E4DD5E51ED4C6`
- The PDF is not vendored in this repository; the URL and digest identify the
  evidence reviewed.

The paper cites Kornfeld, Koch, and Harmann, “Physics and evolution of
HEMP-thrusters,” IEPC 2007, as the source of the 28-equation one-dimensional
model. The ISTS paper describes that model but does not publish the complete
equation set, so it cannot independently settle every executable residual sign
in `Power_B_EQs.m`.

## Evidence classes

- **Code fact:** directly observable in the preserved `FYP/*.m` snapshot.
- **Paper claim:** explicitly printed in ISTS 2017-b-32.
- **Cross-source discrepancy:** paper and snapshot differ; neither is silently
  treated as the publication-run source.
- **Unverified inference:** plausible explanation requiring archived run data,
  original optimizer, or another primary source.

## Equation traceability

- Paper Eq. (3), `P = Phi_a * I_a`, supports
  `Performance_est.m:140` and the modern `anode_power_w` calculation.
- Paper Eqs. (5), (19), (20), and (21) support the structure of beam/grid/total
  efficiency and thrust post-processing in `Performance_est.m:134-139`.
- Paper Eq. (7) supports the singly charged xenon mass-utilization structure.
  Paper §2.5 says it cannot exceed 1, while `Performance_est.m:42` permits 1.2.
- Paper Eqs. (14)-(17) derive the mirror ratio, acceptance angle, and angular
  arrival probability used by `cusp_prob.m:186-190`. The modern analytic and
  Warp kernels implement only this independently testable relation.
- Paper §2.1 states there are 28 simultaneous equations but defers their full
  definition to Kornfeld et al. It does not validate the snapshot's 33
  continuous residuals, appended logical comparisons, or disputed `CE` signs.
- Paper Eqs. (22)-(23) state Sobol sensitivity measures. No sensitivity
  implementation is migrated because the surrogate validity and published
  interpretation conflict.

The accepted L0 implementation has a separate machine-readable equation and
evidence ledger at `../spec/physics/equation-ledger.json`. Its assumptions and
permitted claims are documented in `workstreams/physics-foundation.md`. The
2020 values under `../spec/physics/external-regression-fixtures.json` are
cross-model fixtures only, not fitted truth.

The optimization policy source of truth is
`../spec/optimization/campaign-v1.json`; architecture and known limitations are
in `workstreams/optimization-architecture.md` and
`workstreams/optimization-workstream-report.md`. Its benchmark `results`
remains null. `FIRST_RESULTS.md` records only the L0 numerical sweep and does
not fill or imply optimization benchmark results.

## Published 2020 cross-model evidence

Yeo et al., “Multiobjective Optimization and Particle-in-Cell Simulation of
Cusped Field Thrusters for Microsatellite Platforms,”
DOI [10.2514/1.A34584](https://doi.org/10.2514/1.A34584), reports the S1 rows
`MDO (original)`, `PIC`, and `MDO (modified)`. The source-native labels and
values are preserved in
`../data/validation/yeo-2020-s1-external-evidence-v2.json` and the authorized
shared physics fixture uses `YEO2020-S1-MDO-ORIGINAL`. “Corrected
low-fidelity” is retained only as an editorial interpretation, not substituted
for the publication's model label.

These records are external model outputs for cross-model comparison. They are
not experimental truth, calibration data, or acceptance tolerances. The
validation workstream reserves predictive-validity authority for provenance-
and uncertainty-complete experiments.

## Accepted foundation evidence boundaries

The L1a field equations, artifacts, and verification record are under
`../spec/fields/`, `../examples/axisymmetric/`, and
`workstreams/axisymmetric-workstream-report.md`. L1a means linear-vacuum,
equivalent-current, finite-box FDM—not FEM or material-aware production fields.
Global plasma, hybrid, and PIC reports document numerical foundations rather
than accepted predictive L2/L3 CFT models. The surrogate benchmark report
records failed quality gates, so it is not a successful regression oracle.
The manuscript checker under `../../paper/` keeps L1--L3 result gates closed
until accepted committed manifests exist.

## Publication/snapshot reconciliation

- **Objectives:** paper abstract, §§1, 2.3, and 4 consistently describe three
  maximized objectives: thrust, efficiency, and specific impulse. Code fact:
  `CFTOpt.m:4` and `buildSurrogates.m:18-19` use four objectives, adding anode
  power; `plotParetoOptwithColourDots.m:25,31` instead plots three and colors by
  power. This likely represents different analysis snapshots, but the exact
  publication-run revision is unavailable.
- **Generations:** paper §§2.3, 3, and 3.2 report population 96 for 100
  generations and 8,975 feasible solutions. Code fact: `params.m:12` uses 50
  generations and plotting/sensitivity filenames use generation 50.
- **Decision variables:** paper repeatedly claims eight variables, but §2.3
  fully lists only `Phi_a`, `I_a`, mass flow, IMR, and OMR; Tables 2-4 and the
  sensitivity discussion report those five. Code fact: `CFTOpt.m:2,9-19`
  defines all eight, with three additional shield/enclosure radii. The missing
  publication values prevent reconstruction of the reported designs.
- **Surrogates:** paper §2.3 says surrogate substitution requires every
  objective/constraint MSE to be within 5%. Section 3 then says none of the
  surrogate methods achieved a 5% margin, yet §§2.4 and 3.1 report Sobol
  sensitivity derived through those surrogates. Code fact:
  `SensitivityAnalysis_Surr_rev.m:16-22` loads a saved surrogate and runs
  sensitivity without a quality gate. The published Sobol results are
  exploratory, not a validated regression oracle.
- **Sensitivity interpretation:** the abstract says anode current has the
  greatest influence on all three objectives. Section 3.1, Tables 2-3, and the
  conclusion instead identify mass flow as greatest for all three. The table
  values support the latter statement.
- **S1 power:** Table 4 reports `Phi_a=990.6 V`, `I_a=3.30 A`, implying
  `3268.98 W` from Eq. (3). Section 3.2 states `3466 W`. Its nearby ratios are
  consistent with approximately 3269 W, so 3466 W is most likely a prose typo,
  but no source dataset is available to confirm.

## Sheath-closed four-cell model (v2, development) sources

The v2 development model `cft_revival.plasma_v2`
(`spec/plasma_v2/four-cell-sheath-closure-v2.json`,
`workstreams/plasma-v2-formulation.md`) is built on the v1 ledger plus the
following sources. Full records, verification tags and the numbered
bibliography are in `literature/reduced-models-cusp-topology-blockers.md`
(entries 1, 2, 4, 18, 44, 54, 57, 58, 59); none of these is truth, and the
two published four-cell states are reproduction targets only.

- Kornfeld, G., Koch, N., Harmann, H.-P., "Physics and Evolution of
  HEMP-Thrusters", IEPC-2007-108 (2007), Table 3.1: DM9.2 and DM10 4-stage
  columns at 1 kV / 1 A, including the printed cusp potentials (which cancel
  from every printed equation) and component powers (sum 1005.9 W and
  1003.9 W against 1000 W). Retrieved 2026-09-03 from
  https://electricrocket.org/IEPC/IEPC-2007-108.pdf; text extracted with
  `pdftotext -layout`. Transcribed in `cft_revival.plasma_v2.targets`.
- Puca, N., Panelli, M., Battista, F., "A Methodology for the Preliminary
  Design of a High-Efficiency Multistage Plasma Thruster", *Aerotecnica
  Missili & Spazio* 103(4), 321-338 (2024),
  DOI [10.1007/s42496-024-00203-x](https://doi.org/10.1007/s42496-024-00203-x).
  Table 1 (33-equation GA minimum for DM9.2 and DM10; cathode current an
  input, so `j_e0` is not printed) and Table 3 (Goebel-type cusp model:
  hybrid loss area `sqrt(r_e r_i) L_c`, Boltzmann factor `exp(-q phi_s/kT_e)`,
  leak-width prefactor 1). Tables read from the publisher's table pages on
  2026-09-03. Transcribed in `cft_revival.plasma_v2.targets`.
- Lieberman, M. A., Lichtenberg, A. J., *Principles of Plasma Discharges and
  Materials Processing*, 2nd ed., Wiley (2005), eq. (6.2.17): floating
  potential `T_e ln[(M/(2 pi m_e))^(1/2)]` (rows R28-R30, R31).
- Hobbs, G. D., Wesson, J. A., "Heat flow through a Langmuir sheath in the
  presence of electron emission", *Plasma Physics* 9, 85-87 (1967),
  DOI [10.1088/0032-1028/9/1/410](https://doi.org/10.1088/0032-1028/9/1/410):
  emission-corrected floating potential and the space-charge limit
  (~1.02 T_e, `gamma_crit = 1 - 8.3 (m_e/M)^(1/2)`).
- Goebel, D. M., Katz, I., *Fundamentals of Electric Propulsion: Ion and Hall
  Thrusters*, Wiley (2008), Ch. 4: ring-cusp discharge model, hybrid
  gyroradius loss area, electron energy to the wall `2 T_e + phi_s` (CL-4;
  Maxwellian diagnostics). Hershkowitz, N., Leung, K. N., Romesser, T.,
  *Phys. Rev. Lett.* 35, 277 (1975): leak widths of order four hybrid
  gyroradii (prefactor range 1-4).
- Koch, N. et al., IEPC-2011-236 (2011), finding (ii), and Brandt, T. et al.,
  *Trans. JSASS Aerospace Tech. Japan* 14, Pb_235 (2016),
  DOI [10.2322/tastj.14.Pb_235](https://doi.org/10.2322/tastj.14.Pb_235):
  flat interior potential with one exit drop and ~10 V / ~5 V internal
  steps - the declared `CL-3-potentials` closure.
- Model-to-model context only: `experiments/pic2d_cft_steady_state_v2`
  (development plateau, 300 V / 3.44 mA) and `experiments/cft_orbit_wall_
  loss_v4` (collisionless geometric access fraction 0.6445 on the same P2
  field; screening label).
