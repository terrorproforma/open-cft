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
