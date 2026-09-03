# plasma-v2 devlog

## 2026-09-03 - sheath-closed four-cell power balance (development)

- Branch `feat/plasma-network-v2-sheath` (worktree `uni-project-plasma-v2`)
  from `origin/feat/sota-foundation` at `b6bb6215`.
- New package `src/cft_revival/plasma_v2` (constants, models, residuals,
  manifold, solver, targets, pic_context, verification). The hash-bound v1
  package `cft_revival.plasma` is imported read-only: rows R00-R26 are
  re-implemented with forward-mode duals and tested for round-off parity
  (2.3e-13) against `evaluate_plasma_residual_cpu`;
  `potential_parametrized_state`, `project_nondecreasing`,
  `solve_bounded_least_squares`, `representative_initial_state`,
  `SolverOptions` are called directly.
- Rows: R27 with the two `PROPOSED_NOT_ACCEPTED` corrections (identity on the
  manifold, 4.8e-15 relative over 200 seeded states); R28-R30 floating
  dielectric sheath `Dphi_s,k = c_s,k T_k` (L&L 6.2.17 / Hobbs-Wesson 1967;
  regimes no-emission 5.27, with-emission, space-charge-limited 1.02); R31
  anode electron-collecting sheath (or declared fall); R32-R34 declared
  potential relations (`CL-3-potentials`); R35-R37 cusp-loss closure
  (CL-1 declared, CL-3 `A_k exp(-Dphi/T)`, CL-4 hybrid-area with prefactor).
- Rank at the Kornfeld point: corrected core 21/25, with sheath+anode rows
  28/31, with the declared rows 31/31. Sheath rows identify only their own
  unknowns; R31 identifies one potential relation; three are declared.
- Fail-closed policy extended by the cusp electron wall energy margin
  `dE_k - Dphi_s,k >= 0`; closed form `c_s <= (1 + I_k/Je_k)/CT` derived and
  tested (no-emission sheath fails for `dE_k < 447 V`; SCL always passes).
- Verification record `docs/workstreams/plasma-v2-verification.json`
  (184 s): Kornfeld Table 3.1 DM9.2 reproduced in mode C (their potentials)
  to <= 0.05 eV / 0.006 A and in mode A (`phi_1` solved = 14.07 V vs 14.1);
  DM10 mode A `phi_1` 11.5 vs 12.3 V; Puca 2024 Table 1 states are far from a
  root (R00/R15 at 0.5 normalized, `j_e0` derived); closure grid 192 solves:
  CL-3 SCL 73/80, CL-1 SCL 16/16, no-emission 0/96 (all blocked by the
  energy margin); CL-4 prefactor sweep closes only at 1 kV/1 A, c = 1 with
  PIC segment-mean densities; PIC v2 plateau context table (staircase
  potentials 94/55/125 V, near-wall drops 32-39 V at 5.7-6.4 eV, wall
  currents per cusp, anode ion fraction 1.42 % vs R31-implied fall 11.8 V).
- Spec `spec/plasma_v2/four-cell-sheath-closure-v2.json`; formulation
  `docs/workstreams/plasma-v2-formulation.md`; REFERENCES.md section added.
- Validation run: `tests/plasma_v2` 53 passed; `tests/plasma` 58 passed
  (unchanged); `tests/plasma_network` 66 passed; `python paper/scripts/
  check_paper.py` (see final devlog line); v1 files hash to the manifest.
- Risks / follow-ups: potential steps are declared (PIC shows a staircase);
  sheath temperature = cell temperature (PIC near-wall 6 eV vs axis 40 eV);
  CL-4 needs a cusp sheath-edge density; mode B admissible band is a hairline
  around the mode-A coupling; no re-admission attempted.
