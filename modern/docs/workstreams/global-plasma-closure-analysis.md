# Four-Cell Discharge Closure Analysis for Nonzero Cusp Probabilities

Status of the correction proposed here: `PROPOSED_NOT_ACCEPTED`
(`spec/plasma/equation-ledger.json#global_row_consistency`). The executable
ledger in `cft_revival.plasma` and `cft_revival.plasma_network` is unchanged
except for the solver projection fix described in Section 7.

Date of analysis: 2026-09-03. Branch `fix/plasma-network-closure`.

## 1. Finding under investigation

The MDO L0 campaign v1 probe (2026-09-03 08:20 AEST, recorded in
`experiments/mdo_l0_campaign_v1/DEVLOG.md` and `protocol.json#prior_model_disclosure`)
ran `solve_global_discharge_multistart_cpu` (`start_count=3`,
`residual_tolerance=1e-8`) over `Ua in {150, 300, 500, 1000} V`,
`Ia in {0.1, 0.5, 1, 3} A` and five cusp-probability vectors

```
p_A = (0, 0, 0, 0)
p_B = (0.06, 0.119, 0.16, 0.254)      # Kornfeld DM9.2
p_C = (0.3, 0.3, 0.3, 0.3)
p_D = (0.64, 0.64, 0.64, 0.64)
p_E = (0.05, 0.64, 0.64, 0.0)
```

and reported 13/80 closures, all at `p_A`, with every nonzero `p` ending at
the iteration limit and residual floors `4.7e-4 .. 0.196`.

## 2. Exact reproduction

The probe was re-run with the exact call (system Python 3.12, pure Python
solver, CPU). Result: **13 of 80 converged, 197 s total (2.9 s per failing
multistart solve, 0.3-2 s per closing one)**, floors `4.739e-4 .. 1.960e-1`,
identical to the campaign disclosure. Per-case residual vectors were recorded
(`normalized_residuals`, 28 rows).

Dominant row at the floor (index of the largest normalized residual):

| `p` | 150-500 V | 1000 V |
| --- | --- | --- |
| `p_B` (DM9.2) | R27 (global power) | R27 |
| `p_C` | R27 (R11 at 150 V / 3 A) | R27 |
| `p_D` | R06 / R11 (current rows) | R27 (R23 at 0.5 A) |
| `p_E` | R06 (R27 at 500 V) | R27 |

Two facts not visible in the campaign summary:

- Three of the sixteen `p_A` cases also failed to close (1000 V at 0.1, 1
  and 3 A; floors `3.5e-4`, `2.1e-4`, `7.6e-6`; reasons
  `step_tolerance_without_balance` / `iteration_limit`; dominant row R07).
  These are a solver defect (Section 7), not a model property.
- At every `p != 0` floor point the inequality `phi_4 >= Ua` is active
  (`phi_4 - Ua = 0` exactly) and the LS spread over rows R13/R14/R16/R25/R26
  compensates R27.

## 3. Numerical diagnosis

### 3.1 Jacobian at the floor points

The lowest-cost vector of every attempt was recorded and the scaled analytic
Jacobian (`Ua` for potentials/temperatures, `Ia` for currents) decomposed by
SVD:

- rank 22 of 25 at every floor point and at every closed point (three
  singular values at `1e-16`, the known 3-dimensional potential null space);
- condition of the independent subspace 9-200 (well conditioned; 1000 V is
  the worst at 65-130);
- `|J^T r|_inf` at the `p != 0` floors is `0.8-1.6` times the floor itself,
  i.e. the floor is **not** an unconstrained least-squares stationary point:
  the gradient points to lower `phi_4`, and the admissible region blocks it.

Conditioning does not explain the failure.

### 3.2 Continuation in `p` (300 V, 1 A, 5 starts, 600 iterations)

| pattern | `eps = 1e-4` | `1e-3` | `1e-2` | `3e-2` | `1e-1` | `3e-1` |
| --- | --- | --- | --- | --- | --- | --- |
| `p = eps*(1,1,1,1)` floor | 1.28e-6 | 1.28e-5 | 1.30e-4 | 3.99e-4 | 1.43e-3 | 5.78e-3 |
| `p = eps*(1,1,1,0)` floor | 1.31e-6 | 1.29e-5 | 1.29e-4 | 3.98e-4 | 1.44e-3 | 5.95e-3 |
| `p = (0,0,0,eps)` | closes | closes | closes | closes | closes | closes |

The floor is linear in `eps` from `1e-4` upward with the dominant row R27
throughout. No solution branch exists for any `eps > 0` with an interior
component; nothing is "lost" by the solver. Loss at the anode cusp alone
(`p_4`) closes to `1e-14` at every `eps`.

### 3.3 Global search

- Differential evolution over the full 25-value box (DM9.2 `p`, 300 V, 1 A,
  205 312 evaluations): best `max|r| = 2.06e-2`, worse than the LM floor.
- 200 random feasible starts through the production LM: 0/200 closed; floors
  `1.82e-3` (min) / `2.00e-3` (median) / `8.25e-2` (max).

### 3.4 Relaxing the admissible region

Dropping only `phi_4 >= Ua` admits exact roots (`max|r| ~ 1e-16`) with
`phi_4` below the anode potential by 0.6-12.6 V (DM9.2 `p`: 1.18 V at
300 V/1 A, 0.63 V at 1000 V/1 A). They are rejected by `is_feasible`
(margin 4 negative) and are compensating-error roots (Section 5).

## 4. Analytic structure: the 27-row manifold and the global row

Write `Je_k = j_e,k (1-p_k)` for the electrons that survive cusp `k`,
`L_k = j_e,k p_k` for the electrons lost to it, `dE_k` for the cell energy
gain, and use the ledger row ids of `spec/plasma/equation-ledger.json`.

Rows R00-R26 are mutually consistent and determine every state value from the
four potentials (`potential_parametrized_state`):

```
R00        j_e0 = p0 (phi_1 - phi_0)^{3/2}
R04-R07    I_k  = Je_k CI dE_k / EI
R23-R26    T_k  = [(1-CE) Je_k dE_k - I_k EI] / (Je_k + I_k)
R01-R03    j_e,k+1 = Je_k + I_k                      (k = 0..2)
R11+R18+R19  j_e4 = j_e3 + I_4                       (all cell-4 electrons reach the anode)
R15-R19    j_i,k = Ia - j_e,k
R20-R22    j_ic,k = L_k
R08-R10, R12-R14 are then satisfied identically.
```

(Verified: 200 random `(Ua, Ia, p, phi)` give `max|R00..R26| < 1e-11`.)

Substituting into R27 with the executable power expressions

```
Pb   = j_i3 phi_4 + sum_{k<3} (j_i,k - j_i,k+1) phi_k+1
PI   = EI sum I_k
PE   = CE sum Je_k dE_k
Pcusp = sum_{k=1..3} L_{k-1} (dE_k + EI)
Panode_e = L_3 (Ua - phi_3 + T3) + (Je_3 + I_4)(phi_4 - Ua + T4)
Panode_i = -j_i4 (phi_4 - Ua)
```

every potential, temperature and ion-current term cancels except

```
R27 = 2 (Je_3 + I_4)(phi_4 - Ua) + EI (p1 j_e0 + p2 j_e1 + p3 j_e2).        (*)
```

`global_row_closed_form` implements (*); it agrees with the evaluated R27 to
`1.9e-13` relative over 400 random cases.

Consequences of (*):

1. Inside the admissible region (`phi_4 >= Ua`) both terms are non-negative.
   The global row can vanish only when `p1 = p2 = p3 = 0` **and**
   `phi_4 = Ua`. `p_4` is unrestricted. This is exactly the observed closure
   set (all `p_A` closures have `phi_4 = Ua` to `1e-9`; `(0,0,0,eps)` closes
   for every `eps`).
2. For any positive interior probability the residual on the exact manifold
   is `EI sum p_k j_e,k-1 / (Ua Ia) > 0`; the production LS floor is smaller
   (it spreads the misfit over rows R13/R14/R25/R26) but cannot reach zero.
   For DM9.2 `p` at 1000 V/1 A with the published potentials it is `1.47e-3`,
   which is the R27 misfit `1.49e-3` the ledger already records for the
   rounded DM9.2 table; the `4.89e-4` "residual floor" of the 500-iteration
   DM9.2 probe (`evidence_comparisons`) is the same inconsistency.
3. The infimum of (*) over the *feasible* manifold (`j_e4 >= Ia` through
   `j_i4 <= 0`) is `4.45e-3` (300 V/1 A), `3.19e-4` (1000 V/1 A) and
   `1.47e-2` (150 V/0.1 A) for DM9.2 `p`, and the feasible manifold is
   **empty** for `p_D` at 150-500 V and `p_E` at 150-300 V: with three
   cusps each losing 64 % the electron cascade cannot reach `j_e4 >= Ia`.
   Those cases fail on current rows (R03/R06/R07/R11) before the power row
   matters. This is a second, independent infeasibility of the prescribed-
   current model at large interior loss and low voltage.

## 5. Where the inconsistency comes from

Kornfeld, Koch and Harmann, IEPC-2007-108, Section III.B (text extracted from
the published PDF on 2026-09-03):

- Assumption 8: "Ionization and excitation losses are considered as frozen
  losses not contributing to the thermal load of the thruster, **except the
  recombination losses at boundaries**."
- Global balance: `Ua Ja = Pb + IL + EL + CL (electronic-, ionic- and
  recombination) + AL (electronic- ionic and recombination)`, with
  `CL = je0 p1 (phi_c1 - phi_0 + phi_1 - phi_c1 + IE) + ...`.

So the source books the ionization energy of every ion once in `IL` (frozen
loss) and again as recombination heat in `CL` for the ions that reach the
cusp. Against the electrical input `Ua Ia` each joule can be spent once: the
recombination heat *is* the frozen loss reappearing at the wall. The
executable `Pcusp` inherits the `+EI` (legacy `FYP/Power_B_EQs.m` line 137
carries the same terms), which is the second term of (*).

The first term of (*) comes from the printed anode electron term
`(I4 + je3 (1-p4)) (phi_4 - Ua + T4)` together with the restriction
`phi_4 > Ua`. An electron that leaves the cell-4 plasma at `phi_4` and
arrives at the anode at `Ua` deposits `T4 + (Ua - phi_4)`; the printed sign
is that of an ion. The same ledger uses the electrostatically consistent
signs for the direct-cusp electrons (`Ua - phi_3 + T3`) and for the anode
ions (`-j_i4 (phi_4 - Ua)`). With the printed sign the global row retains
`2 (Je_3 + I_4)(phi_4 - Ua)`, which is why Kornfeld's own DM9.2 solution has
`phi_2 = phi_3 = phi_4 = 1000 V = Ua` and why every rebuilt closure sits on
the `phi_4 = Ua` boundary. (The paper reports that MathCAD's root finder
"did mostly not work for the 28 equation problem" and that the minimum-error
solver was used with "power accuracy within 0.5 %"; the DM9.2 column sums to
1005.9 W against 1000 W.)

The relaxed-constraint roots of Section 3.4 satisfy
`2 (Je_3 + I_4)(phi_4 - Ua) = -EI sum p_k j_e,k-1`: a negative anode-fall
term of the wrong sign cancels a double-counted recombination term. They are
not physical solutions.

The 2026-09-01 audit corrections (signed terminal row `j_i3 - I4 - j_i4 = 0`,
fourth-cell transport `T4 (j_e3(1-p4) + I4)`, excitation as a loss, cusp
current subtracted in R08-R10) cancel exactly in the substitution and do not
appear in (*). They did not introduce the inconsistency; the legacy system was
additionally inconsistent in the current rows (`+jic`, 0.204 A miss).

## 6. Classification

- **(a) Genuine inconsistency of the corrected equation set for interior
  `p != 0`.** Primary cause. No admissible root exists; the honest residual
  floor reported by the solver is the correct outcome.
- **(d) Sub-region with solutions:** exactly `p1 = p2 = p3 = 0`, any `p4`,
  `phi_4 = Ua`. Additionally, the prescribed-current model has no feasible
  state (independent of R27) for large interior loss at low voltage
  (Section 4, item 3).
- **(b) Solver defect, secondary:** the `sorted()` ordering projection
  stalled 3/16 zero-cusp cases at 1000 V. Fixed (Section 7). It does not and
  cannot change the `p1..p3 > 0` outcome.
- **(c) Caller/units:** none. The probe called the public API correctly.

## 7. Actions taken on this branch

1. `cft_revival.plasma.solver.project_nondecreasing`: Euclidean projection
   onto the nondecreasing cone (pool adjacent violators). `sorted()` permutes
   variable identities, so an LM step that lowers `phi_k` below `phi_{k-1}`
   became a step on `phi_{k-1}`; damping then grew to `1e10` and the solve
   stopped on the step tolerance with `phi_3 = phi_4` pinned. Both
   `solve_global_discharge` and `plasma_network.solver._projection` now use
   the isotonic projection. Zero-cusp probe grid: 16/16 close (was 13/16);
   every previously closing case still closes; existing suites unchanged.
2. `cft_revival.plasma.residuals.potential_parametrized_state` and
   `global_row_closed_form`: diagnostic parametrization of the R00-R26
   manifold and the closed form (*). They evaluate; they never publish.
3. Tests pinning the behaviour: `tests/plasma/test_closure_p_nonzero.py`
   (manifold exactness, closed form, DM9.2 misfit magnitude, anode-only
   closure, linear-in-`eps` misfit, no admissible root for DM9.2 `p`,
   compensating-error root rejected, ledger flag),
   `tests/plasma_network/test_plasma_network_closure_p_nonzero.py`
   (N=4 network inherits both behaviours),
   `tests/plasma/test_solver.py` (projection properties, zero-cusp grid
   including the 1000 V cases, determinism).
4. Ledger entry `spec/plasma/equation-ledger.json#global_row_consistency`
   with status `PROPOSED_NOT_ACCEPTED`; the 28 rows and 7 power expressions
   are unchanged.

## 8. Proposed minimal correction (not accepted)

```
Pcusp'    = j_e0 p1 (phi_1 - phi_0 + T0) + j_e1 p2 (phi_2 - phi_1 + T1)
          + j_e2 p3 (phi_3 - phi_2 + T2)                      # drop +EI
Panode_e' = j_e3 p4 (Ua - phi_3 + T3) + (I4 + j_e3 (1-p4)) (Ua - phi_4 + T4)
Panode_i  = -j_i4 (phi_4 - Ua)                                # unchanged
```

Justification: `Pcusp'` is the electron plus ion kinetic energy delivered to
the dielectric cusp (the cusp potential cancels as in the source); the
recombination energy remains booked once in `PI`. An equivalent alternative
that keeps the cusp *thermal load* explicit is to retain `+EI` in `Pcusp` and
book `PI' = EI (sum I - sum jic)` as the frozen loss carried by the beam.
`Panode_e'` restores electrostatic consistency with the other two anode terms;
the nonnegative-loss margin then reads `T4 >= phi_4 - Ua`.

Consequence: with both corrections R27 is implied identically by R23-R26 and
charge conservation for every `p` (the substitution of Section 4 gives zero).
The global row becomes a verification identity, the structural rank drops
from 22 to 21 (nullity 4: all four potentials free), and the model needs an
independent potential closure before any state can be published. Kornfeld's
assumption 2 ("cusp potentials determined self-consistently by the dielectric
zero-net-current condition") was meant to supply it; the cancelled cusp
potentials and the missing sheath relation are already listed as an evidence
blocker in `global-plasma-formulation.md`. Accepting the corrections without
that closure would replace a non-closing model by an underdetermined one,
which is why the status is `PROPOSED_NOT_ACCEPTED`.

## 9. Implications for the MDO campaign

The MDO v1 disclosure ("the 28-by-25 system closed only for `p = (0,0,0,0)`;
not used as an evaluation chain") remains historically correct. The solver
fix does not make the solver usable for interior cusp probabilities: a v2
campaign could use it only for anode-cusp-only loss (`p1 = p2 = p3 = 0`) or
after the correction above is accepted together with a potential closure.

## 10. Reproduction

Scratch scripts (not committed) ran from `%TEMP%\plasma_closure`:
`probe_repro.py` (exact probe), `manifold_analysis.py` (closed form,
Jacobian SVD), `floor_jacobian.py` (floor-point Jacobians),
`continuation_global.py` (continuation, relaxed root, DE, 200 starts),
`manifold_infimum.py` (feasible-manifold infimum), `stall_diagnosis.py` and
`projection_fix_trial.py` (1000 V stall). All numbers above are reproduced by
the committed tests where they are pinned.
