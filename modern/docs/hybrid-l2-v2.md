# Hybrid L2 v2 - the per-cell hybrid between the 0-D network and the PIC

Status: DEVELOPMENT model, preregistered comparison against the accepted PIC steady-state v2 base
plateau (`experiments/pic2d_cft_steady_state_v2/results`, I_d 3.44 mA, channel-only, model v1.3).
Code: `cft_revival.hybrid` (`cells`, `rates`, `pb_solver`, `ions`, `l2`, `checkpoint_v2`, `gates`);
experiment: `experiments/hybrid_l2_v2`; dashboard: `visualization/hybrid-l2-v2.html`.
This document is the predeclaration: the cell concept, the equations, what is predicted, what is
taken from the PIC, the six GATE-L2 gates made concrete with their tolerances, and the claim boundary.
It was written before the preregistered run; section 9 records what the non-evidentiary shakedowns
showed and how the design responded, so the reader can see what was fixed before freezing and what
was not.

## 1. Why v2, and what changed since v1

The v1 slice (`hybrid-formulation.md`) advanced prescribed-field Cartesian macroparticles with
synthetic collisions; its roadmap listed six unimplemented gates and the stop note said the model had
"nothing to model" because the preregistered topology nulls had not found wall-cusp cells. Both
blockers are gone:

* cusp topology search v3.1 (`experiments/cusp_topology_search_v3_1`, literature definition
  axis-null -> separatrix -> wall intersection) catalogued the cells of 281 designs; for the P2
  reference design it finds three wall cusps at **6.028 / 12.000 / 17.972 mm**, the planes the PIC's
  ionisation "flames" and wall fluxes sit on;
* the PIC steady-state v2 base plateau is accepted and its convergence pairs (seed-b, W x 0.7) give
  particle bands; the design mini-sweep (`pic2d_design_mini_sweep_v1`) defines the per-cell / per-cusp
  extraction (`closure.extract_targets`) that maps PIC maps onto plasma-network v2 parameters.

L2 v2 therefore uses the catalogue cells as its control volumes and the mini-sweep's extraction as the
common language between L2 and the PIC.

## 2. The redefined cell concept

A **cell** is a catalogue cell of the v3.1 topology search: the wall interval between two consecutive
separatrix-wall intersections (plus the anode-side and exit-side partial cells). For the reference
design `p2_divergent_exit:divergent-exit-stack` the catalogue gives `cell-01` (anode partial,
1.0-6.028 mm), `cell-02` (6.028-12.000), `cell-03` (12.000-17.972), `cell-04` (exit partial,
17.972-18.0 mm). The catalogue covers the straight dielectric only; L2 extends the end cells to the
electrodes (injector zone 0-1 mm -> anode cell; divergent cone 18-24 mm -> exit cell), exactly as the
mini-sweep's Kornfeld mapping does. `cells.load_reference_partition` verifies the catalogue bytes
against the sealed bundle manifest and refuses a catalogue whose planes differ from the declared PIC
planes by more than half a 50 um cell (`tests/hybrid/test_l2_cells_and_rates.py`).

Inside a cell the electrons are one fluid: one count `N_k`, one thermal energy `W_k` (so one
temperature `T_k`), one Boltzmann reference `C_k`. Between cells the electrons cross the cusp only
through an effective conductance (section 3.4). Electrons occupy only the **populated flux tubes** of
the cell (section 3.3): the tubes whose footprint on the dielectric lies inside the cusp's leak window.
The wall-to-wall arcs between cusps carry no electrons - the PIC shows n_e falling by 10^2-10^3 in the
last 0.5 mm before the wall between cusps and the wall floating *above* the adjacent plasma.

## 3. The model

Grid, mesh and Poisson operator are the PIC's (`pic2d.models.Grid2D`, `pic2d.mesh.build_mesh_masks`,
the finite-volume Gauss law of `pic2d.poisson`), on the PIC base plateau's own grid (60 x 480, 50 um)
for the headline case. The magnetic field is the PIC's bound P2 node sample
(`fields.build_p2_psi_field` + `sample_field_map`, `field_source_sha256` identical to the base plateau's
on the same grid). Cross sections are the PIC's hash-bound LXCat tables (`XenonCrossSections`).

### 3.1 Ions (kinetic)

Xe+ macroparticles (weight W = 3e5, ~2e5 particles at the PIC plateau density) pushed with the PIC's
CPU reference kernels (`pic2d.kernels`: bilinear gather, relativistic Boris in the meridional frame,
Cartesian advance with rotation, boundary classification against the plasma-cell mask, renormalised
bilinear wall deposit) at **dt = 1 ns** (ion Courant <= 0.5 at 300 eV; the electron time scales are
not resolved by construction). Absorbed ions deposit their charge on the dielectric wall nodes or count
as anode / beam current; the ion field work is the Boris kinetic-energy change.

Births: `S_k = n_g k_iz(T_k) N_k` per cell, positions sampled from the cell's Boltzmann density on the
node grid (so ions are born where the electrons are), Maxwellian at 300 K, carry-rounded expected
counts.

### 3.2 Field step (self-consistent, per-cell Poisson-Boltzmann)

**Choice: the PIC's `Poisson2D` mesh and operator on the same grid**, not a per-cell axial 1-D
Poisson. Argument: (i) the operator, its Gauss-law identity (plasma + wall + induced charge = 0) and
its convergence order are verified in `tests/pic2d`; (ii) the L2 potential lives on the PIC's nodes,
so the per-cell extraction (`extract_targets`) is applied to both models unchanged and every per-cell
/ per-cusp comparison is node-for-node; (iii) the radial structure - the sheath in the leak windows,
the positively floating insulated wall between cusps, the ions' radial-versus-axial competition - is
exactly what a 1-D axial Poisson with a radial sheath closure would have to assume, and it is the
part of the PIC result that discriminates the model (section 9); (iv) the cost is irrelevant: the
exact block-Thomas factorisation of the 60 x 480 operator takes 40 ms, a back-solve 5 ms.

Unknowns: the potential on the unknown nodes and `ln C_k` per cell, with

    n_e(node) = exp(ln C_k + phi(node) / T_k)      on the populated nodes of cell k, 0 elsewhere.

Equations: the discrete Gauss law at every unknown node with ion charge (deposited, `charge_to_source`
as the PIC), electron charge `-e n_e V_node`, wall surface charge (previous ions + the **implicit**
electron deposit `-e dt (1/4) n_e(wall) v_bar_k A_eff`), and one constraint per cell

    sum_{node in k} n_e (V_node + dt (1/4) v_bar_k A_eff,node) = N_k^target,

i.e. the electrons left in the cell plus those lost over the step to the wall and the electrodes (at
the new state: the losses are implicit, which is what makes 1 ns stable against the sheath response).
Damped Newton with backtracking; Jacobian = PIC operator + positive diagonal, factorised by the
block-Thomas scheme of `pic2d.poisson` with the shifted diagonal (`pb_solver.ShiftedBlockSolver`);
the K constraints are eliminated by a bordered Schur solve. Fail-closed publication: recomputed Gauss
residual <= 1e-7 of the source norm, every cell constraint <= 1e-7, total-charge identity, finiteness,
non-negative fluxes, no emptied cell (`pb_solver.PoissonBoltzmannSolver.solve`).

### 3.3 Populated flux tubes (leak-window closure)

`psi(r,z) = int_0^r B_z r' dr'` on the nodes. For cusp k with leak half-width `w_k`, the wall flux
`|psi_w(z_c +- w_k)|` bounds the populated tubes: a node of cell k is populated when `|psi| <= max`
over the cell's cusps of that bound. Electron losses to the wall happen only on populated wall nodes,
weighted by the magnetic access `|B.n|/|B|` of the face (radial faces `|B_r|/|B|`, stair-step risers
`|B_z|/|B|`, floor 0). Electrode fluxes are the Boltzmann thermal flux through the populated electrode
faces weighted by `|B_z|/|B|`.

### 3.4 Electron fluid per cell

Counts: `N_k^target = N_k + dt (S_k + F_k - F_{k-1} + injection_k)` with the cusp currents

    e F_k = G_k [ (phi_k - phi_{k+1}) - (n_k T_k - n_{k+1} T_{k+1}) / mean(n_k, n_{k+1}) ]

(generalised Ohm's law across cusp k with the declared conductance `G_k`; `phi_k` is the cell's
density-weighted potential of the previous solve, `n_k = N_k / V_k`); the wall and electrode losses
come out of the implicit solve. Energies: explicit Euler with (per electron) `2 T_src` convected across
an interface plus the field work `e (phi_dst - phi_src)` in the receiving cell; injected electrons
bring `2 T_inj + e (phi_D - 0)`; escaping electrons remove `2 T_k` plus the sheath they climb (the
boundary receives `2 T_k` plus the fall through an attracting boundary); ionisation `-12.13 eV`,
excitation `-8.32 eV` with the Maxwellian rates of the PIC's tables (`rates.build_rate_table`); plus
the exact field work of the intra-cell Boltzmann redistribution `e sum (phi_mid - phi_k) dn_e V`
(one-step lag). `T_k = W_k / (1.5 e N_k)`; a cell whose energy or count would reach zero fails closed.

### 3.5 Neutrals

PIC model v1.3 verbatim (`pic2d.neutrals.NeutralInventory`): 0-D inventory, feed 8.551e16 /s,
exit effusion at 300 K, artificial relaxation 30 ns to the fixed point (only the fixed point is
physical), atom ledgers exact to round-off, null-collision ceiling 5.5e19 as a hard bound.

### 3.6 Ledgers (fail-closed conservation)

Charge: plasma + wall + induced electrode charge = 0 at every solve (published check). Atoms: the
inventory ledger. Energy: `d(KE_ions + W_e + U_field) = phi_anode (dQ_induced - dQ_absorbed) +
injected thermal + born-ion thermal - absorbed ion KE - electron energy to boundaries - inelastic`,
closed exactly for the discrete Gauss law with the electrode work evaluated between consecutive
solves; the residual (Boris/leapfrog work vs the exact field-energy identity, midpoint errors of the
fluid transfers) is recorded per series interval and gated on the trailing window as the PIC's v2.0.3
windowed residual (here two-sided, 5 % of the electrode work).

## 4. What L2 v2 predicts

Per cell: electron count / density, temperature, density-weighted potential (so the potential steps
between cells), ionisation share, ion wall-loss fraction. Per cusp: ion wall current, electron wall
current (= ion current at a floating dielectric), near-wall sheath drop, effective per-transit
electron loss probability (the Kornfeld chain on L2's own currents). Global: I_d, S, utilisation, n_g
fixed point, peak n_e, beam current, exit electron return, wall currents, anode ion fraction. All of
these are compared with the PIC where the PIC gives a band (section 6); none is an input.

## 5. What L2 v2 takes from the PIC (closure inputs, fidelity)

Shared inputs (not closures): operating point (300 V, 3 mA at 2 eV, feed, seed), the P2 field
source, the cross-section tables, the grid. Closures - the only two, read from the accepted base
plateau maps through the mini-sweep extraction (`experiments/hybrid_l2_v2/closure.py`,
provenance = the maps' byte hash in `protocol.json`):

| closure | value (anode -> exit cusps) | PIC pair spread | fidelity |
|---|---|---|---|
| cusp conductance `G_k = e F_pass,k / drive_k` [S] | 2.865e-5, 4.976e-5, 1.698e-5 | +16 / +17 / +12 % (seed-b), +20 / +9 / +12 % (W x 0.7) | window-averaged chain currents and cell potentials of ONE design; design-independence untested |
| leak half-width `w_k = FWHM_k / 2` [mm] | 0.200, 0.225, 0.500 | 0 / 0 / -50 % | FWHM of a 50 um wall-flux profile: 4-20 cells wide, quantised to the cell |

The input-uncertainty family (`closure-g-low/high`, `closure-w-low/high`: x 0.7 / x 1.3) brackets
these spreads. The mini-sweep will provide the same two closures for four designs; L2 v2 is NOT run on
them here (the sweep is still running) and makes no claim about them.

## 6. The six GATE-L2 gates made concrete

`paper/evidence/result-gates.json#GATE-L2` requires seven metric constraints. Their L2 v2 reading
(`cft_revival.hybrid.gates`, evaluated by `experiments.hybrid_l2_v2.run assess`; every gate has a
synthetic test that fails, `tests/hybrid/test_l2_simulation_and_gates.py`):

1. `interface_conservation_passed` - charge identity <= 1e-7 (relative to the electron charge) at
   every solve, atom ledger closure <= 1e-9 of the inventory, |trailing-window energy residual| <= 5 %
   of the electrode work, and the PIC's plateau rule reached (trailing-20 % drifts of I_d, N_e, n_g
   < 5 % after >= 3 ion transits of 2.4 us).
2. `spatial_levels >= 3` - 100 / 50 / 33.3 um grids (`spatial-coarse`, `base`, `spatial-fine`), each
   to its own plateau; the spread of I_d, S, peak n_e, beam, step 1, T_e of cells 2-3 is reported.
3. `temporal_levels >= 3` - dt 2 / 1 / 0.5 ns (`temporal-coarse`, `base`, `temporal-fine`).
4. `code_comparison_passed` - the base case against the PIC base plateau: 30 quantities with a PIC
   particle band, tolerance `clip(2 x band, 5 %, 12 %)`; 12 quantities whose PIC band exceeds 12 %
   are `not_compared` (the reference is less precise than the cap): steps 0 and 2, cell-2 potential,
   cusp-1/2 wall currents, cell-0/2/3 ionisation shares, cell-0/3 ion wall-loss fractions, cell-0/2
   inventories. The compared list and tolerances are frozen in `protocol.json#pic_reference`.
5. `numerical_uncertainty_reported` - the spatial, temporal and statistical (W x 0.5, second seed)
   spreads are reported for the level quantities.
6. `failed_cases_count` - cases that did not reach the plateau rule within their budget.
7. `uncertainty_components` - input (closure sensitivity), numerical (levels), emulator (identically
   zero: no surrogate anywhere), model discrepancy (L2 - PIC over the compared quantities).

Verdicts: `accepted` (all hold), `rejected_on_comparison` (conservation and the refinement families
hold, the comparison does not - a valid, informative outcome), `not_evaluable`.

## 7. Cost budget

Headline case: 60 x 480 nodes, ~2e5 macro-ions at plateau, one CPU process (numpy, BLAS 4 threads):
~90-115 ms/step measured in the shakedowns -> 7,200 steps (3 transits) in 11-14 min, 12,000-step budget
(12 us) in ~20 min. The PIC base plateau cost 10,141 s on an RTX 5090 (5.12 M steps at 1.5 ps). The
achieved ratio is reported in the assessment as wall-clock (PIC GPU / L2 CPU) - a same-hardware
ratio would need the PIC's CPU step cost, which is not measured here.

## 8. What L2 v2 must NOT claim

* Nothing about thrust, specific impulse, efficiency or any performance quantity.
* Nothing about the other mini-sweep designs, or about design ranking: the two closures are the
  reference design's; their design-independence is untested.
* No validation: the reference is a development PIC plateau, itself unvalidated and on a grid whose
  Debye resolution (3.17 cells per lambda_D at the peak) sits at the heating threshold; agreement with
  it is model-to-model consistency, not truth.
* No electron kinetics: T_k is a Maxwellian fluid temperature; the PIC's hot tails (T_e,max 59 eV,
  wall electron mean energy 200 eV) have no counterpart, so a hotter L2 fluid for the same S is a
  model discrepancy, not a cross-section difference.
* No claim that the cusp conductance is a transport law; it is one number per cusp from one PIC run.
* No claim that the per-cell Boltzmann electrons reproduce the PIC's intra-cell radial structure
  beyond the populated/depleted split of section 3.3.

## 9. Shakedown record (non-evidentiary; written before the preregistered run)

* Synthetic field, 30 x 240, 400 steps: full path run -> checkpoint -> resume -> finalize -> assess
  exercised; charge identity <= 1e-7; atom ledger 1e-15.
* Real field, first design (pure per-cell Boltzmann, wall access `|B_r|/|B|` floored at 0.02): stable
  staircase [340, 262, 230, 55] V, I_d 2.6-3.2 mA, S ~3e16 /s, but ions were lost radially everywhere
  along the wall (wall ion current 4.7 mA, beam 0.13 mA vs the PIC's 3.7 / 2.29 mA): an isothermal
  Boltzmann cell shorts out the intra-cell axial field and puts a sheath along the whole wall, whereas
  the PIC wall between cusps floats above the plasma and repels ions. The populated-flux-tube closure of
  section 3.3 was adopted in response (one more PIC quantity: the leak width, already in the sweep's
  closure list). With it the wall fluxes are confined to the cusps and the walls between cusps float
  positive, as in the PIC.
* Real field with the frozen design, 300 steps (0.3 us, 0.13 transits): I_d 2.6 mA, S 4.3e16 /s,
  cells [358, 299, 256, 119] V and T [20, 26, 31, 23] eV against the PIC's [309, 184, 129, 34] V and
  [11.6, 6.8, 7.2, 5.8] eV at plateau - far from plateau, but the direction (hotter fluid, higher cell
  potentials, larger exit-side step) is the expected consequence of Maxwellian rates and a single
  fluid temperature per cell. The comparison gate may well fail on T_e and the potentials; that is what
  the run is for.
