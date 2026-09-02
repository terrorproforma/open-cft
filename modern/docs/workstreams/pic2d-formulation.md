# Axisymmetric (r,z) electrostatic PIC-MCC: formulation and claim boundary

## Claim boundary

`cft_revival.pic2d` is a development/screening-fidelity kinetic model of the
divergent-exit CFT discharge channel. Its numerics are verified by
`modern/tests/pic2d` (Poisson order, Gauss law, orbit_mc parity, collision
rates, checkpoint determinism, CPU/GPU parity). Its physics is deliberately
simplified (section "Simplifications"), it has not been compared with any
experiment, and no run under this package is preregistered. Nothing here is a
thruster performance prediction. The first snapshot experiment
(`modern/experiments/pic2d_cft_snapshot_v1`) is labelled
`development_screening_not_preregistered`.

## Geometry and mesh

The channel is the P2 `divergent-exit-stack` bore: radius 2 mm from the anode
face (`z = 0`) to `z = 18 mm`, then a linear cone to 3 mm at the exit plane
`z = 24 mm` (regions `injector-zone`, `channel-straight`,
`channel-divergent-exit` of the qualified P2 mesh). A uniform node-centred grid
covers the bounding box `r ∈ [0, 3 mm] × z ∈ [0, 24 mm]`; `Grid2D` requires the
bore radius to fall on a radial grid line so the straight wall is exact. The
plasma region is the union of *plasma cells*: cells whose outer radius lies
inside the wall radius at the cell's lower-z edge (the wall radius is
non-decreasing in z). In the cone this is a one-cell stair-step. All
derived geometry (node volumes, conductances, wall nodes, particle boundary
classification, surface-charge targets) comes from this one cell mask.

## Field solve

Gauss's law is discretised by cell-wise finite volumes. Every plasma cell
contributes four edge conductances `C = ε0 · face area / edge length`
(radial faces `2π r_{i+1/2} Δz/2`, axial faces `π(r_{i+1/2}² − r_i²)` and
`π(r_{i+1}² − r_{i+1/2}²)`), giving the symmetric positive-definite system

    Σ_e C_e (φ_n − φ_m) = Q_n^vol · V_geom,n / V_shape,n + Q_n^surf

on unknown nodes. Dividing by the geometric node volume recovers the standard
second-order cylindrical stencil, including the regular axis form
`4(φ_1 − φ_0)/Δr²`. Faces towards non-plasma cells carry no flux: the
dielectric backing is treated as a perfect insulator (homogeneous Neumann) and
the deposited wall surface charge enters the node balance directly. The anode
face is Dirichlet `Ua` and the exit plane Dirichlet `0 V`.

Node charge `Q^vol` comes from bilinear deposition; the density estimate
`ρ = Q/V_shape` uses shape-function volumes (∫ S_n 2πr dr dz), so a uniform
density deposits a uniform `ρ` including at the axis (Verboncoeur 2001). The
Gauss law needs `ρ V_geom`, hence the ratio `V_geom/V_shape` (exactly 1 on
interior nodes, 3/4 on the axis).

Two solvers share the operator. The default is an exact block-Thomas (block
Cholesky) factorisation over axial columns: Schur complements are inverted
once and each solve costs two small dense matvecs per column
(1 ms at 31×241 nodes, 5 ms at 61×481, 20 ms at 121×961). A Jacobi-PCG with
deterministic reductions (and CUDA graphs on the device) is the alternative.
Both publish only after an independently recomputed true residual meets
`relative_tolerance · |rhs|` (default 1e-10). Verified: manufactured solution
order 1.999 over three refinements, direct = dense = PCG to 1e-9, discrete
Gauss law (electrode induced charge balances the total source) to 1e-9,
finite regular potential on the axis.

`E = −∇φ` on nodes uses central differences where both neighbours are plasma
nodes, second-order one-sided differences at electrodes and walls, and
`E_r = 0` on the axis (order > 1.8 verified). Particles gather `E` and `B`
bilinearly with the deposition weights (momentum-conserving, not
energy-conserving).

## Prescribed magnetic field

`B(r,z)` is static and comes from the qualified P2 divergent-exit `A_φ`
checkpoint (level-1, hashes in `modern/spec/pic2d/p2-field-authority-v1.json`,
verified at load against file, payload, mesh, run and sidecar SHA-256).
`BoundP2Evaluator` (the orbit campaign's evaluator carried into the package)
samples `ψ = r A_φ` on a regular grid with the orbit campaign's primary spacing
(Δr 0.125 mm, Δz 0.25 mm) over the whole channel bounding box. Sampling
includes the μ0 dielectric regions; this differs from the campaign's
conservative `r ≤ 2 mm, 1 ≤ z ≤ 23 mm` subdomain and is justified because all
sampled regions have reluctivity 1/μ0 and no source, so `A_φ` is smooth across
their geometric interfaces. `PsiBicubicField` (g = (ψ − ψ_axis)/r²,
`B_r = −r ∂_z g`, `B_z = 2g + r ∂_r g`) then supplies `(B_r, B_z)` at every PIC
node. Verified: node samples equal the bicubic field exactly; bilinear gather
error is second order; the withheld mid-cell P2 error is 3.2e-4 relative RMS;
primary vs refined ψ grid differ by 0.25 % of max |B| at the PIC nodes; a
collisionless electron orbit converges to `orbit_mc.integrate_orbit` as
O(Δx²) (error ratio > 2.5 per refinement, < 5 % of a gyroradius at 48×384),
and in a uniform field the leapfrog orbit matches orbit_mc to ≤ 1e-6 gyroradii
over 3000–4000 steps. Maximum |B| is 0.226 T inside the bore and 0.291 T at
the dielectric corner of the bounding box.

## Particles

Electrons and Xe⁺ (mass 2.180e-25 kg, no mass scaling) are kinetic with one
common macro weight `W`. The pusher is the relativistic-momentum Boris update
in the particle's meridional frame (`x` radial, `y` azimuthal) with the exact
operation order of `orbit_mc.integrator.relativistic_boris_push`
(8-ulp agreement); the γ factor is the guard against superluminal states. After
the velocity update the position advances in Cartesian and the frame rotates
back (`r' = √(x'² + y'²)`), which treats the axis without special cases. The
cycle is leapfrog: `x^n, v^(n−1/2)` → deposit → `φ^n`, `E^n` → push →
`x^(n+1)` → boundaries → MCC → injection.

Boundaries after the push: `z < 0` anode absorption (counted current),
`z ≥ 24 mm` exit absorption (ion beam / electron exit current, binned in r),
landing in a non-plasma cell → dielectric wall absorption with the particle's
charge deposited on the plasma-side wall nodes using renormalised bilinear
weights (counted per axial column with its kinetic energy). A landing cell with
no plasma node means the particle crossed more than one cell: the run fails
closed (Courant violation).

Deposition rounds each bilinear weight to `2^-40` and accumulates integers, so
node charges are independent of particle order and bit-identical between numpy
and Warp atomics; the quantisation error is ≤ 2^-40 |q| W per contribution.

## Collisions

Null-collision MCC for electrons on a uniform static Xe background with the
LXCat Biagi-v7.1 set (`modern/spec/pic2d/xenon-cross-sections-v1.json`,
tabulated, source bytes hashed): momentum-transfer elastic (isotropic, speed
preserved; the 2 m_e/M loss is neglected), one lumped excitation channel with
8.32 eV loss (sum of the four Biagi-v7.1 levels), and single ionisation with
12.13 eV threshold. The secondary electron energy follows Vahedi–Surendra
(`E_s = B tan[ξ atan((E − E_iz)/2B)]`, `B = 8.7 eV`); both electrons are
emitted isotropically; the ion is born at the event position with a
Maxwellian neutral velocity at `T_g`. Cross sections are resampled once onto a
0.05 eV uniform grid shared by CPU and GPU so `σ(E)` is bit-identical; the
random streams differ (numpy `default_rng([seed, step, stream])` vs Warp
`rand_init(hash(seed, step, stream), particle)`), so CPU/GPU MCC parity is
statistical (chi-square). Verified: per-process rates within 4σ of
`n_g σ v Δt` on 4e5 electrons at 40 eV; energy bookkeeping; deterministic seed
replay.

## Sources

Electrons are injected in the last half cell before the exit plane, uniform
in area, with Maxwellian transverse velocities and a flux-weighted
half-Maxwellian axial velocity directed into the channel, at a fixed current
(fractional counts carried between steps). Optional uniform quasi-neutral seed
plasma bootstraps the discharge. Injected velocities are assigned as
`v^(n+1/2)` without a backward half-kick (O(Δt) inconsistency, documented).

## Stability and admission

`stability_report` evaluates `ω_pe Δt ≤ 0.2` (reference and runtime peak node
density), `Ω_ce Δt ≤ 0.2` from the node-map maximum |B|, `max(Δr,Δz)/λ_D`
(reference density and temperature) against a configured limit,
particle Courant from the configured maximum electron energy, and the
null-collision probability `≤ 0.1`. `Simulation` refuses to construct on a
violation; at runtime the peak-density `ω_pe Δt` check and the one-cell
crossing check stop the run with a typed `PIC2DStabilityError`. These are
necessary gates, not proof that sheaths or gyro-orbits are resolved.

## Energy accounting

Kinetic energies use the stable relativistic form `(γ−1)mc²` summed with `W`;
field energy is the exact discrete `½ φᵀAφ`. In a pure magnetic field a single
electron conserves kinetic energy to < 1e-9 over 500 steps. For the open
system the series report `Δ(K+U) − (K_injected − K_absorbed − E_inelastic +
K_born ions + W_electrode)` per interval. `W_electrode = Σ_k V_k (ΔQ_induced,k −
q_absorbed,k)` (v1.1) is the work of the supplies holding the Dirichlet
electrodes: `Q_induced,k = Σ_{nodes of k} (Aφ)` is the conductor charge of the
discrete Gauss law and `q_absorbed,k` the net particle charge collected there.
With this term the ledger closes to within ~15 % of the electrode work in the
verification test (`test_ledger_closes_with_electrode_work`); the remaining
residual is the momentum-conserving scheme's grid heating and is reported per
case, not claimed to vanish. In v1 the term was absent and the residual was
essentially the supply work.

## Backends and determinism

The numpy CPU backend is the reference and is bitwise deterministic
(checkpoint/resume verified). The Warp backend (CPU or CUDA) reproduces the
deposition bit-for-bit, the push to roundoff (FMA contraction), uses the same
host direct field solve (identical φ), counter-based RNG, deterministic
compaction and spawn (prefix scans), and two host reads per step; it is
deterministic and checkpoint-resumable bitwise for the dynamical state, while
ledger energies accumulated with float64 atomics agree to roundoff. The device
PCG path is available (`PoissonConfig2D(method="pcg")`) and verified against
the direct solve; it is slower than the host direct solve for these grids
because ~500–1000 Jacobi-PCG iterations per step dominate.

### v1.1 all-GPU step

`PoissonConfig2D(method="device-direct")` runs the block-Thomas sweeps on the
device (factorisation built once on the host): one launch per radial block,
256-lane coalesced row dot products with deterministic tile reductions. φ
equals the host direct solve to 1e-10 and the true residual contract
(`‖Aφ − b‖ ≤ tol‖b‖`) is re-evaluated on the device at every host sync and fails
closed. The particle counts, the statistics vector (work, max speed², absorbed
tallies, collision tallies) and the series sample are read by the host every
`device_sync_steps` steps only; between syncs the step is entirely on the
device. Per-particle same-address atomics in the push and MCC kernels were
replaced by per-block `tile_sum`/`tile_max` reductions (block 256) with one
atomic per block; the tallies stay exact integers identical to numpy. Ions are
pushed every `ion_subcycle = k` steps with `k Δt` (leapfrog stagger `k Δt/2`),
newborn ions join the charge immediately and are first pushed at the next ion
step; `k = 4` vs `k = 8` agree in ion count, positions and φ (see
`tests/pic2d/test_pic2d_v11_step.py`). The v1 fine case (5.4 M macro-particles)
went from 40.7 ms/step (host solve 18 ms) to 5.46 ms/step (device solve
2.3 ms); the specification is `modern/spec/pic2d/pic2d-model-v1.1.json`.

## Artifacts

Canonical JSON (sorted, compact, finite) with `.sha256.json` sidecars,
sorted-key npz with sidecars, and `cft.pic2d.checkpoint.v1` checkpoints
binding config, field, cross-section and code identities. Provenance records
grid, species, stability gate, field map hash, P2 authority hashes,
cross-section payload/file hashes, code hash, runtime identity.

## Operating point and resolution budget (v1.1)

v1 ran at `n_g = 5e20 m⁻³`, 100 mA injection: the ionisation source exceeded the
unmagnetised Bohm ion loss 12× at the observed `T_e ≈ 18 eV`, and because ions
need ~1 µs to reach the walls there is no saturation channel before then, so
the avalanche had to overshoot and tripped the `ω_pe Δt` gate at 49–60 ns
(`diagnosis.json` in the v1 results). v1.1 chooses the operating point from the
0-D balance `n_g⟨σ_iz v⟩ n_e V = n_e v_Bohm A_loss` (`V = 3.5e-7 m³`,
`A_loss = 3.6e-4 m²`): at `n_g = 1e20 m⁻³` the particle balance closes at
`T_e ≈ 7.9 eV`, and a power balance at 3 mA gives `n_eq ≈ 2e17 m⁻³`.
The numerics budget uses a 2× design ceiling `n_max = 4e17 m⁻³`, `T_e = 8 eV`:

| quantity | value |
|---|---|
| `λ_D,min` | 33.2 µm |
| `Δt` | 1.5 ps (`ω_pe Δt = 0.054` at `n_max`; `Ω_ce Δt = 0.077` at 0.291 T) |
| fine grid 60 × 480 | `Δr = 33.3 µm`, `Δz = 50 µm`, `max(Δr,Δz)/λ_D = 1.50` (gate 2.0) |
| coarse grid 30 × 240 | `66.7 × 100 µm`, ratio 3.01 (gate 3.1, under-resolved by design) |
| macro-particles per cell at `n_max`, r = 1 mm | fine 70 / 140, coarse 70 / 140 |
| ion subcycle | `k = 8`, `k Δt = 12 ps`, `ω_pi k Δt = 8.8e-4` |
| runtime gate trip density | `5.6e18 m⁻³` (14× `n_max`) |

The first v2 attempt used `Δt = 3 ps` (`ω_pe Δt = 0.107` at `n_max`) and was
stopped fail-closed after 0.27 µs in both coarse cases: the runtime gate reads
the instantaneous peak node density, and the axis nodes (a handful of
macro-particles) carry ~3× shot noise on top of the window-averaged peak
(1.4e18 instantaneous vs 4.7–5.4e17 window). Halving `Δt` moves the trip
density far above the design ceiling so the gate only fires on a genuine
runaway; the shot-noise reading is recorded in the v2 protocol.

## Simplifications (v1.1)

- static uniform neutral background; no depletion, no neutral flow (at
  `n_g = 1e20 m⁻³` the 0-D ionisation rate `n_e v_Bohm A_loss ≈ 2e17 s⁻¹`
  removes ~1 % of the 3.4e13 neutrals in the channel over 1.6 µs, locally a few
  per cent near the ionisation peak, so depletion is deferred and stated rather
  than implemented; a real campaign at longer times needs it);
- no ion–neutral collisions (elastic or charge exchange): the hash-bound LXCat
  extract holds only the e–Xe set, so the Xe⁺–Xe data would need a new
  provenance-bound source first;
- elastic e–Xe isotropic without recoil energy loss; one lumped excitation;
  isotropic ionisation products;
- dielectric wall perfectly absorbing with surface charge; zero field in the
  backing; no secondary emission or sputtering;
- exit plane Dirichlet 0 V as cathode/neutraliser reference; electrons injected
  there at fixed current;
- anode perfectly absorbing at fixed potential;
- one-cell stair-step cone wall;
- prescribed static B; bilinear node interpolation of the bicubic ψ field;
- electrostatic, azimuthally symmetric: no azimuthal instabilities or anomalous
  transport;
- momentum-conserving bilinear scheme (energy not exactly conserved).

## Remaining for a preregistered PIC campaign

1. Operating-point study beyond the single v1.1 point (3 mA, 1e20 m⁻³): map
   the resolved envelope in (I_inj, n_g, V) and confirm the 0-D balance against
   the kinetic plateau density; runs to several ion transit times.
2. Ion–neutral elastic and charge-exchange collisions (needs a hash-bound
   Xe⁺–Xe cross-section source); neutral depletion or a neutral flow model;
   wall secondary electron emission.
3. Energy-conserving or explicitly corrected deposition/gather: the electrode
   work now closes the supply term, the remaining per-case residual is the
   scheme's grid heating and must be bounded, not just reported.
4. Exit boundary alternatives (plume domain, floating cathode line) and
   sensitivity to the exit reference.
5. Convergence in grid, timestep and particles per cell with quantified
   uncertainty; run-to-run seed variance.
6. Cross-code comparison (e.g. WarpX/PICMI) and any experimental observable.
7. Preregistration under `experiment_runtime` with frozen protocol, hashes,
   shakedown on the real field, and one immutable execution.
