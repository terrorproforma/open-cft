# AI surrogate trained on PIC-MCC outputs: plan (doc only, no build)

**Status: plan, not an implementation and not a result.** Written 2026-09-05 against
`feat/sota-foundation` at `036bd679` after reading (read-only) the user's welding AI subsystem
`Reality-Simulator/ai` and this repository's surrogate / active-learning / optimisation /
validation / experiment-runtime stack and the PIC records that exist today. Nothing in this
document is preregistered; every number about *our* data is quoted from a recorded artifact and
every number about cost is a projection with its anchor named. The brief: apply "the same
principle and architecture" as the welding reality model - a learned surrogate trained on simulator
outputs, used to accelerate design optimisation - to the PIC-MCC plasma simulator
(`cft_revival.pic2d`).

The short answer, before the detail:

* **The principle carries over intact**: the simulator stays the oracle and the only source of
  labels; a registry-driven, hash-bound, fail-closed data plane feeds a surrogate with explicit
  aleatoric + epistemic uncertainty; the surrogate runs a cheap "objective-only" mode for search
  and a "full-field" mode for inspection; every surrogate-selected design goes back to the
  simulator for verification and every disagreement becomes an active-learning sample.
* **The architecture only partly carries over**: the welding model is a *time-stepping world model*
  (state + future controls -> future 3-D fields, GRU control sequence, ConvGRU rollout, a
  real-time sensor student). Our object is a *steady-state design operator* (geometry field map +
  operating point -> time-averaged 2-D plateau fields and their scalars). The neural-operator
  backbone, the heteroscedastic heads, the deep ensemble, the multi-fidelity residual and the
  verification queue map one-to-one; the rollout, control-sequence and real-time layers have no
  analogue and are dropped.
* **The data reality decides the order**: today there are plateaus for **four distinct designs at
  one operating point** (three qualified 33 um channel plateaus recorded or record-pending, one
  running), at 4-7 GPU-h per channel plateau and 17-50 GPU-h per plume run. No field-operator
  network can be trained on that; a scalar GP cannot be gated on it either. What CAN be built now
  is the dataset contract, the ingestion, the label-noise floor and the campaign generator, so
  that the first 30-100 qualified PIC plateaus (roughly 150-1000 GPU-h including replicates and
  the runs that never plateau; USD 0.5k-3k at the Lambda rate) land directly in a training set
  instead of in ad-hoc result directories. That is the plan below.

---

## 1. What `Reality-Simulator/ai` is

Repository studied: `C:\Users\Angus\Desktop\projects\Reality-Simulator\ai` (read-only; paths below
are relative to it). Package `src/reality_ai`, 67 source modules (excluding `__init__`), 28 test
files, 17 scripts, 16 docs, a pre-results architecture manuscript (`paper/latex/`). Status per
`IMPLEMENTATION_STATUS.md`: the software is complete through the data plane, model ladder,
optimiser and real-time student, but **no model has been trained on real simulator output** - the
only datasets are synthetic harnesses, explicitly "plumbing evidence only". The plan document
(`docs/FULL_PHYSICS_SURROGATE_PLAN.md`, 2569 lines) is the design; the code is the first
executable slice of it.

### 1.1 Architecture in plain terms

Three levels (`docs/ARCHITECTURE.md` section 1):

1. **Reality-Simulator (`weldsim`)** - the expensive Warp-based welding physics oracle: the label
   generator, the calibration target, the verifier of every optimiser output, and the fallback
   outside the surrogate's validated envelope. It never imports the AI package.
2. **Large full-physics surrogate** - approximates the conditional operator

   `S_Theta(G, M, B, X_0, U_{0:T}, theta) -> (X_{0:T}, Q, Sigma)`

   geometry `G`, material functions `M`, boundaries `B`, current state `X_0`, the *future control
   sequence* `U` (torch path, power, speed, wire feed), uncertain parameters `theta` -> future
   fields `X`, engineering objectives `Q` (penetration, bead width, porosity, distortion, cycle
   time) and uncertainty `Sigma`. It replaces routine simulator inference inside a validated
   envelope and drives offline weld-program optimisation.
3. **Small real-time model** - a sensor-conditioned belief-and-correction student distilled from
   the large model's privileged state; bounded corrections under a deterministic safety layer.
   (No analogue for us; listed for completeness.)

### 1.2 Load-bearing design choices, with file references

| choice | where | what it does | why it matters to us |
| --- | --- | --- | --- |
| **Registry-generated channel layout, fail-closed** | `src/reality_ai/data/physics_registry.py` (`audit_simulator_inventory`, `build_dense_layout`, `assemble_dense_fields`), `data/layout.py`, `configs/physics/roadmap_coverage.json`, `scripts/audit_physics_coverage.py` | Every simulator state value has a semantic id, unit, support, cadence, disposition (input / target / preserve-raw / excluded). Channels are compiled from the registry, sorted by id, hashed. An exported source id not in the registry **blocks training and release**; unknown state is kept raw and fails closed until classified. | Exactly the discipline our `experiment_runtime` already imposes on protocols; the surrogate dataset must inherit it (section 4.4). |
| **Episode store with content identity** | `data/npy_store.py`, `data/zarr_store.py`, `data/manifests.py`, `data/bundle.py` (`DatasetBundle.open` cross-checks manifest / registry / normalisation / layout hashes), `docs/DATA_PLANE.md` | NPY (tests) or Zarr v2 (production) episodes; every array carries dtype, shape, SI unit, frame and a SHA-256 of its canonical bytes; episode metadata pins simulator commit, config hash, seed, fidelity. | Our records already have byte hashes + sidecars (`pic2d/artifacts.py`); the missing piece is the *dataset* manifest over many records. |
| **Normalisation fit on the training split only; immutable lineage** | `common/normalisation.py`, `data/splits.py` | Streaming per-component moments; identity / z-score / log1p+z / signed-log; unit drift is a hard error; the normalisation file is bound to one registry hash and dataset id. | Our targets span decades (n_e 1e17-1e19, S 1e16-1e17/s): log-normalisation is mandatory; must be recorded, never refit on held-out data. |
| **Whole-scenario splits** | `data/splits.py` (split = hash(seed, scenario_id)), `data/campaign.py` (scenario vs program scope) | Every programme, replicate and frame of one physical scenario lands in one split; the manifest validator rejects leakage. | For us: every grid rung, seed replicate, W replicate and frame of one *design* must share a split (section 4.5). |
| **Model ladder behind one contract** | `surrogate/fno.py` (pure-PyTorch 3-D FNO), `surrogate/unet.py`, `surrogate/tfno.py` (CP-factorised), `surrogate/ufno.py`, `surrogate/backbone_factory.py`, `surrogate/configurable_model.py` (`ConfigurableFullPhysicsSurrogate`), `configs/surrogate/model_ladder.yaml`, `docs/SURROGATE_ARCHITECTURES.md` | Interchangeable spatial backbones (U-Net / FNO / TFNO / U-FNO) with a fixed input-output contract so architecture comparisons hold dataset, layout, splits, horizons and compute constant. U-FNO is the stated leading hypothesis; nothing is selected before data exist. | The same "ladder, not a favourite" rule applies to us; and our first ladder rung is not a network at all (section 4.2). |
| **Conditioning** | `surrogate/control_encoder.py` (`ControlParameterEncoder`: GRU over the future control sequence + MLP over scalar parameters -> context vector), `surrogate/model.py` (context broadcast as extra channels into the backbone) | Static geometry and materials enter as channels; controls and parameters enter as a context vector tiled over the grid. `encode_static_context(..., detach=True)` caches the geometry encoding for thousands of candidate programmes. | Our operating point is a static scalar vector (no sequence): MLP context, tiled. The magnetic field map IS the design and enters as channels. |
| **Three inference modes** | `surrogate/model.py` (`SurrogateMode.FULL_FIELD / OBJECTIVE_ONLY / SPARSE_QUERY`), `surrogate/objective_decoder.py` (pooled latent + context -> MLP -> per-horizon objective mean and log-variance), `surrogate/query_decoder.py` (`grid_sample` on the latent at normalised coordinates + field/horizon embeddings), `surrogate/field_decoder.py` | Broad search decodes only objectives; shortlists decode sparse points; finalists decode full fields (`configs/optimisation/default.yaml` `surrogate_mode`). Each mode is validated and approved independently (`docs/VALIDATION_PLAN.md`). | Directly reusable: our optimiser needs scalars (I_d, S, utilisation, loss fractions) per candidate; fields are for inspection and for physics-consistency checks. |
| **Uncertainty = heteroscedastic heads + deep ensemble + explicit multi-fidelity residual** | `surrogate/model.py` (every head returns mean and `log_variance.clamp(-12, 6)`), `physics/pino_losses.py` (`gaussian_negative_log_likelihood`), `surrogate/ensemble.py` (`SurrogateEnsemble`: epistemic = variance of member means, aleatoric = mean of member variances, >= 2 members), `surrogate/multifidelity.py` (`MultiFidelitySurrogate`: `prediction_f = base + residual_f`, variances added, fidelity graph validated), `surrogate/uncertainty.py` | Aleatoric and epistemic are separate outputs; fidelities are explicit ids in a parent chain, never pooled as equivalent labels. | Our labels have a *measured* aleatoric band (seed + particle weight) and a *systematic* grid rung: the same decomposition, with the grid rung as the fidelity axis. |
| **Physics-informed losses** | `physics/pino_losses.py` (`compute_physics_informed_loss`: field MSE (masked), objective MSE, Gaussian NLL, energy balance, filler-mass balance, boundary, constitutive, phase-simplex, **sensitivity** = model Jacobian vs paired-intervention finite difference), `physics/energy.py`, `physics/mass.py` | Conservation and constitutive consistency as loss terms; "sensitivity matching" proves the network *uses* a parameter. | Our cheap exact operators: the discrete Gauss law (the PIC's own Poisson stencil), the ion current ledger, S = integral of the ionisation map. |
| **Training** | `training.py` (`train_surrogate_step`), `scripts/train_dataset_surrogate.py`, `scripts/evaluate_dataset_surrogate.py` | Checkpoints embed dataset / registry / normalisation / layout identity; evaluation refuses a bundle with different lineage. | Reuse the rule verbatim. |
| **Dataset campaign generator** | `data/campaign.py`, `scripts/prepare_dataset_campaign.py`, `configs/datasets/s355_gmaw_tfillet_pilot_campaign.json`, `docs/DATASET_GENERATION_CAMPAIGN.md` | Deterministic Latin hypercube over declared axes (each with scope, bounds, transform, unit); scenario-level vs programme-level variables; replicates; snapshot schedule; hash-addressed plan; `simulation_queue.jsonl` one task per line; manifest template marked `planned_not_generated`. Pilot: 16 scenarios x 4 programmes x 2 replicates = 128 episodes, bounds explicitly non-authoritative. | This is the shape of our Phase-2 campaign generator, fed into `tools/cloud/schedule.py`. |
| **Optimisation under uncertainty + mandatory verification** | `optimisation/planner.py` (`OfflineWeldPlanner`: CEM -> projected Adam hybrid over structured programme knots), `optimisation/objectives.py` (`RiskAwareObjective`: minimise / maximise / target / bounds + aleatoric sd + epistemic sd + CVaR over ensemble samples), `optimisation/verification.py` (`VerificationQueue`: best and most uncertain candidates re-run in the simulator; `DISAGREEMENT` -> active-learning priority), `optimisation/surrogate_adapter.py`, `configs/optimisation/default.yaml` (aleatoric 0.10, epistemic 1.00, CVaR 0.50 at alpha 0.90; `reality_simulator_required: true`) | The surrogate accelerates search; the simulator remains the authority; the most important metric is "the probability that surrogate-selected programmes reproduce their predicted quality when re-evaluated by the oracle" (`paper/latex/sections/03_training_optimisation_control.tex`). | Our `cft_revival.optimization` already has the F0-F3 fidelity ladder, "every non-F3 promoted candidate is rejected pending highest-fidelity reevaluation" (`optimization/guardrails.py`), and constrained qLogNEHVI (`optimization/botorch_adapter.py`). The welding queue and our campaign are the same idea. |
| **Validation and governance** | `docs/VALIDATION_PLAN.md`, `docs/TRAINING_PLAN.md`, `paper/latex/sections/04_protocol_status_conclusion.tex` | Per-mode approval; machine-readable operating envelope; held-out geometry / parameter / material / resolution; uncertainty must correlate with error; optimiser-exploitation tests; no replacement claim without envelope + signed report + exact checkpoint + dataset provenance + fallback policy. Mandatory baselines: persistence, linear, U-Net, FNO/TFNO, U-FNO; data-only vs PINO; single vs ensemble. | Matches our evidence rules (preregistered held-out designs, gates, claim boundaries). |
| **Real-time student, distillation, anomaly gate** | `realtime/*`, `docs/PLANNING_AND_REALTIME.md`, `docs/REALTIME_MODEL_PLAN.md` | Privileged teacher -> sensor-only student; bounded corrections; deadline-aware runtime. | **No analogue** (we have no closed-loop hardware). Dropped. |

### 1.3 What it predicts

Per `configs/surrogate/baseline.yaml` and the plan: enthalpy-primary future **fields** at horizons
0.05-1.0 s on a regular voxel grid (`[batch, channels, z, y, x]`, 8-16 spectral modes, 4 FNO
layers, hidden 64), conservative fill fraction, phase fractions, hydrogen, displacement; **scalar
quality objectives** per horizon (penetration, fusion coverage, bead width / height, throat, HAZ
width, hardness, porosity, lack-of-fusion, burn-through, humping / crack risks, residual stress,
distortion, cycle time, energy); **stochastic event** probabilities (spatter, pores, transfer-mode
transitions); and per-output **aleatoric variance** plus ensemble epistemic variance. Inputs are
voxel geometry (occupancy, SDF, interface, fixture masks), sampled material constitutive
functions, the future control sequence, and the uncertain-parameter vector (process efficiency,
Goldak source dimensions, convection, emissivity, ...).

### 1.4 What is honest about it, and what we must not import uncritically

* The welding docs repeatedly separate "implemented" from "validated": synthetic fixtures are
  never welding evidence; "full physics" means "full retention of the simulator state at a
  version", not physical reality. We keep the same language.
* It is a very broad architecture built ahead of any data ("may be over-parameterised relative to
  the initial dataset" - their own Limitations section). For us the data are far scarcer (tens of
  runs, not thousands of episodes), so the ladder must start at the bottom (GP / POD-GP), not at
  U-FNO.
* Their first decisive experiment is a *field* prediction (next second of enthalpy). Ours cannot
  be: with < 50 plateaus the first decisive experiment is a *scalar* prediction with a measured
  noise floor (section 5, Phase 1).

---

## 2. Mapping to the PIC-MCC

### 2.1 Inputs

| welding input | PIC analogue | where it exists today | representation for the surrogate |
| --- | --- | --- | --- |
| voxel geometry (occupancy, SDF, interfaces) on the simulator grid | **the magnetic design**: parametric geometry v1.1 (`cft_revival.geometry`, 11 sweep variables: `stage_count_selector` (3-5 stages), `stage_pitch_m` 3.4-6.5 mm, `magnet_axial_fraction`, `chamber_outer_radius_m` 1.4-4.2 mm, `dielectric_thickness_m`, `radial_clearance_m`, `magnet_radial_thickness_m`, `source_strength_scale` 0.75-1.3, `exit_length_fraction` 0-0.28, `exit_expansion_descriptor`, `first_polarity_selector`; `experiments/l1a_geometry_sweep_v3/protocol.json`) -> material-aware P2 FEM solve (linear iron mu_r 4000 poles + yoke, recoil magnets; `experiments/pic2d_design_mini_sweep_v1/fields.py::produce_field`, 2-7 CPU-min per design, gated on mesh angle / residual / coverage / topology) -> bilinear node field `(B_r, B_z)` on the PIC grid (`cft_revival.pic2d.fields`, `p2_field.py`) plus the mesh masks (plasma / dielectric / anode / exit / far-field; `pic2d/mesh.py`) | mini-sweep `fields/<design>/binding.json` + LFS checkpoints for 5 designs; the reference's authority checkpoints | **static channels on a canonical grid**: `B_r`, `B_z`, `\|B\|`, plasma mask, wall mask, anode / exit masks, normalised `r / r_w`, `z / L_ch`, plus physical scales (`r_w`, `L_ch`, `dr`) as scalars. Derived topology descriptors (cusp planes, per-cusp Koch rho, wall `\|B\|` at cusps, x_w = pi r_w / pitch, N cusps; `cusp_topology_search_v3_1`, sweep-v3 catalogue) as **scalar features** - these carried all the signal in surrogate v2 |
| material constitutive functions | xenon cross-sections (fixed in `pic2d/mcc.py`); no design variable today | - | none; becomes a token only if propellant becomes a variable |
| future control sequence (TCP, power, speed, wire) | **operating point** (static): anode voltage `V` (300 V today), neutral feed `Q_in` / initial `n_g0` (5.5e19 today; the mini-sweep froze equal `n_g0`, not equal mass flow - a declared choice, section 3.4), electron injection 3 mA @ 2 eV (channel-only) or the cathode source (plume: flux-tube cathode, `max_current_a`) | every `protocol.json` `operating_point` block | scalar vector `(V, Q_in or n_g0, I_inject or cathode spec, T_inject)` -> MLP context tiled over the grid (GP inputs at level (a)) |
| uncertain physical parameters (efficiency, Goldak, emissivity) | **closure and model-version flags**: neutral model (0-D inventory v1.3; recycling v1.4), Bohm alpha (planned), SEE (planned), plume domain (channel-only exit-plane Dirichlet vs v2.x plume box), Poisson method (block-Thomas vs `device-mg`; identity rule "a different solver is a different `config_sha256`") | `summary.json` `model_version`, `config_sha256`, `protocol_sha256`, `simplifications`; `spec/pic2d/pic2d-model-v2.0.json` | **fidelity / task tokens**, never mixed silently: one label set per closure; a closure change is a new fidelity id in the residual chain (welding `FidelitySpec.parent_id`) |
| fidelity tiers F0 (batched) / F1 / F2 / F3 (experiment) | **grid rung** 50 / 33.3 / 25 um with particles-per-cell parity (`W = 6e4 x dr dz / (50 um)^2`), `dt`, macro-weight `W`, seed; channel-only vs plume box | `pic2d_cft_steady_state_v2/v4/v5` ladder; `assessment.json` verdicts (`converged` / `resolution_limited` / `refinement_heating` / `no_plateau`) | grid rung = fidelity axis of the multi-fidelity residual (`surrogates.TwoFidelityAR1` or BoTorch `MultiTaskGP` with the rung as the task, `optimization/botorch_adapter.py::build_source_task_models`); seed and W replicates = aleatoric band |
| numerical metadata (commit, dt, grid, seed) | the same | `execution-lock.json`, `summary.json` `provenance` / `platform_fingerprint` / `gpu_identity` | provenance only; hashed into the dataset manifest, never a model input except the fidelity token |

### 2.2 Outputs, level (a): plateau scalars with uncertainty bands

What one accepted PIC plateau record emits today (`pic2d_cft_steady_state_v4/results/`,
`pic2d_design_mini_sweep_v1/results/<design>-channel-33um/`; schema
`cft-revival.pic2d-cft-steady-state-v4.assessment/1.0.0` and the steady-state summary):

| quantity | source key | today's value (v4, reference, 33 um) | measured band | analogue in the welding model |
| --- | --- | --- | --- | --- |
| discharge current `I_d` | `summary.window_currents_a.discharge_a` | 3.80 mA | seed <= 1.1 %; W x0.7 5.7 %; grid 50->33 um +10.4 % | quality objective (e.g. penetration): scalar head |
| ionisation rate `S` | `window_currents_a.ionization_rate_per_s`, `neutral_inventory` | 3.60e16 /s | W 4.6 %; grid -8.5 % | quality objective |
| gross / net utilisation | `neutral_inventory` trailing means | 0.420 | W ~4 %; grid -8.5 % | quality objective |
| neutral density `n_g` (fixed point) | `neutral_inventory.final_density_per_m3`, `neutral_fixed_point_per_m3` | 3.19e19 | W 4.0 %; grid +7.3 % | mass-ledger quantity (filler volume) |
| exit ion beam `I_beam` | `window_currents_a.exit_ion_beam_a` | 2.46 mA | grid +7.3 % | quality objective |
| wall electron / ion currents, anode ion fraction | `window_currents_a.wall_*`, `anode_ion_a` | 3.03 / 3.02 mA; 0.016 | - | quality objective |
| peak `n_e` (window, >= 32 macro-electrons), `T_e,peak`, `T_e,dense` | `peak_node_debye`, `window_maps_summary` | 1.29e18; 5.58 eV | W 11.9 % / 9.3 %; grid -21 % / -24.5 % | quality objective (with the largest band) |
| per-cusp transit-loss probability `p_k`, cusp electron / ion wall current `L_k`, leak width FWHM, sheath drop, near-wall `T_e` / `n_e`, cusp wall `\|B\|` | `closure.extract_targets` (`pic2d_design_mini_sweep_v1/closure.py`) from `maps.npz` wall profiles + `summary.json` | shakedown only (056 at 0.14 us: p 0.136 / 0.096 / 0.071 - non-evidentiary); sweep-wide `targets` deferred to the assessment | block standard error of the trailing window (recorded per target) | **the closure targets the 0-D model was going to consume** - now recorded data only (user direction 2026-09-04 21:23). For the surrogate they are the most design-sensitive scalar heads |
| per-cell ionisation share `S_k / S`, ion wall-loss fraction, cell potential and `T_e`, potential steps | same | same | same | quality objectives |
| plateau / heating verdict, residual power, `Delta / lambda_D`, ignition | `assessment.json`, `grid_heating_triad`, `plateau`, `ignition` | `resolution_limited`; -7.7 % windowed | - | **stochastic-event heads** in welding (spatter, burn-through): here a *feasibility classifier* (ignites / plateaus / heats / avalanches) - section 4.3 |
| `I_beam`, divergence half-angles, IEDF mean, thrust, `Isp`, `eta_a` | plume runs only (`plume`, `maps.npz` `plume_ion_current_per_sr_a`, `iedf_*`) | attempt 7: T 20.9 uN drifting +20 %/window, NOT quotable; no plume plateau exists | - | quality objectives - **only when the plume model is qualified** |

Energy-ledger residual and the neutral ledger closure are *gates and witnesses*, not targets
(welding: energy / mass balance enter as losses; here they decide whether a run is a label at all).

### 2.3 Outputs, level (b): fields (operator-learning target)

`maps.npz` of an accepted plateau (window averages over the trailing 400 000 steps; 91 x 721 nodes
for the reference at 33 um, 67 x 779 / 81 x 685 / 116 x 513 for 047 / 009 / 056):

| array | meaning | welding analogue | surrogate role |
| --- | --- | --- | --- |
| `n_e_per_m3`, `n_i_per_m3` | time-averaged densities | enthalpy / temperature fields | primary field targets (log) |
| `phi_v` | potential | temperature (derived, constitutive) | field target; **Gauss-law consistency** with `n_i - n_e` through the PIC's own discrete operator (`pic2d/poisson.py`) is the cheap exact physics term |
| `t_e_ev` | electron temperature (velocity-moment window) | - | field target (largest shot noise; weight by `sample_count_e`) |
| `ionization_rate_per_m3_s` | ionisation map ("flames" at the cusp planes) | phase fraction (non-negative) | field target (log1p); `S` = its volume integral -> scalar / field consistency term |
| `sample_count_e` | accumulated macro-particle samples per node | - | **per-node label variance** (shot noise ~ 1/N): the heteroscedastic target variance is *given*, not learned |
| `wall_electron_flux_per_m2_s`, `wall_ion_flux_per_m2_s`, wall mean energies, side / exit current densities (1-D along the wall / exit) | wall profiles | surface queries | 1-D targets; the closure targets are integrals of these |
| `iedf_*`, `plume_*` | ion energy distribution, angular plume current | - | plume only; not until qualified |

Plus `series.npz` (61 time series at 200-step cadence: currents, counts, ledgers, peak-node
statistics) and, for the newer runs, 28 ns frame interval averages (100-300 per run). The series
and frames are the transient; the surrogate target is the plateau. Frames may serve as
*augmentation* for an operator that also learns the approach to plateau, but every frame of a
design shares the design's split (leakage rule) and their information content is far below their
count (autocorrelated).

**Grids differ per design.** Cells are `dr = r_w / round(r_w / Delta)`, `dz = L_ch / round(L_ch / Delta)`
(mini-sweep composer): 66-123 radial x 513-778 axial at 33 um. An FNO tolerates varying sizes by
mode truncation, but the cleanest contract is a **canonical normalised grid** `(r / r_w, z / L_ch)` at
a fixed resolution (e.g. 128 x 1024) with the mask and the physical scales as inputs, and a
recorded, invertible resampling (bilinear on node values; conservative for `ionization_rate`) whose
own error is measured against the native grid before it is used.

### 2.4 The estimand, stated once

The welding surrogate approximates a *trajectory* operator. Ours approximates a *steady-state
design* operator under a declared closure:

`F_closure, grid( design field map + masks, operating point ) -> ( plateau scalars, plateau fields, feasibility ) +- band`

where "plateau" means the trailing-window means under the recorded plateau rule (>= 3 ion
transits, trailing-20 % drifts of `I_d`, `N_e`, `n_g` < 5 %, triad soft bounds, peak-Debye soft
margin), "band" is the recorded seed + W spread at that rung, and the grid rung is a fidelity, not
noise.

---

## 3. Data reality

### 3.1 How many PIC plateaus exist (2026-09-05 01:00 AEST)

| record | design | grid / W | verdict / status | label status for a surrogate |
| --- | --- | --- | --- | --- |
| steady-state v2 base (`24ab82f4`) | reference (`divergent-exit-stack`, rho 0.60) | 50 um / 6e4 | plateau; classified **resolution-limited** by v4 (`0d228ad2`); with the ledger correction (section 3.2) it was **heating (~+13 %)** | low-fidelity rung only, flagged heating |
| v2 seed-b (`41ccb1ef`), W x0.7 (`542496fb`) | reference | 50 um | plateaus; <= 1.1 % seed spread, +5.7 % `I_d` / -12 % peak `n_e` at W x0.7 | the **only replicate pair** in the project -> the aleatoric band |
| steady-state **v4** (`0d228ad2`) | reference | 33.3 um / 2.667e4 | plateau at 3.03 transits, residual -7.7 %, `resolution_limited` verdict on the 50 um base; ss-v4 corrected end state ~+1.9 % | **qualified 33 um label** (pending the v5 caveat) |
| mini-sweep **047** (`b424ea37`) | `l1a-gs-v2-047` (rho 0.38) | 33 um / parity | plateau at 3.00 transits, `I_d` 1.925 mA, residual -7.1 % (corrected ~+2.6 %); `assess` / `targets` deferred | qualified 33 um label (assessment pending) |
| mini-sweep **009** | `l1a-gs-v3-009` (rho 0.92) | 33 um | plateau at 3.02 transits 23:59 AEST 09-04; **record commit pending** | qualified when recorded |
| mini-sweep **reference** | reference | 33 um | past 3 transits ~00:10 AEST 09-05 (numerical replication of v4 on the H100); record pending | replicate of v4 across GPUs (numerical, not bitwise) |
| mini-sweep **056** launch 2 (`ee35bc84`) | `l1a-gs-v3-056` (rho 2.36, HEMP-like) | 33 um | running, ETA 06:00-08:00 AEST 09-05 (launch 1 gate-stopped on a shot-noise artefact, `ccee5c60`, no plateau) | pending |
| mini-sweep 106, 056 seed replicate | sealed, not launched | 33 um | - | - |
| steady-state **v5** (`69ff435d`, launch 2 `ce1d96cb`) | reference | 25 um / 1.5e4 | running on the H100; verdict ~10-11 AEST 09-05 | decides whether 33 um carries a grid band |
| plume attempts 6 / 7 / 8 (`9cf7ca39`, `24ea2f65`, `ac248e05`) | reference, plume box | 50 um | ignited; **no plateau** (gate stop, budget stop, finite-grid heating) | feasibility-class labels only ("heats at 50 um plume") |
| external validation v0 (`036bd679`) | Brandt 2016 geometry | 20 um | stopped on the residual-power gate; genuine heating; **inconclusive** | not a label |

Count: **four distinct designs with a qualified or imminent 33 um channel plateau (reference,
047, 009, 056), all at ONE operating point (300 V, `n_g0` 5.5e19, 3 mA @ 2 eV), one closure (v1.3,
no recycling); one replicate pair at 50 um; zero plume plateaus; zero experimentally validated
points.** That is the whole training set. A 7-D surrogate needs on the order of 10 points per
dimension for an *initial* GP design (Loeppky, Sacks and Welch 2009, Technometrics 51:366-376,
doi:10.1198/TECH.2009.08040 - an informal rule they support with evidence, not a guarantee, and
one that assumes a smooth response without regime boundaries); four points fit nothing and
*cannot be gated* - the only things they can measure are the pipeline and the noise floor.

### 3.2 Cost per label (measured anchors)

| run type | anchor | wall time | GPU-h (solo-equivalent) |
| --- | --- | --- | --- |
| channel-only 33 um plateau, reference | ss-v4 on the RTX 5090, 5.2 M steps, 5.0 h; 047 on the H100 under MPS-4 6.91 h wall at 4.48 ms/step | 5-7 h | **4-7** (H100 ~1x a 5090 per process; MPS-4 gives 1.54x aggregate, so ~3.5-4.5 box-GPU-h per plateau when four run together) |
| channel-only 33 um, dense HEMP-like (056) | 16.6 ms/step MPS-4 projection, 3.65 M steps | 17 h MPS-4 | ~8-11 |
| channel-only 25 um (v5) | preflight 9.7-17.4 ms/step under contention, 7.2 M steps | 15-37 h | 15-35 |
| plume 50 um, 24 mm box (v2.1) | 8.2 ms/step, 7.6 M steps | 17-20 h | 17-20 |
| plume 33 um | 22.4 ms/step model, 8.2 M steps | 47-52 h | 47-52 |
| material-aware P2 field (input) | `produce_field` | 2-7 min CPU | 0 |
| orbit_mc test-particle wall loss (cheap fidelity, saturated labels) | screening v1: 96 designs / 95 min / 12 workers | ~12 CPU-min per design | 0 |

Money: the Lambda 8x H100 SXM box is USD 24/h (`tools/cloud/PLAN.md`), i.e. **USD 3 per GPU-h** when
packed; a 1x H100 is ~USD 3/h. The bill is the *makespan*, so a campaign of many equal channel
runs (32 concurrent under MPS-4 on 8 GPUs) packs well, a single plume run does not. The v2.0.5
step (`f80c6441`, ~x1.2 channel / x1.09 plume contended, more solo) and the GMG Poisson
(`9c2e4222`, +1.3x channel / 2.2x plume projected solo; not yet solo-probed) lower these numbers
but are not yet in any preregistered protocol - cost the plan at today's anchors and record the
saving when measured.

Per-design cost is not known a priori to better than a factor 2-4: the mini-sweep's particle
projections were 3.9x (047) and 4x (056) too high because the low-rho and HEMP-like designs sit
at 0.25-0.5 of the reference's mean density. The campaign scheduler must use *measured* early
ms/step per job (it does: `schedule.py status`) and budget at 1.5x.

### 3.3 The label-noise problem that killed the earlier surrogates, and how this plan handles it

What was rejected and why (all recorded; the user dropped the 0-D model and every prior surrogate
iteration on 2026-09-04):

| iteration | label | why it failed | lesson carried |
| --- | --- | --- | --- |
| L0 surrogates v1-v9 (`experiments/l0_surrogate*`) | outputs of a hypothetical analytic generator | no measured data; v9 tautology (the analytic leading term equals the target) | a surrogate of a model that has no accepted higher-fidelity output is not evidence; the useful object is a *discrepancy* to a higher fidelity |
| L1a field surrogates v1 / v2 (`l1a_field_surrogate_v*`) | vacuum field solves | localised interpolators underperformed the coarse solve across regime boundaries | do not interpolate across discontinuities with a stationary model |
| wall-loss geometry surrogate v1 (`b400d924`) | binomial P(wall) from 128 launches per cell | pooled RMSE 0.0562 vs gate 0.05; signal on step-discontinuous selectors (`stage_count_selector`, `exit_length_fraction`) a stationary GP cannot represent | derived physical features; per-category models; tree baseline |
| wall-loss geometry surrogate v2 (`a2b503be`) | same | pooled 0.0337 (pass), cells 0.0904 (fail); **ridge = GP = trees within noise**; learning curve flat from 30 designs; cell floor 0.035 = 70 % of the gate | **label-noise-limited, not model-limited**; gate against the replicate floor; delta-method logit noise breaks at saturated counts |
| screening v2 -> surrogate v3 / MDO v3 | 4x launches | **all 181 interior cells saturated at P(wall) = 1.0** | "a saturated label is a finding that ends a chain, not a dataset to fit" -> move the closure source to the PIC |

The PIC labels have a *different* noise structure, and the plan is built on it:

1. **The aleatoric band is measured, not assumed.** Seed replicate <= 1.1 %; particle-weight
   replicate 5.7 % (`I_d`), 4.6 % (`S`), 4.0 % (`n_g`), 11.9 % (peak `n_e`), 9.3 % (`T_e,peak`) -
   the "5-12 % particle-resolution band" - at 50 um on one design. The band is *heteroscedastic
   across quantities* (peak quantities twice as noisy as integrals) and, by the 047 / 056 particle
   counts, will vary across designs. Handling: every label row carries its own variance; the
   scalar model is stochastic kriging with *known* per-row noise (`ExactGP` per-row heteroskedastic
   measurement variance; BoTorch `SingleTaskGP` with `train_Yvar`) - the structure surrogate v2
   already had right - fit in **log space** for positive quantities so the relative band becomes
   additive and no delta-method fails at a bound.
2. **The grid rung is a bias, not noise.** 50 -> 33 um moved `I_d` +10.4 %, peak `n_e` -21 %,
   `T_e,peak` -24.5 %: larger than the W band and one-signed (less finite-grid heating).
   Handling: the rung is a fidelity id; labels from different rungs never share a row; the
   33 um labels are the primary set; 50 um records enter only as the low-fidelity branch of a
   residual model (`TwoFidelityAR1`: `high = rho x low + delta(x)`), and the whole surrogate
   carries the v5 verdict as its claim caveat ("at 33 um, converged / resolution-limited /
   uncertified", the mini-sweep's acceptance (f) verbatim).
3. **Only qualified runs are labels.** Qualification = plateau rule held AND residual-power
   acceptance (b) AND the peak-Debye soft margin AND the ledger *as corrected*: the
   `inelastic_loss_j` macro-weight bug (scratchpad 2026-09-05 00:50) biased every recorded
   residual negative by the inelastic power (7-14 % of electrode work) - v2 base was heating
   (+13 %), ss-v4 ~+1.9 %, 047 ~+2.6 %, 056 L1 ~+0.7 %. Until the ledger fix lands and the
   corrected residual is recomputed for each record, the "qualified" flag is *provisional* and
   the dataset stores both the recorded and the corrected residual. Gate-stopped, heating and
   no-plateau runs are **not** regression labels; they are *classification* labels (section 4.3).
4. **Replicate policy** (Binois, Huang, Gramacy and Ludkovski 2019, already cited in
   `docs/literature/surrogate-mdo-validation-blockers.md` section 1.2): one seed replicate per
   design family (HEMP-like / mid / low rho) per campaign batch, one W x0.7 replicate per batch,
   and replication preferred over exploration wherever the predicted noise-to-signal ratio is
   high (peak quantities). Replicates share the design's split. Expect ~15-20 % of the budget in
   replicates.
5. **Band-aware gates**, not fixed ratios: scalar RMSE on held-out designs <= 1.5x the pooled
   replicate floor of that quantity; 90 % interval coverage in [0.85, 0.97] with the binomial
   standard error of the coverage estimate (`active_learning.calibration.coverage_diagnostics`);
   the split-half reliability ceiling of the review's recommendation (attainable `R^2` bounded by
   the replicate reliability; useful = recovers >= 70 % of it). No "2x the best baseline".
6. **Regime discontinuities are modelled first, then the regression.** Ignition / no ignition,
   avalanche (`nu_iz tau > 1`), finite-grid heating (peak `Delta / lambda_D` past pi at the planned
   grid), plateau reached: a design in a non-plateau regime has no plateau scalar. The surrogate
   therefore has a feasibility classifier with its own calibrated probability, and the
   regression is conditioned on feasibility (the welding model's "stochastic event heads").
7. **Derived physical features, always, and a tree / linear baseline, always** (the v2 lesson):
   per-cusp rho under iron, wall `|B|` at each cusp, `x_w`, cusp count, channel length `L_ch`, `r_w / L` (L = PPM pitch, the sweep-v3 convention), taper length,
   `V`, `Q_in`; raw design selectors only as a secondary check.

### 3.4 The design space

Geometry (7 nominal axes the brief names, mapped onto the existing v1.1 parameterisation):

| axis | v1.1 variable(s) | range sampled so far (sweep v3 box) | note |
| --- | --- | --- | --- |
| Koch rho (per cusp, under iron) | derived from `chamber_outer_radius_m`, `stage_pitch_m`, `source_strength_scale`, magnet dimensions | 0.2-2.9 (catalogue); PIC has 0.38 / 0.60 / 0.92 / 2.36 | the primary physical coordinate; not an input variable but a *feature* |
| `r_w / L` (wall radius to PPM pitch; `L` = pitch here, as in sweep v3) | `chamber_outer_radius_m / stage_pitch_m` | 0.215-1.235 | HEMP threshold ~0.62-0.75 |
| pitch | `stage_pitch_m` | 3.4-6.5 mm | |
| exit taper | `exit_length_fraction`, `exit_expansion_descriptor` | 0-0.28 of L; 0-1 | the mini-sweep avoided short steep tapers (mesh slivers, L1b 028 / 048); the PIC mesher represents straight bore + linear cone exactly |
| N stages (cusps = N - 1) | `stage_count_selector` -> 3-5 | 3 / 4 / 5 | a **discrete** selector: per-count models or a one-hot / latent-variable input; the 4-stage / 3-cusp family maps 1:1 onto the recorded closure |
| magnet strength | `source_strength_scale` | 0.75-1.3 | scales `\|B\|` linearly (linear iron) |
| others | `magnet_axial_fraction`, `dielectric_thickness_m`, `radial_clearance_m`, `magnet_radial_thickness_m`, `first_polarity_selector` | as sweep v3 | keep the sweep-v3 box so the P2 field production, mesh gates and topology definition apply unchanged |

Operating point (never varied in a PIC run so far): `V` (300 V; propose 200-400 V), neutral feed
(equal `n_g0` 5.5e19 today; the campaign must *declare* whether it holds `n_g0`, `Q_in` or
`Q_in / A_exit` constant across designs - the mini-sweep chose `n_g0`), injection / cathode
current (3 mA; the plume's flux-tube cathode gave ~6 mA uncapped). Each operating-point axis is a
new dimension the labels have never seen; ignition and heating admissibility change with it (PIC
operating points must be sized from the measured `tau_i,eff`, scratchpad).

Feasibility gates that must run over the WHOLE sampled set before any prereg (the L1b v1 lesson):
level-0 mesh angle >= 5 deg; `omega_ce dt` <= 0.19 at the composed `dt`; projected peak
`Delta / lambda_D` at the planned grid <= 2.5 (from a cheap density estimate or from the nearest
labelled design); cathode connectivity (plume); projected particles <= the 12 M cap; projected
wall time inside the batch budget.

### 3.5 Sample-efficient acquisition plan (multi-fidelity, honest counts)

Cheap fidelities we have, and what they are good for:

| level | cost | usable as | NOT usable as |
| --- | --- | --- | --- |
| geometry descriptors + material-aware P2 field + topology (cusps, rho, wall `\|B\|`) | minutes CPU | **inputs / features** for every design; the sampler's feasibility screen; hundreds of designs are already catalogued (sweep v2 / v3, 224 designs) | a label of anything plasma |
| orbit_mc collisionless wall-loss probability (screening v1 / v2) | ~12 CPU-min per design | a *feature* where it is not saturated (exit-side partial cells 0.17-0.87) | the low-fidelity branch of an AR1 for cusp loss: its interior cells are saturated at 1.0 (the finding that ended that chain) |
| 50 um channel PIC | ~2-3 GPU-h | low-fidelity rung of the grid residual (one design so far); *screening* runs to locate the ignition / heating boundary cheaply | a primary label (resolution-limited, heating with the corrected ledger) |
| **33 um channel PIC** | 4-11 GPU-h | **the primary label** | plume quantities |
| 25 um channel PIC | 15-35 GPU-h | the convergence witness for a few designs (the ladder), not a training set | - |
| plume PIC (50 / 33 um) | 17-52 GPU-h | `I_beam`, divergence, thrust - **once a plateau exists** | anything today |

How many primary labels a *first useful* surrogate needs (be honest):

| target | dimensions | designs (labels) | replicates | GPU-h at 4-11 per label | USD at 3/GPU-h | wall on 1 H100 (MPS-4, 1.54x) | wall on 8 H100 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| **A. trend surrogate**: 4-cusp family, one operating point, 3-4 geometry axes (rho, `r_w / L`, pitch, taper) | 4 | 24-30 (the 4 existing + 20-26 new) | 4-6 | **120-250** | 400-750 | 3.5-7 days | 0.5-1 day |
| **B. design surrogate**: + `N` stages (3 / 4 / 5) + `V` + feed -> 7-D | 7 | 70-100 | 10-15 | **400-800** | 1.2k-2.4k | 11-22 days | 1.5-3 days |
| **C. field operator** on B's maps | - | needs B's 80-115 map sets; no extra PIC | - | +5-10 (training) | ~30 | hours | hours |
| **D. plume / thrust heads** | +plume domain | 15-25 plume plateaus at 20-50 GPU-h **after** the plume model is qualified | 3-5 | **400-1200** | 1.2k-3.6k | weeks | 2-6 days |

So: **30 plateaus (~USD 500) buys a gated trend model on the family we already run; 70-100
plateaus (~USD 1.5-2.5k, 300-800 GPU-h) buys the first surrogate that can sit inside the
optimiser for the channel quantities; thrust needs the plume model first and roughly doubles the
bill.** These numbers assume the ignition / heating boundary does not swallow a large fraction of
the samples - every non-plateau run is a spent label that only feeds the classifier; the sampler
must stay inside the admissible region learned from the first batch (Bayesian active learning on
the classifier, `active_learning.acquisition.score_candidate` with the feasibility probability
term).

Acquisition, in order: (i) a space-filling first batch (scrambled Sobol, as the sweeps; 16
designs) on the 4-cusp family with a 10 % boundary challenge (`optimization/sampling.py`), (ii)
replicates where the first-batch noise-to-signal is worst, (iii) constrained multi-objective
batch acquisition on the GP posterior (qLogNEHVI / qLogNParEGO, `botorch_adapter.py`) with the
feasibility probability as a constraint and the `HighestFidelityQuota` of
`active_learning.acquisition.select_fidelity` reserving 33 um slots, (iv) a small number of 25 um
and 50 um rungs on designs chosen by the AR1's discrepancy variance, never uniformly. The stopping
policy already exists (`active_learning.stopping.StoppingPolicyV14`: F3 count, F3 fraction,
verified-hypervolume stall, calibration checked, no pending work).

---

## 4. Architecture proposal for our case

### 4.1 Reuse vs adapt vs drop (against `Reality-Simulator/ai`)

| welding component | decision | our form |
| --- | --- | --- |
| registry-driven layout, fail-closed unknown state, hashed lineage | **reuse the principle**, implement on our stack | a `pic2d` record registry: semantic ids for every `summary.json` scalar, `maps.npz` array and closure target, with unit, support (scalar / 2-D node / 1-D wall / 1-D angular), disposition (target / feature / gate / provenance); compiled to a layout hash; unknown keys in a record fail ingestion closed; built on `experiment_runtime.canonical` / `filesystem.verify_pair` |
| Zarr episode store | adapt | our records already are the store (hash-bound `results/` trees, LFS for arrays); the dataset is a **manifest over records** (paths, byte hashes, `config_sha256`, `protocol_sha256`, commit, verdict, label status) plus derived canonical-grid arrays written once with their own hashes |
| whole-scenario split | reuse | split by *design id*; all rungs, seeds, W, frames, closures of a design in one split; `surrogates.validation.grouped_spatial_split` enforces the transitive closure |
| normalisation fit on train only, unit-aware | reuse | log for positive quantities; per-quantity bands in the same space |
| campaign generator (LHS, scenario vs programme scope, replicates, queue) | reuse the shape | design-scope axes (geometry) vs operating-point axes; batch = one preregistration; emits per-design sealed protocols (the mini-sweep composer `protocol.py` already does this for 5 designs) + a `jobs.yaml` block for `tools/cloud/schedule.py` |
| FNO backbone + heteroscedastic heads | **adapt to 2-D**, use only at Phase 3 | 2-D spectral convolution on `[batch, channels, r, z]`; the 3-D code is a direct template |
| U-Net / TFNO / U-FNO ladder | defer | only if the FNO fails the Phase-3 gates and > 100 maps exist |
| GINO point-to-grid geometry encoder | **not needed** | our geometry is already a field on the grid; the P2 solve is the encoder |
| material-function encoder | drop (fixed propellant) | - |
| GRU control-sequence encoder | drop | operating point is a static vector: MLP context |
| ConvGRU latent rollout, multi-rate dynamics | drop | steady-state estimand |
| objective-only / sparse-query / full-field modes | reuse | objective-only = the scalar heads (and at level (a) the GP itself); full-field = the operator; sparse-query = wall-profile queries (leak width, sheath drop) if useful |
| deep ensemble (epistemic) | reuse | 3-5 members for the network; the GP posterior for level (a) |
| explicit multi-fidelity residual | reuse | grid rung and closure version as fidelity ids (`TwoFidelityAR1`; BoTorch `MultiTaskGP` with the rung as task and inferred task noise - the adapter documents that BoTorch 0.18.1 cannot take differing known noise per task) |
| PINO losses (energy, mass, constitutive, sensitivity) | adapt | discrete Gauss law on `(phi, n_i - n_e)`; ion ledger `e S = I_wall,i + I_beam + I_anode,i`; `S` = integral of the ionisation map; monotone sensitivity checks (`dI_d / dV > 0`, `dS / dQ_in > 0`) as *tests*, and the Jacobian-vs-paired-runs sensitivity loss once paired runs exist |
| `RiskAwareObjective` + CVaR, `VerificationQueue`, disagreement -> active learning | reuse the logic on our stack | `cft_revival.optimization` campaign with the surrogate as an F2 source and the PIC as F3; promotion requires F3; the welding CVaR term maps onto `active_learning.robustness.cvar` |
| real-time student, distillation, anomaly gate | drop | - |

### 4.2 Model family, by data regime (the ladder starts at the bottom)

**Level (a), scalars - N < 100 labels: Gaussian processes on physical features, not networks.**

* Inputs: per-cusp rho under iron (or a 3-slot vector for the 4-stage family; a latent-variable /
  one-hot for N stages), wall `|B|` at each cusp, `x_w`, `r_w`, channel length `L_ch`, taper length, `V`, `Q_in`
  (and `n_g0`), injection current; raw v1.1 variables as a secondary feature set for the
  tautology / leakage checks.
* Model: `surrogates.ExactGP` / `IndependentMultiOutputGP` (Matern-5/2 ARD, per-row known
  heteroskedastic noise = the replicate band of that quantity at that rung) on log targets;
  BoTorch `SingleTaskGP` with `train_Yvar` for the acquisition; `TwoFidelityAR1` when the 50 um
  branch is used. Baselines fitted and reported every time: mean, ridge on the same features,
  gradient-boosted trees, k-NN. If ridge equals the GP, say so (it did in v2).
* A **feasibility classifier** alongside: a GP classifier or calibrated logistic model on the same
  features for `P(plateau at this rung)`; trained on every run including gate stops.
* What it is for: the optimiser's objective-only mode, the campaign's acquisition, and the
  measured noise floor.

**Level (b), fields - N >= 50-100 map sets: a 2-D operator, with a POD-GP baseline.**

* Baseline first: `surrogates.PODFieldSurrogate` on the canonical grid (POD over log-density /
  potential / log-ionisation maps; GPs on the modal coefficients; mesh hash bound) - it is the
  right model for tens of snapshots and it gives pointwise variance.
* Then the network: 2-D FNO (adapted from `Reality-Simulator/ai/src/reality_ai/surrogate/fno.py`)
  with inputs `[B_r, B_z, |B|, masks, r / r_w, z / L_ch]` + tiled operating-point context, outputs
  `[log n_e, log n_i, phi / V, T_e, log1p S_map]` each with a log-variance head; masked Gaussian
  NLL where the *target variance floor* per node is the shot-noise variance from `sample_count_e`
  (band-aware by construction); physics terms as in 4.1; ensemble of 3-5 (different seeds and
  member-wise bootstrap of designs). U-FNO only if the FNO fails on interfaces (sheaths are 1-3
  cells wide: the same "local interfaces vs global coupling" argument the welding ladder makes).
* Scalar consistency: the integrals of the decoded fields (S, N_e) must agree with the scalar
  heads within the band; disagreement is a diagnostic the dashboard shows.

### 4.3 Conditioning, uncertainty and the feasibility head

* **Conditioning on the field map**: static channels (4.2) plus the topology scalars. The
  canonical-grid resampling is a recorded, tested transform (its error against native-grid
  integrals is part of the dataset validation). Physical scales (`r_w`, `L_ch`, `dr`) enter as
  scalars so the operator is not asked to infer dimensional information from a normalised image.
* **Uncertainty decomposition** as in `active_learning.uncertainty.decompose_prediction`:
  emulator (GP posterior / ensemble spread) + aleatoric (replicate band) + rung discrepancy
  (AR1 delta) + bias-estimation variance, combined in quadrature under the declared
  zero-covariance policy; never one unexplained sigma. The rung caveat (v5 verdict) is a
  *statement*, not a variance.
* **Feasibility**: `P(ignites)`, `P(plateau)`, `P(heating at the planned grid)` from all runs;
  regression conditioned on feasibility; the optimiser treats `P(plateau) >= 0.95` as a chance
  constraint (the promotion rule in `optimization/pareto.py` already requires every marginal
  probability above threshold).

### 4.4 Training pipeline (reads hash-bound records through `experiment_runtime`)

```
records (results/ trees: summary.json, assessment.json, maps.npz, series.npz,
         checkpoint-final.json, execution-lock.json, protocol.json, closure targets .json)
  -> ingest: verify every sidecar (`filesystem.verify_pair`), strict JSON (`canonical.strict_json_file`),
     record registry check (unknown key -> fail closed), verdict + corrected-ledger qualification
  -> label ledger (one row per record x quantity: value, band, rung, closure, seed, W, verdict,
     label_status in {qualified, provisional_ledger, classifier_only})
  -> canonical-grid arrays (+ resampling error record) for the field level
  -> dataset manifest: record paths + byte hashes + config/protocol sha256 + commits + registry hash
     + normalisation hash + split assignment (by design id) -> one canonical SHA-256
  -> train (GP: CPU, seconds; POD-GP: CPU, minutes; 2-D FNO ensemble: local 5090 or one H100-hour)
  -> checkpoint embeds the manifest / registry / normalisation / split identities (welding rule)
  -> evaluate only against a bundle with identical lineage; write assessment.json under the gates
```

Everything except the network training runs on the CPU, in `.venv-sota` (torch / BoTorch already
provisioned; nothing installed globally), never on the local GPU while a preregistered run holds
it, never as a sub-process on the H100 beside PIC clients without a slot.

### 4.5 Evaluation protocol (consistent with the repository's evidence rules)

1. **Preregistered held-out designs**: before any fit, a maximin subset of >= 20 % of the labelled
   designs (whole designs, all rungs / seeds / frames) is sealed in the protocol as the assessment
   role; a second small extrapolation role outside the training hull is *reported, not gated*
   (the surrogate v1 / v2 partition design, inherited by hash).
2. **Gates in physical units against the band**: for each gated scalar, held-out RMSE (log space)
   <= 1.5x the pooled replicate floor of that quantity at that rung; 90 % coverage in
   [0.85, 0.97] with its binomial standard error; the reliability-ceiling check; the classifier's
   Brier score and calibration. Field level: masked band-normalised RMSE per quantity, Gauss-law
   residual and integral consistency inside declared bounds, POD-GP must be beaten or the POD-GP
   is the model.
3. **No tautology / leakage**: inputs are geometry, field and operating point only; nothing derived
   from the plateau; the `no_tautology` and single-use-label gates of the surrogate experiments
   verbatim; method selection by nested cross-validation or a single declared model (Cawley-Talbot).
4. **Claim envelope**: the convex hull of the sampled design box at the sampled operating points,
   one closure, one rung, with the v5 verdict quoted; `surrogates.validation.OODDetector` flags
   every query outside it and the optimiser is refused there (guardrail, not a warning).
5. **Verdicts**: `accepted_surrogate_for_channel_quantities_at_33um` / `rejected_surrogate` /
   `noise_floor_only` (Phase 1's expected verdict), each with what the next version needs, as in
   `wall_loss_geometry_surrogate_v2/README.md`.

### 4.6 How it plugs into the optimisation loop

`cft_revival.optimization` already defines the loop: F0 (cheap analytic / descriptors) -> F1
(fields / reduced) -> F2 -> F3 (PIC). The surrogate is the **F2 source**; the PIC campaign is F3.

* Objectives (channel-quantity set until the plume is qualified): maximise utilisation and
  `I_beam / I_d`, minimise the ion wall-loss fraction and the diffuse wall electron current,
  minimise peak `n_e` relative to the grid's admissibility (or treat it as a constraint), subject to
  `P(plateau) >= 0.95`, sheath / `T_e` bounds and the manufacturability gates of geometry v1.1.
  Thrust / divergence / `Isp` enter only when a plume plateau and a qualified plume model exist.
* Acquisition: constrained qLogNEHVI on the F2 posterior with the feasibility constraint; MORBO
  trust regions if the box is large; qLogNParEGO fallback (`spec/optimization/campaign-v1.json`).
* Promotion: rank-zero, robust feasibility (`mean + k sigma`), every marginal probability >= 0.95,
  guardrails (`optimization/guardrails.py`: normalised nearest-training distance, combined
  uncertainty, conservation flags), **then F3 verification by a preregistered PIC batch**. A
  surrogate-PIC disagreement beyond the band is recorded (`VerificationStatus.DISAGREEMENT`
  analogue) and becomes the next batch's highest-priority sample.
* Reporting: the Pareto front is drawn from F3-verified points only; surrogate hypervolume is
  never reported as verified hypervolume (anti-pattern list in `optimization-architecture.md`);
  at least two closures / rungs are compared before any "design X is best" sentence (the MDO v2
  Jaccard-0 lesson).

---

## 5. Phased plan, prerequisites, costs, risks, claims

### Phase 0 - now (in progress; prerequisites, no surrogate work)

* PIC physics completeness and correctness: the ledger `inelastic_loss_j` macro-weight fix and the
  particle-side identity check; occupancy floors in accumulated particle-steps on every density
  gate; wall-ion recycling as a declared closure; the plume model qualified (a plateau at an
  admissible operating point / grid); STOP-file / SIGTERM handler in the shared runner.
* Qualified fast solver: v2.0.5 measured solo; GMG Poisson solo probe and its v4 replay campaign;
  a re-priced cost model.
* Convergence ladder: v5 verdict (25 um) -> the rung caveat every 33 um label carries.
* Mini-sweep closure: records for 009 / reference / 056, the sweep-wide `assess` citing v4, and
  `targets` for all four - the first four rows of the label ledger.
* Freeze the **record contract** the ingestion will read: schema versions of `summary.json`,
  `assessment.json`, `maps.npz` keys, the closure-target JSON; any later change bumps the version
  and the ingestion refuses unknown keys.
* Cost: already budgeted elsewhere (running); no new GPU-h for the surrogate.
* Exit criterion: >= 4 qualified 33 um labels with corrected residuals and a frozen contract.

### Phase 1 - dataset schema, ingestion of existing records, scalar baseline to measure the noise floor

* Build `cft_revival/pic_dataset` (or `surrogates/pic_records.py`): the record registry, the
  ingestion, the label ledger, the dataset manifest with lineage hashes, the canonical-grid
  resampler with its error record, split assignment by design id. Tests: fail-closed on a
  tampered sidecar, an unknown key, a mixed rung, a design straddling splits.
* Noise-floor study on what exists: seed-b / W x0.7 / v4-vs-H100-replication pairs (reference),
  the 056 seed replicate when run; per-quantity band table in log space; the split-half reliability
  where two replicates exist. This is the number the gates will be built on.
* Scalar baseline: GP + ridge + trees + mean on the 4-6 labelled designs, **labelled
  `noise_floor_only` / plumbing evidence**, not a surrogate claim; leave-one-design-out error
  reported against the band to show what four points cannot do.
* Cost: CPU only, ~1 week of work; 0 GPU-h.
* Exit: the manifest reproduces byte-for-byte from the records; the floor table is recorded; the
  campaign generator (Phase 2) knows which quantities need replicates.

### Phase 2 - PIC campaign generator on the H100 (design sampler, prereg per batch, scheduler)

* Sampler: scrambled Sobol on the declared box (4-cusp family first; then N stages, `V`, feed),
  the whole-set feasibility screen of section 3.4 before the prereg commit, boundary challenges,
  replicate allocation from Phase 1's floor.
* Per batch (8-16 designs + 2-4 replicates): one preregistration (protocol.json binding the sealed
  per-design protocols, preflight, shakedown on ONE real design through run -> finalize -> assess
  -> targets, MPS replay), launched through `tools/cloud/schedule.py` at MPS-4 per GPU; results
  committed from the job worktrees on `results/<id>` branches; the ingestion of Phase 1 runs on
  each batch as it lands. Never kill a client under MPS; budgets at 1.5x the measured early rate.
* After batch 1 (16 + 4 runs, ~80-160 GPU-h, USD 250-500, 2-4 days on one H100 under MPS-4 or
  under a day on eight): the first *gated* scalar attempt (target A of 3.5). After batches 2-4 (operating-point
  axes; cumulative 70-100 labels, 400-800 GPU-h, USD 1.2k-2.4k): target B.
* Exit: the acquisition loop (GP posterior -> next batch) runs from the label ledger with the
  stopping policy's diagnostics visible; the scalar surrogate has a verdict under the Phase-1 gates.

### Phase 3 - field-operator surrogate

* POD-GP baseline on the canonical-grid maps of Phase 2, then the 2-D FNO ensemble with the
  shot-noise-weighted NLL and the Gauss-law / integral consistency terms; U-FNO only on a measured
  failure at the sheaths. Training on the local 5090 (minutes to an hour; the maps are ~1 M nodes x 5
  channels per design) or one H100-hour; never beside a preregistered run.
* Gates as 4.5; a dashboard of predicted vs PIC maps on the held-out designs with the band shown.
* Cost: ~5-10 GPU-h; the data are Phase 2's.
* Exit: a verdict; if rejected, the POD-GP or the scalar GP remains the optimiser's F2 source.

### Phase 4 - optimisation with UQ and held-out PIC confirmation

* The campaign of 4.6 on the accepted F2 source: a constrained multi-objective acquisition
  proposes 8-12 candidates per round; each round is a preregistered PIC batch (F3); disagreement
  feeds the next round; two rounds minimum before any front is drawn.
* Plume objectives only if Phase 0's plume qualification succeeded and target D of 3.5 has been
  paid for; otherwise the channel-quantity objective set, declared as such.
* Cost: 2-3 rounds x 10 designs x 4-11 GPU-h = 100-300 GPU-h (USD 300-900) for channel
  confirmation; plume confirmation 20-50 GPU-h per point on top.
* Exit: an F3-verified Pareto set with bands, the surrogate-vs-PIC disagreement table, and the
  claim envelope; the paper admission would be a `numerical-campaign` gate with the surrogate as
  a *tool* of the campaign, not as a finding.

### Cost summary

| phase | GPU-h | USD (3/GPU-h) | wall (1 H100, MPS-4) | wall (8 H100) |
| --- | --- | --- | --- | --- |
| 0 | (already running) | - | - | - |
| 1 | 0 | 0 | 1 week CPU | - |
| 2 | 300-800 | 900-2,400 | 8-22 days | 1-3 days |
| 3 | 5-10 | 15-30 | hours | hours |
| 4 (channel) | 100-300 | 300-900 | 3-8 days | 0.5-1 day |
| 4 (+ plume) | +400-1,200 | +1,200-3,600 | weeks | 2-6 days |
| **first useful loop (1 -> 4 channel)** | **~400-1,100** | **~1.2k-3.3k** | **~2-4 weeks** | **~2-4 days** |

### Risks

| risk | evidence | mitigation in the plan |
| --- | --- | --- |
| the 33 um rung is itself resolution-limited (v5 pending) | v4 moved `I_d` +10 %, peak `n_e` -21 % vs 50 um | labels carry the rung caveat verbatim; the surrogate is "of the 33 um model"; a few 25 um rungs enter the AR1 discrepancy, not the training set |
| the ledger correction reclassifies "accepted" plateaus as heating | v2 base +13 % corrected | provisional label status until recomputed; only corrected-qualified rows train |
| per-design noise band wider than the reference's | 047 / 056 at 0.25-0.5 of the reference density, fewer particles per node | per-row measured variance; replicates allocated to the noisiest family |
| regime boundaries eat the budget (no ignition, avalanche, heating) | plume attempts 3-8; ext-val v0 | feasibility screen before prereg; classifier-guided sampling; 50 um screening runs for the boundary |
| operating-point axis changes admissibility | `nu_iz tau` sizing lesson; heating at 20 um / 8.6x parity W | size every new operating point from the measured `tau_i,eff` and the projected `Delta / lambda_D` at the planned grid; W parity is a resolution requirement, not a cost knob |
| label count too small for the model family (over-fitting, unstable selection) | v1 / v2 selection instability on 10 designs | GP first; nested CV or one declared model; the reliability ceiling; N stages as a discrete factor with per-count models |
| cost model wrong by 2-4x | mini-sweep projections 3.9-4x high | measured early ms/step per job; 1.5x budgets; batch sizes set from the previous batch |
| the optimiser exploits the surrogate | welding paper's central risk; MDO v2 closure dependence | mandatory F3 verification; chance constraints; OOD refusal; two closures / rungs before any ranking sentence |
| cross-platform hash drift (Windows vs Linux ULP) | field-map hash `d30d2d24 -> 1f124047` | dataset manifest hashes stored bytes, not derived arrays; scale-aware tolerances for derived maps |
| H100 operations (MPS Xid 31, no clean stop) | 2026-09-04 incidents | scheduler rules of `PLAN.md` / mini-sweep 8.3; STOP-file handler in Phase 0 |
| the plume model is not qualified in time | no plume plateau after 8 attempts | thrust heads are gated behind qualification; the channel objective set is the declared fallback |

### What must NOT be claimed (at any phase)

* Thruster performance - thrust, `Isp`, efficiency, discharge power - from any model trained on
  channel-only records; nor from plume records until a plume plateau under an accepted residual
  exists. Any "development" thrust number with > 5 % residual power is non-quotable.
* "Converged" for any grid rung before the ladder says so; the surrogate inherits the rung's
  verdict and says "at 33 um, <verdict>".
* Experimental validity: the PIC is not externally validated (ext-val v0 inconclusive); the
  surrogate is a surrogate of *this* model under *this* closure (electrostatic, axisymmetric, no
  anomalous transport by construction, 0-D neutral inventory, no SEE).
* A design rule or ranking from the four mini-sweep points, or from any front not verified at F3;
  a single-closure ranking (MDO v2: CL-1 vs CL-2 Jaccard 0).
* Surrogate uncertainty as physical uncertainty: it is emulator + numerical band; model-form
  uncertainty is declared, not estimated.
* Anything outside the sampled envelope (design box, operating points, closure, rung); the OOD
  guard refuses rather than extrapolates.
* That the surrogate "replaces" the PIC: it accelerates the search; every promoted design is a
  PIC run. The welding docs' governance sentence applies verbatim: no replacement claim without
  envelope, report, exact checkpoint, dataset provenance and fallback.

---

## 6. File references (target repository)

* Surrogate runtime: `modern/src/cft_revival/surrogates/{gp.py, multifidelity.py, pod.py, validation.py, benchmark.py, interop.py}`; contract `modern/spec/surrogates/runtime-v1.json`; `modern/docs/workstreams/surrogates-architecture.md`.
* Active learning / UQ: `modern/src/cft_revival/active_learning/{acquisition.py, calibration.py, uncertainty.py, robustness.py, stopping.py, contracts.py}`; `modern/docs/workstreams/active-learning-foundation.md`.
* Optimisation: `modern/src/cft_revival/optimization/{campaign.py, domain.py, pareto.py, sampling.py, guardrails.py, botorch_adapter.py, spec.py}`; `modern/spec/optimization/campaign-v1.json`; `modern/docs/workstreams/optimization-architecture.md`.
* Validation / evidence: `modern/src/cft_revival/validation/{contracts.py, evidence.py, metrics.py, workflow.py}`.
* Experiment runtime: `modern/src/cft_revival/experiment_runtime/{lifecycle.py, canonical.py, filesystem.py, recovery.py}`; `modern/docs/workstreams/experiment-runtime-architecture.md`.
* PIC records and their writers: `modern/src/cft_revival/pic2d/{artifacts.py, simulation.py, fields.py, mesh.py, poisson.py, mcc.py, frames.py}`; `modern/experiments/pic2d_cft_steady_state_v1/run.py` (shared runner), `pic2d_cft_steady_state_v4/run.py` (`assess`), `pic2d_cft_steady_state_v5/`; `modern/experiments/pic2d_design_mini_sweep_v1/{closure.py, protocol.py, designs.py, fields.py, cost.py, preflight.py}`; `modern/spec/pic2d/pic2d-model-v2.0.json`.
* Geometry and fields: `modern/src/cft_revival/geometry/{model.py, descriptors.py, generators.py, adapters.py}`; `modern/experiments/l1a_geometry_sweep_v3/{protocol.json, designs.py}`; `modern/experiments/cusp_topology_search_v3_1/`; `modern/experiments/l1b_hemp_confirmation_v1_1/`.
* Cloud execution: `modern/tools/cloud/{PLAN.md, schedule.py, jobs.yaml, bench_gpu_concurrency.py, bootstrap_lambda.sh}`.
* Prior surrogate record: `modern/experiments/wall_loss_geometry_surrogate_v1/`, `_v2/`, `modern/experiments/l0_surrogate_v9/`, `modern/experiments/l1a_field_surrogate_v2/`; the reasons in `modern/docs/literature/surrogate-mdo-validation-blockers.md` sections 1.1-1.4.
* Performance and acceleration: `modern/docs/pic2d-performance-audit.md`; `modern/docs/literature/pic-acceleration-methods.md`.

## 7. File references (`Reality-Simulator/ai`, read-only)

`README.md`; `IMPLEMENTATION_STATUS.md`; `TODO.md`; `docs/ARCHITECTURE.md`; `docs/FULL_PHYSICS_SURROGATE_PLAN.md`;
`docs/SURROGATE_ARCHITECTURES.md`; `docs/DATA_PLANE.md`; `docs/PLANNING_AND_REALTIME.md`; `docs/TRAINING_PLAN.md`;
`docs/VALIDATION_PLAN.md`; `docs/DATASET_GENERATION_CAMPAIGN.md`; `paper/latex/sections/03_training_optimisation_control.tex`,
`04_protocol_status_conclusion.tex`; `configs/surrogate/{baseline.yaml, model_ladder.yaml}`;
`configs/datasets/{production.yaml, s355_gmaw_tfillet_pilot_campaign.json}`; `configs/optimisation/default.yaml`;
`src/reality_ai/surrogate/{model.py, fno.py, unet.py, tfno.py, ufno.py, backbone_factory.py, configurable_model.py,
control_encoder.py, objective_decoder.py, query_decoder.py, field_decoder.py, ensemble.py, multifidelity.py,
uncertainty.py, geometry_encoder.py, material_encoder.py, multiphysics_fusion.py, latent_dynamics.py}`;
`src/reality_ai/physics/{pino_losses.py, energy.py, mass.py}`; `src/reality_ai/training.py`;
`src/reality_ai/data/{physics_registry.py, layout.py, campaign.py, splits.py, episode_dataset.py, bundle.py,
manifests.py, npy_store.py, zarr_store.py}`; `src/reality_ai/common/normalisation.py`;
`src/reality_ai/optimisation/{planner.py, objectives.py, verification.py, surrogate_adapter.py, sampling_search.py,
gradient_search.py, hybrid_search.py, constraints.py}`; `scripts/{prepare_dataset_campaign.py, train_dataset_surrogate.py,
evaluate_dataset_surrogate.py, benchmark_surrogate_architectures.py, optimise_weld.py}`.

Literature the welding plan builds on and this plan inherits (arXiv ids as cited there): FNO
2010.08895; PINO 2111.03794; GINO 2309.00583; CoDA-NO 2403.12553; the repository's own reviews
cover stochastic kriging / heteroscedastic GPs, replication allocation, binomial-likelihood GPs
and multi-fidelity co-kriging (`surrogate-mdo-validation-blockers.md` bibliography A-B) and the
surrogate-assisted CFT optimisation lineage (Yeo and Ogawa 2022, `pic-mcc-blockers.md` ref. 113).
Added here: the initial-design sample-size rule, Loeppky, Sacks and Welch, "Choosing the Sample
Size of a Computer Experiment: A Practical Guide", Technometrics 51(4):366-376 (2009),
doi:10.1198/TECH.2009.08040.
