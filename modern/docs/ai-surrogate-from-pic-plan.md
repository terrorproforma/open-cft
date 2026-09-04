# AI surrogate trained on PIC-MCC outputs: plan (doc only, no build)

**Status: plan, not an implementation and not a result.** Written 2026-09-05 against
`feat/sota-foundation` at `036bd679` after reading (read-only) the user's welding AI subsystem
`Reality-Simulator/ai` and this repository's surrogate / active-learning / optimisation /
validation / experiment-runtime stack and the PIC records that exist today. **Revised 2026-09-05
against `8c70cff0`** after user feedback: (i) the `Reality-Simulator/ai` docs describe *two
competing model cores* (a Fourier/tensorised neural-operator core and a latent-token
*transformer* core), and the transformer-core idea traces to Arena Physica's Heaviside models -
section 1.5 documents the two cores and their decision criteria as written there, section 1.6
studies Heaviside-0 / Heaviside-1 / Atlas RF Studio from the public sources and states what they
change in our recommendation; (ii) **the order of work is fixed by the user**: (1) finish the 2-D
PIC physics, (2) design the 3-D PIC and verify that it works, (3) *then* the AI run - section 5
is reordered accordingly (Phase 0 2-D physics, Phase 1 3-D design + verification, Phase 2 data
plane, Phase 3 AI campaign, Phase 4 optimisation). Nothing in this document is preregistered;
every number about *our* data is quoted from a recorded artifact and every number about cost is a
projection with its anchor named. The brief: apply "the same principle and architecture" as the
welding reality model - a learned surrogate trained on simulator outputs, used to accelerate
design optimisation - to the PIC-MCC plasma simulator (`cft_revival.pic2d`, later a 3-D code).

The short answer, before the detail:

* **The principle carries over intact**: the simulator stays the oracle and the only source of
  labels; a registry-driven, hash-bound, fail-closed data plane feeds a surrogate with explicit
  aleatoric + epistemic uncertainty; the surrogate runs a cheap "objective-only" mode for search
  and a "full-field" mode for inspection; every surrogate-selected design goes back to the
  simulator for verification and every disagreement becomes an active-learning sample.
* **The architecture only partly carries over**: the welding model is a *time-stepping world model*
  (state + future controls -> future 3-D fields, GRU control sequence, ConvGRU rollout, a
  real-time sensor student). Our object is a *steady-state design operator* (geometry field map +
  operating point -> time-averaged plateau fields and their scalars). The neural-operator
  backbone, the heteroscedastic heads, the deep ensemble, the multi-fidelity residual and the
  verification queue map one-to-one; the rollout, control-sequence and real-time layers have no
  analogue and are dropped.
* **Heaviside changes three things in the recommendation, not the principle** (section 1.6.4):
  (1) *fields-first supervision* - the plateau maps, not the scalars, are the primary training
  target wherever maps exist, because Arena's own ablation is that dense field supervision is what
  buys out-of-distribution generalisation; the scalar GP stays as the noise-floor instrument and
  acquisition model. (2) A *tokenised transformer-core operator* over the canonical grid
  (geometry / field-map / mask patch tokens + operating-point, closure and fidelity tokens +
  arbitrary-point query decoder) becomes the declared level-(b) target once >= 100-300 map sets
  exist, and the same tokeniser is the 2-D -> 3-D transfer path (the 2-D model is the m = 0
  slice); FNO stays in the ladder as the cheaper rung, POD-GP below it. (3) *Held-out design
  FAMILIES* and a frozen "challenge" split (Arena's EMVal discipline), on top of the repository's
  held-out designs. Solver-in-the-loop verification is what Arena calls the verifier's rule and
  what this repository already enforces (F3 promotion).
* **The data reality and the user's order decide the schedule**: today there are plateaus for
  **four distinct designs at one operating point**, at 4-7 GPU-h per 33 um channel plateau and
  17-50 GPU-h per plume run; with the v2.0.6 ledger correction (`4b53012d`) the 33 um v4 plateau
  fails acceptance (b) at +2.46 % and every 50 um plateau was heating (+7-13 %). No field-operator
  network can be trained on that; a scalar GP cannot be gated on it either; and a 3-D label will
  cost ~10-80x a 2-D label (section 5, Phase 1 table). What CAN be built now is the 2-D physics
  itself (Phase 0), the 3-D design and its verification programme (Phase 1), and the dataset
  contract that both codes will write into (Phase 2), so that the first 30-100 qualified plateaus
  land directly in a training set instead of in ad-hoc result directories.

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
| **Dataset campaign generator** | `data/campaign.py`, `scripts/prepare_dataset_campaign.py`, `configs/datasets/s355_gmaw_tfillet_pilot_campaign.json`, `docs/DATASET_GENERATION_CAMPAIGN.md` | Deterministic Latin hypercube over declared axes (each with scope, bounds, transform, unit); scenario-level vs programme-level variables; replicates; snapshot schedule; hash-addressed plan; `simulation_queue.jsonl` one task per line; manifest template marked `planned_not_generated`. Pilot: 16 scenarios x 4 programmes x 2 replicates = 128 episodes, bounds explicitly non-authoritative. | This is the shape of our Phase-3 campaign generator, fed into `tools/cloud/schedule.py`. |
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
  noise floor (section 5, Phase 2) - although, after Heaviside (1.6), the maps of those same
  plateaus are ingested and fitted (POD-GP) from the first batch, not deferred.

### 1.5 `Reality-Simulator/ai`: the two competing models

Read again for this revision (`docs/`, `IMPLEMENTATION_STATUS.md`, `TODO.md`, the paper draft
and its LaTeX sections, `configs/`, and the model code; there is no ADR directory - the decision
record is `docs/ARCHITECTURE.md` "Decision date: 2026-08-29" and `docs/README.md`'s normative
order; paths relative to `Reality-Simulator/ai`). The documents contain **one system with three
levels**
(`docs/ARCHITECTURE.md` section 1: oracle, large surrogate, small real-time student - "two learned
models instead of one", `docs/ARCHITECTURE_DETAIL.md` section 12.3) and, *inside the large
surrogate*, **two competing spatial cores** that the plan keeps side by side and refuses to choose
between before data exist. The user's phrase "two competing models" is read here as these two
cores; the other pairings are noted at the end of this section.

| | **Core A - grid neural operator** | **Core B - latent-token transformer** |
| --- | --- | --- |
| lineage named in the docs | Anandkumar programme: FNO 2010.08895, TFNO (Kossaifi), PINO 2111.03794, GINO 2309.00583; `docs/FULL_PHYSICS_SURROGATE_PLAN.md` section 1 "Yes: the proposed physics-model backbone directly uses the research lineage associated with Anima Anandkumar", section 2 table | Universal Physics Transformers (Alkin et al. 2402.12365), "anchored/branched UPT-style decoders", CoDA-NO codomain attention (Rahman et al. 2403.12553), Poseidon (2405.19101); `FULL_PHYSICS_SURROGATE_PLAN.md` section 3 table ("What comes from other research lines"), `paper/latex/sections/01_introduction_related.tex` subsection "Physics foundation models and hybrid simulation" |
| what it is | dense complex spectral weights on a regular voxel grid (FNO), CP-factorised spectral quadrants (TFNO), U-Net hierarchy with spectral bottleneck (U-FNO), plus a plain 3-D U-Net as the local baseline; shape-preserving `[batch, channels, z, y, x]` operator | geometry, material and state compressed into **latent tokens**; local and global branches that "exchange latent tokens through cross-attention"; **codomain attention** across field-family tokens; **arbitrary-point query decoders** instead of full-grid decode; `FULL_PHYSICS_SURROGATE_PLAN.md` section 7 rule 9 "local/global compression rather than an unstructured full-voxel transformer", 7.1 "latent tokens for long-range coupling; arbitrary point-query decoders", 14.2 "the scalable target may use UPT latent tokens, geometry anchors, multigrid FNO blocks, graph tokens ..., arbitrary query decoders", 14.3 step 3, section 15 |
| status in the code | **implemented** behind one contract: `src/reality_ai/surrogate/{fno.py, tfno.py, ufno.py, unet.py, backbone_factory.py (BackboneFamily), configurable_model.py (ConfigurableFullPhysicsSurrogate)}`; `configs/surrogate/model_ladder.yaml` candidates `unet3d / fno3d / tfno3d / ufno3d`; `scripts/benchmark_surrogate_architectures.py` | **partially implemented, as ablations only**: `CodomainAttentionFusion3d` (`multiphysics_fusion.py`: an `nn.TransformerEncoder` over the *field-family* tokens at each voxel, chunked over positions), attention over property tokens in `MaterialFunctionEncoder` (`material_encoder.py`), a "typed token context for inspection or future attention blocks" in `heterogeneous_encoder.py` (`docs/EXPORTER_AND_HETEROGENEOUS_STATE.md`); the grid-`sample` `query_decoder.py` is the arbitrary-point decoder. **No latent-token transformer backbone exists in the code**; `IMPLEMENTATION_STATUS.md`: "Production selection or validation of CoDA-style attention; it is implemented as an ablation candidate, not assumed superior" |
| the docs' own verdict | "U-FNO is the current leading hypothesis for the fixed-grid weld model, but it must beat U-Net, FNO, and TFNO on accuracy, stability, speed, and memory before becoming the default" (`docs/SURROGATE_ARCHITECTURES.md`); "Do not begin with a colossal transformer. The FNO baseline is the cleanest test of whether the simulator data supports accurate field-level surrogate prediction" (`FULL_PHYSICS_SURROGATE_PLAN.md` 13.4) | "Add CoDA-style functional attention when channel concatenation demonstrably under-models cross-field interactions ...; add stronger latent compression when full-grid inference misses memory or latency targets" (`ARCHITECTURE_DETAIL.md` 4.1); model-size ladder "1 small local FNO baseline; 2 local/global two-branch; 3 codomain coupling; 4 latent query architecture; 5 multi-material pretraining; 6 larger foundation model only when scaling curves justify it" (`FULL_PHYSICS_SURROGATE_PLAN.md` 42.4); baseline 9 "UPT-style latent model" vs 10 "proposed local/global multiphysics model" (33.1) |
| training objective | identical for both: masked field MSE + objective MSE + Gaussian NLL + energy / mass / boundary / constitutive / phase-simplex + **sensitivity** (Jacobian vs paired interventions) (`physics/pino_losses.py`; 7.4 loss list) | same; the transformer core adds nothing to the loss - the competition is about the representation |
| data pipeline | identical for both: registry-generated channel layout, fail-closed unknown state, Zarr episodes with SHA-256 identity, whole-scenario splits, deterministic LHS campaign (`data/*`, `docs/DATA_PLANE.md`, `docs/DATASET_GENERATION_CAMPAIGN.md`) | same |
| evaluation | `docs/SURROGATE_ARCHITECTURES.md` "Required evaluation": hold constant registry, layout, splits, normalisation, horizons, optimizer-step or compute budget, parameter budget; compare field error by variable / region / horizon, interface and spectral error, conservation, free-running rollout drift, geometry and parameter holdouts, resolution transfer, calibration, latency per inference mode, peak memory; `model_ladder.yaml` `required_comparisons`; `docs/VALIDATION_PLAN.md` ablations ("fixed-grid versus geometry-conditioned encoding; simple field fusion versus CoDA-style coupling") | same list; plus the risk sentence in `paper/latex/sections/04_protocol_status_conclusion.tex`: "Geometry and cross-field attention may add cost without measurable benefit" |

**Decision criteria as documented there** (none has been exercised: `IMPLEMENTATION_STATUS.md`
"Actual training on weldsim output and resulting validated model selection" is *not implemented*;
the only datasets are synthetic):

1. Core A is the default; every Core-B mechanism must "beat a simpler controlled baseline"
   (`ARCHITECTURE_DETAIL.md` 4.1 last line) on the identical dataset / layout / splits / compute.
2. Attention across fields is admitted only on a *measured* under-modelling of cross-field
   interactions, a required transfer across physical or material systems, or a measured
   sample-efficiency gain from masked multiphysics pretraining (4.1 "Add CoDA-style functional
   attention when ...").
3. Latent compression (the UPT route) is admitted only when full-grid inference misses memory or
   latency targets, larger parts or finer resolutions are needed, or sparse queries dominate the
   workload (4.1 "Add stronger latent compression when ...").
4. Scaling to a "larger foundation model" only "when scaling curves justify it" (42.4 item 6).
5. Losing architectures are kept as reproducible baselines (`SURROGATE_ARCHITECTURES.md`
   "Advancement order" item 7).

Other pairings a reader could mean by "two competing models", for completeness: (i) the *large
surrogate vs the small real-time student* - complementary, not competing (`ARCHITECTURE.md`
sections 3 and 6); (ii) *U-FNO vs FNO/TFNO* inside Core A - a within-family ladder decided by the
same evaluation list; (iii) *data-only vs PINO-constrained*, *single vs ensemble*, *Tier-0 vs
multi-fidelity residual* - training ablations, not architectures (`VALIDATION_PLAN.md`,
`paper/ARCHITECTURE_PAPER_DRAFT.md` section 13.3 "Architecture comparisons"). The Arena Physica
pair Heaviside / Marconi
(forward / inverse) is a different kind of pair again - see 1.6.

**What this means for us.** The welding plan's rule is *operator first, attention when the data
justify it*, and it wrote that rule while expecting thousands of episodes. Our expected label count
is one to two orders of magnitude smaller (section 3), so the rule binds harder here, not looser:
the transformer core is the *target* of the ladder, and the criterion for reaching it is the same
measured one (held-out-family error at fixed compute), with the additional honesty that at 30-100
designs a transformer has no chance of beating a GP on scalars and the comparison is only
meaningful at the field level with >= 100-300 map sets (1.6.4).

### 1.6 Heaviside / RF Studio (Arena Physica)

Sources read (2026-09-05): the two company posts the user pointed at -
"Introducing Atlas RF Studio: Toward a Foundation Model for Electromagnetics", Bryant et al.,
2026-03-31, <https://www.arenaphysica.com/publications/rf-studio>, and "Introducing Heaviside-1",
Bryant et al., 2026-09-01, <https://www.arenaphysica.com/publications/heaviside-1> - plus the
public benchmark dataset card `ArenaPhysica/EMVal` on Hugging Face
(<https://huggingface.co/datasets/ArenaPhysica/EMVal>). A search for a technical report, arXiv
preprint or peer-reviewed paper on Heaviside-0/1 or Marconi-0 found none as of this date; there
is no independent replication. Everything below that is not on the dataset card is **a company
claim** and is marked so.

#### 1.6.1 What the models are (as stated)

| | Heaviside-0 (2026-03) | Heaviside-1 (2026-09) | Marconi-0 (2026-03) |
| --- | --- | --- | --- |
| role | **forward** model: geometry + materials -> S-parameters ("characterize") | forward model: 3-D geometry + materials + excitation -> complex E and H **near-fields at arbitrary probe locations** (S-parameters derived) | **inverse** model: target S-parameters + port locations -> geometry ("design") |
| architecture (stated) | "a transformer-based neural network" - *no further detail*: tokenisation of the 2.5-D layer stack (a stack of 2-D metal/dielectric images), attention layout, frequency conditioning are undisclosed | geometry encoder "overhauled" to ingest "fully 3D complex structures" with "invisible material properties like conductivity, permittivity"; **350 M parameters** ("roughly the size of GPT-2"), > 10x Heaviside-0; the backbone is not restated (continuity with the transformer is implied, not asserted) | conditional **diffusion** model over a pixelated 2-layer canvas (cites Dall-EM and pixelated inverse design); generates candidates in parallel, Heaviside ranks them, iterative refinement = "thinking time" |
| inputs | 2-layer 8 x 8 mm boards, ground vias, 3 dielectric choices; procedural variants of 25 expert templates + random "organic" structures | 3-D structures (BGA solder balls, hairpin filters shown); materials; excitation pattern (ports) | target S-parameters, ports, optional template conditioning |
| outputs | complex S-parameters at 51 frequencies 1-20 GHz, all port pairs | E, H at 10 000 probe locations x 100 frequencies x 2 ports per design; S-parameters; (far-field "in the coming months") | a geometry |
| training data (claimed) | **3 M simulated designs**, "over 20 years of combined simulation time"; S-parameter labels for all, full-wave **field labels for 10 000 designs (0.3 %)**; measured VNA data from their lab "steers the model" (amount not stated) | **250 k unique 3-D field simulations**, "500 B total field samples, > 20 TB"; the solver is not named (commercial full-wave solvers, HFSS is the example used in the text) | trained on the same simulated corpus |
| speed (claimed) | 13 ms per board, 0.3 ms batched (1024) vs ~4 min per commercial-solver run: "18 000x to 800 000x" | "10^5x faster than commercial solvers"; Atlas Fields Studio renders fields "in milliseconds" | seconds to minutes of "thinking time" |
| accuracy (claimed, vs the solver) | magnitude weighted-MAE "well under 1 dB" (their metric: sigmoid weight centred at -20 dB); RMSE (re+im) and phase MAE also reported; frontier LLMs used as the only external baseline | EMVal-SP in-distribution ~0.06 RMSE, ~0.45 dB, ~0.09 rad; **near-field**: ~19 % global relative L2, ~22 % median local error in-distribution; **~33 % / ~28 % on unseen geometry families**; median vector alignment ~98 %; "accuracy stays within 1 dB" (S-parameters) | evaluated by re-simulating its designs in the solver: better than frontier LLMs on the same metrics; no absolute yield figure |
| generalisation (claimed) | ablation: adding field labels on 0.3 % of designs cut in-distribution validation loss 15 % and "modestly" improved out-of-distribution (held-out template families) | **central claim**: a Heaviside-1 trained on fields + S-parameters equals an S-only twin in-distribution but is "substantially better" out-of-distribution (0.99 dB -> 0.53 dB in the TL;DR); OOD = *design templates* excluded from training | "alien structures" that meet spec but are outside Heaviside-0's reliable envelope |
| uncertainty / verification | **no uncertainty output is described**; the design loop keeps the solver for "production-level accuracy"; the models are called the *verifier* for the AI design loop ("Verifier's rule") | same - EMVal reports error distributions with bootstrap intervals, not per-prediction uncertainty | candidates are verified by Heaviside, final designs by the solver, fabricated designs by VNA |
| stated limits | 2.5-D planar only; S-parameters only; weaker far from the training distribution | "a limited set of geometries, board sizes, frequency ranges, material properties, and field probe distances"; OOD near-field error ~33 % | many generated candidates are "too far outside of the familiar design space for Heaviside-0 to accurately characterize" |

#### 1.6.2 What is public and checkable vs what is marketing

* **Checkable**: the EMVal v0 *public* split (dataset card, HF): 500 boards, 8 x 8 mm, three
  layers (35 um Cu signal / 200-203 um dielectric / 35 um Cu ground), two laminates (low-loss RF
  on 281 boards, FR4-class on 219), 1-2 ports, 79 boards with 2-64 plated vias, no solder balls;
  101 frequencies 1-20 GHz; 10 000-probe 3-D cloud of complex E and H per board (~41 GB) plus
  S-parameters (~7 MB); labels are **full-wave simulation outputs, not measurements**; license
  CC BY-NC 4.0 (marked "pending legal review"); metric definitions (global relative L2, median
  local error, median local alignment; RMSE / weighted MAE for S-parameters). The metric
  definitions and the public-split numbers in the Heaviside-1 post can therefore be re-scored by
  anyone with a competing model. The Atlas Fields Studio beta is public; it was not exercised for
  this note.
* **Company claims, not independently verifiable**: the training corpus sizes (3 M / 250 k
  designs, 500 B samples, "20 years of simulation"), the solver used and its settings, the
  speed-ups, the architecture beyond "transformer-based", the private standard / challenge splits
  and every number reported on them, the measured-data pipeline's contribution, and the
  "electromagnetic superintelligence" framing. No paper, no model weights, no code.
* **Method that IS verifiable in principle and is the useful part**: (i) fields as the primary
  supervision target with S-parameters as a derived downstream head; (ii) the OOD protocol -
  hold out *template families*, not just designs; (iii) forward verifier + generative proposer in
  a tight loop, with the slow solver kept as the final authority; (iv) a public, versioned
  benchmark with frozen metrics.

#### 1.6.3 Mapping to our case

| Heaviside element | maps well to our PIC surrogate | does not map / must be adapted |
| --- | --- | --- |
| geometry + materials + excitation -> steady-state fields at arbitrary points | yes: our design *is* a field map (`B_r`, `B_z`) plus masks on the grid; the operating point is the excitation; the plateau maps (`n_e`, `n_i`, `phi`, `T_e`, ionisation) and wall profiles are the fields; a query decoder gives wall-profile and sheath queries (2.3) | our fields are **time-averaged stochastic plateaus** with a measured 5-12 % particle-resolution band and shot-noise per node (`sample_count_e`), not deterministic solver outputs; every label is gate-qualified (plateau rule, residual power, Debye margin) or it is not a label |
| transformer core over a canonical representation with material / boundary conditioning | yes in form: patch tokens over the canonical `(r / r_w, z / L_ch)` grid + condition tokens (`V`, feed, injection, closure id, grid rung, code = 2-D / 3-D) + query decoder; natural 3-D extension (theta patches or azimuthal Fourier modes as tokens; the 2-D model = the m = 0 slice) | **scale**: 250 k designs vs our 30-100; a 350 M-parameter model is out of the question; a small (1-10 M) transformer with strong regularisation is only testable at >= 100-300 map sets and must beat FNO and POD-GP on held-out families at fixed compute (1.5 criteria) |
| field supervision -> generalisation | yes, and it is the strongest external argument for our level (b): train on maps, derive scalars by integration; the maps exist for every plateau already (`maps.npz`) | Arena's ablation is at 3 M / 250 k designs; at our scale the mechanism (dense supervision regularises the representation) plausibly holds but is unproven - our own ablation (scalar-only vs field-supervised at equal design count) is a Phase-3 deliverable, not an assumption |
| solver-in-the-loop verification ("verifier's rule") | already enforced: F0-F3 ladder, "every non-F3 promoted candidate is rejected pending highest-fidelity reevaluation" (`optimization/guardrails.py`), preregistered PIC batches | Arena's verifier is *the fast model* and the solver is the final check; for us the PIC is both label source and verifier because no model of ours will be trusted as a verifier before Phase 4 |
| generative proposer (Marconi) + forward verifier (Heaviside) tandem | our analogue is acquisition (qLogNEHVI / qLogNParEGO on the surrogate posterior) + surrogate objective-only mode + PIC F3; a diffusion proposer over the v1.1 geometry parameters is possible later | a diffusion model over 11 design parameters buys nothing over Sobol + BO at our scale; over field maps it would need thousands of maps; **deferred**, not adopted |
| public benchmark with frozen metrics and an OOD "challenge" split | adopt the *form*: a frozen set of PIC records (by design family) as our benchmark, band-aware metrics (4.5), a challenge family (e.g. 3-stage or 5-stage designs, or a new operating point) never used in training | our "public" is the repository's hash-bound record; there is no external community to score it |
| no per-prediction uncertainty | **not adopted**: our labels are noisy and few; heteroscedastic heads, ensembles, GP posteriors and the AR1 rung discrepancy stay (4.3) | - |
| measured (VNA) data steering the model | the analogue is *external validation* of the PIC (Brandt 2016 route, ext-val v0) - the surrogate inherits whatever the PIC's experimental standing is | no experimental thruster data of ours exists |

#### 1.6.4 What the transformer core buys over the FNO / GP ladder at our data scale - honestly

* **Scalars, N = 30-100 designs**: nothing. A GP with per-row known noise on 6-9 physical features
  is the right model and also the *instrument* that measures the noise floor (3.3); ridge and
  trees are the baselines; a transformer at this N is a random function with 10^6 parameters. This
  does not change.
* **Fields, N = 30-100 map sets** (~65 k-90 k nodes each): POD-GP on the canonical grid is the
  model; an FNO *may* fit and *may* beat POD-GP on held-out designs (the weld plan's own first
  experiment); a transformer with patch tokens at this N will over-fit unless heavily
  regularised and pretrained on cheap fidelities (P2 field -> orbit_mc wall-hit maps, 50 um
  screening runs). Test it as the last rung, report it, do not expect it to win.
* **Fields, N >= 100-300 map sets (2-D) or once 3-D records exist**: here the transformer core is
  the *declared target* for three concrete reasons. (a) *Conditioning*: tokens are the clean way
  to carry heterogeneous conditions - operating point, closure id, grid rung, code dimensionality,
  per-cusp descriptors - without tiling scalars into channels; the weld plan's `encode_static_context`
  cache is the same idea. (b) *Geometry across families*: attention does not assume a fixed
  aspect ratio or periodicity; an FNO on a canonical `(r / r_w, z / L_ch)` grid implicitly assumes
  the physics is stationary in normalised coordinates, which a 3-stage and a 5-stage channel are
  not; patch tokens with physical-coordinate embeddings do not. (c) *2-D -> 3-D transfer*: with
  azimuthal Fourier modes (or theta patches) as extra tokens, a 3-D record is a superset of a 2-D
  record and the pretrained 2-D weights initialise the m = 0 path - the transfer-learning route
  the user asked for (scratchpad 2026-09-05 01:37). None of (a)-(c) is a measured advantage yet;
  all three are hypotheses to be decided by the 1.5 criteria on our own held-out families.
* **What it never buys**: label count, band width, or the right to skip PIC verification.

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
| steady-state v2 base (`24ab82f4`) | reference (`divergent-exit-stack`, rho 0.60) | 50 um / 6e4 | plateau; classified **resolution-limited** by v4 (`0d228ad2`); ledger corrected post hoc (`3ec2af92`, v2.0.6): **heating at +13.0 %** of the electrode power | low-fidelity rung only, flagged heating |
| v2 seed-b (`41ccb1ef`), W x0.7 (`542496fb`) | reference | 50 um | plateaus; <= 1.1 % seed spread, +5.7 % `I_d` / -12 % peak `n_e` at W x0.7; corrected residuals +11.1 % / +7.2 % (heating) | the **only replicate pair** in the project -> the aleatoric band (of a heating rung: the band is quoted, its rung is not a label) |
| steady-state **v4** (`0d228ad2`) | reference | 33.3 um / 2.667e4 | plateau at 3.03 transits, residual recorded -7.7 %, **corrected +2.46 %** (`02013df0`, v2.0.6): predeclared acceptance (b) "< +2 %" **PASS -> FAIL**; (a) plateau, (c) convergence and the `resolution_limited` verdict on the 50 um base stand | 33 um label with a **disclosed (b) failure at +2.46 %** - usable as `provisional_ledger` only under a declared tolerance, or re-run under v2.0.6 |
| mini-sweep **047** (`b424ea37`) | `l1a-gs-v2-047` (rho 0.38) | 33 um / parity | plateau at 3.00 transits, `I_d` 1.925 mA, residual recorded -7.1 %, **corrected +0.9 %** (`c95919a3`): (b) pass; `assess` / `targets` deferred | qualified 33 um label (assessment pending) |
| mini-sweep **009** | `l1a-gs-v3-009` (rho 0.92) | 33 um | plateau at 3.02 transits 23:59 AEST 09-04; **record commit pending** | qualified when recorded |
| mini-sweep **reference** | reference | 33 um | past 3 transits ~00:10 AEST 09-05 (numerical replication of v4 on the H100); record pending | replicate of v4 across GPUs (numerical, not bitwise) |
| mini-sweep **056** launch 2 (`ee35bc84`) | `l1a-gs-v3-056` (rho 2.36, HEMP-like) | 33 um | running, ETA 06:00-08:00 AEST 09-05 (launch 1 gate-stopped on a shot-noise artefact, `ccee5c60`, no plateau) | pending |
| mini-sweep 106, 056 seed replicate | sealed, not launched | 33 um | - | - |
| steady-state **v5** (`69ff435d`, launch 2 `ce1d96cb`) | reference | 25 um / 1.5e4 | running on the H100; verdict ~10-11 AEST 09-05 | decides whether 33 um carries a grid band |
| plume attempts 6 / 7 / 8 (`9cf7ca39`, `24ea2f65`, `ac248e05`) | reference, plume box | 50 um | ignited; **no plateau** (gate stop, budget stop, finite-grid heating) | feasibility-class labels only ("heats at 50 um plume") |
| external validation v0 (`036bd679`) | Brandt 2016 geometry | 20 um | stopped on the residual-power gate; genuine heating; **inconclusive** | not a label |

Count: **four distinct designs with a qualified or imminent 33 um channel plateau (reference,
047, 009, 056), all at ONE operating point (300 V, `n_g0` 5.5e19, 3 mA @ 2 eV), one closure (v1.3,
no recycling); one replicate pair at 50 um (a heating rung under v2.0.6); zero plume plateaus;
zero experimentally validated points; and the reference's 33 um plateau itself fails its
predeclared residual acceptance by 0.46 percentage points under the corrected ledger.** That is
the whole training set. A 7-D surrogate needs on the order of 10 points per
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
   `inelastic_loss_j` macro-weight bug biased every recorded residual negative by the inelastic
   power (7-14 % of electrode work). The fix is **model v2.0.6** (`4b53012d`; spec entries in
   `8c70cff0`) with post-hoc `ledger-corrected.json` sidecars for 13 runs: v2 base +13.0 %,
   seed-b +11.1 %, W x0.7 +7.2 % (all heating), ss-v4 **+2.46 % -> acceptance (b) FAIL**, 047
   +0.9 %, 056 L1 +0.6 %, plume attempts 3-8 +11 to +67 %, ext-val v0 +61.7 %; thresholds kept at
   5 % hard / 2 % acceptance. Records executed on pre-v2.0.6 code (009, reference replication,
   056 L2, v5) carry `provisional_ledger` until their sidecar is written; the dataset stores both
   the recorded and the corrected residual and the label status names which one qualified it.
   Gate-stopped, heating and no-plateau runs are **not** regression labels; they are
   *classification* labels (section 4.3).
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
| FNO backbone + heteroscedastic heads (Core A, 1.5) | **adapt to 2-D**, use only at Phase 3 | 2-D spectral convolution on `[batch, channels, r, z]`; the 3-D code is a direct template; the middle rung of the field ladder (POD-GP -> FNO -> token transformer) |
| latent-token / attention core (Core B, 1.5; Heaviside, 1.6) | **adopt as the declared level-(b) target**, testable only at >= 100-300 map sets | patch tokens over the canonical grid (geometry / field / mask channels) + condition tokens (operating point, closure id, rung, code dimensionality, per-cusp descriptors) + arbitrary-point query decoder (the weld `query_decoder.py` idea) + heteroscedastic heads; 1-10 M parameters, ensemble of 3-5; azimuthal modes / theta patches as extra tokens for 3-D records (the 2-D model is the m = 0 slice); admitted only by the 1.5 criteria on held-out families |
| U-Net / TFNO / U-FNO ladder | defer | only if the FNO fails at the sheaths and > 100 maps exist; the transformer rung supersedes U-FNO as the "sharp interfaces + global coupling" candidate |
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

**Level (b), fields - fitted from the FIRST batch (POD-GP), networks at N >= 100-300 map sets;
the transformer core is the declared target.** (Revised after 1.6: fields are the primary
supervision target wherever maps exist; the scalar heads are integrals of decoded fields, checked
against the level-(a) GP.)

* Baseline first, and from the first batch: `surrogates.PODFieldSurrogate` on the canonical grid
  (POD over log-density / potential / log-ionisation maps; GPs on the modal coefficients; mesh
  hash bound) - it is the right model for tens of snapshots and it gives pointwise variance.
* Middle rung: 2-D FNO (adapted from `Reality-Simulator/ai/src/reality_ai/surrogate/fno.py`)
  with inputs `[B_r, B_z, |B|, masks, r / r_w, z / L_ch]` + tiled operating-point context, outputs
  `[log n_e, log n_i, phi / V, T_e, log1p S_map]` each with a log-variance head; masked Gaussian
  NLL where the *target variance floor* per node is the shot-noise variance from `sample_count_e`
  (band-aware by construction); physics terms as in 4.1; ensemble of 3-5 (different seeds and
  member-wise bootstrap of designs).
* Target rung, the **tokenised transformer-core operator** (1.6.4): the canonical grid cut into
  patches (e.g. 8 x 8 nodes on a 128 x 1024 grid = 2048 patch tokens; coarser patches first),
  each patch token = linear embedding of the static channels + a physical-coordinate embedding
  `(r, z, r / r_w, z / L_ch)`; condition tokens = operating point, closure id, grid rung, code
  dimensionality (2-D / 3-D), per-cusp descriptors (rho, wall `|B|`, plane position), N stages;
  an encoder of 4-8 pre-norm blocks (1-10 M parameters); outputs by (i) patch de-embedding to the
  grid for full-field mode and (ii) an arbitrary-point query decoder (cross-attention from a
  coordinate query to the latent tokens) for wall profiles, sheath drops, cusp leak widths and
  the sparse-query mode; heteroscedastic heads and the same masked NLL / physics terms as the
  FNO. Regularisation and pretraining are mandatory at our N: dropout on tokens, weight decay,
  masked-patch pretraining on the *cheap* fidelities (P2 field maps for hundreds of catalogued
  designs; orbit_mc wall-hit maps; 50 um screening maps) before fine-tuning on qualified
  plateaus. For 3-D records (Phase 1 onward) the azimuthal Fourier modes `m = 0, 1, ..., M` of each
  field become additional tokens; a 2-D record is exactly the `m = 0` subset, so 2-D pretrained
  weights initialise the 3-D model.
* Selection: the three rungs are compared on the identical dataset / splits / normalisation /
  compute (the `Reality-Simulator/ai` "Required evaluation" list, 1.5) on **held-out design
  families** (4.5); a rung is adopted only if it beats the rung below on the band-normalised
  field error of the held-out family, keeps Gauss-law and integral consistency inside bounds and
  is calibrated. U-FNO is dropped from the ladder; the transformer rung is the "sharp interfaces +
  global coupling" candidate (sheaths are 1-3 cells wide).
* Scalar consistency: the integrals of the decoded fields (S, N_e, wall currents, `I_beam`) must
  agree with the level-(a) GP and the recorded scalars within the band; disagreement is a
  diagnostic the dashboard shows and a reason to distrust the field model, not the GP.

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

1. **Preregistered held-out designs AND a held-out design family**: before any fit, a maximin
   subset of >= 20 % of the labelled designs (whole designs, all rungs / seeds / frames / codes)
   is sealed in the protocol as the assessment role; a second small extrapolation role outside
   the training hull is *reported, not gated* (the surrogate v1 / v2 partition design, inherited
   by hash). Added after 1.6 (Arena's EMVal "standard" vs "challenge" splits, where "challenge"
   excludes whole design *templates*): a **challenge role** = one whole design family never seen
   in training (e.g. the 3-stage or 5-stage family, or one operating point), whose error is
   reported as the out-of-distribution number every version must quote; the field-supervised vs
   scalar-only ablation (1.6.3) is evaluated on this role. The frozen record set with its metric
   definitions is the repository's own benchmark, versioned like EMVal.
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
   `noise_floor_only` (Phase 2's expected verdict), each with what the next version needs, as in
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

**Order fixed by the user (2026-09-05): (1) finish the 2-D PIC physics, (2) design the 3-D PIC
and verify that it works, (3) then the AI run.** The phases below follow that order. Phase 0 and
Phase 1 are PIC work with no surrogate content; they are in this document because they decide
which simulator generates the labels, what the labels' bands and caveats are, and what the
tokeniser of 4.2 must carry. Every cost is a projection anchored on a recorded number; the
anchor is named in each row.

### Phase 0 - finish the 2-D physics (physics completeness audit -> implementations -> qualified fast solver -> ladder verdicts)

The content is the physics completeness audit `modern/docs/pic2d-physics-completeness-audit.md`
(`0901138a`, 2026-09-05; 11 graded gaps, ranked roadmap R0-R6, 151 resolved references) and the
acceleration review `modern/docs/literature/pic-acceleration-methods.md`. This plan does not restate
them; it fixes the order and the exit criteria the later phases depend on.

1. **Audit (done, `0901138a`)**: "physics first" gaps R1 anomalous transport (absent by construction
   in (r,z); every HEMPT PIC imposes a Bohm closure), R2 SEE from the dielectric at the cusps
   (cusp sheath drops -10 to -45 %), R3 the full e-Xe set (four excitation levels) + Xe+/Xe CEX and
   MEX with a fast-neutral thrust tally, R4 Coulomb collisions; second wave R5 spatial neutrals +
   metastable pool, R6 diagnostics (Xe2+, neutraliser gas, beta map, sputter yield). Section 6 of
   the audit states what the 2-D model can never claim (a self-consistent anomalous mobility,
   azimuthal structure, instability heating) - that list is the specification of what Phase 1
   must answer.
2. **Implementations, in the audit's rank order**, each as a declared closure change with its own
   `pic2d-model-v2.x` spec entry, on/off comparison against the accepted reference plateau
   (`0d228ad2`) at the same seed / grid / W / gates, and the expected sign of every change written
   down before the run (audit section 5). Prerequisite R0 (the v2.0.6 ledger, `4b53012d`) has
   landed. The Bohm alpha closure (R1) is the one that makes every 2-D label *conditional on a
   declared alpha*; the label ledger of Phase 2 carries alpha as part of the closure id.
3. **Qualified fast solver**: v2.0.5 measured solo; GMG Poisson (`poisson_gmg_v1`, `9c2e4222`) solo
   probe and its v4 replay campaign; launch fusion + cell sort (shared with the Coulomb kernel);
   mixed precision; an explicit energy-conserving gather trial only against the explicit 33 / 25 um
   ladder (the review's protocol: bitwise where claimed, +-5 % band otherwise). A re-priced cost
   model from measured solo ms/step. Nothing in the surrogate plan depends on the speed-ups; the
   3-D cost table below is quoted at today's rate and again at the review's 2-3.5x.
4. **Ladder verdicts**: v5 (25 um, `69ff435d`, running) decides whether 33 um carries a grid band;
   every closure change (R1-R5) re-runs the reference at 33 um and, for the final combined closure,
   the 33 / 25 um pair, so the closure that generates labels has its own `converged /
   resolution_limited` verdict. The mini-sweep closure (009 / reference / 056 records, sweep-wide
   `assess`, `targets` for all four) and the like-for-like external validation
   (`channel-20um-bohm-0.4-see`, the audit's single most valuable run) belong here.
5. **Record contract freeze**: schema versions of `summary.json`, `assessment.json`, `maps.npz`
   keys, the closure-target JSON, `ledger-corrected.json`; any later change bumps the version and
   the Phase-2 ingestion refuses unknown keys. The contract must already reserve the fields a 3-D
   record will add (section Phase 1, item 6).
6. **STOP-file / SIGTERM handler** in the shared runner; occupancy floors in accumulated
   particle-steps on every density gate (v2.0.4 / v2.0.6, audit them all at once); plume
   qualification (a plateau at an admissible operating point / grid; under v2.0.6 every 50 um plume
   attempt read >= +11 % of the electrode power in its first complete window and +17 to +67 % at
   its end, `37665d70` - the 50 um plume grid with the flux-tube cathode was never conservative).

Cost (anchors: audit section 5; ss-v4 5.0 h on the 5090; H100 ~1x per process, MPS-4 1.54x
aggregate): audit roadmap runs R1-R6 ~15 runs, **75-90 GPU-h**; ladder re-runs for the final
closure (33 + 25 um on the reference, 25 um at 15-35 GPU-h) and two more design points
**60-120 GPU-h**; ext-val like-for-like 20 um **12-30 GPU-h**; plume qualification attempt at
v2.1 / 50 um or 33 um **20-50 GPU-h**. **Total ~170-290 GPU-h, USD 0.5-0.9k, 4-8 weeks wall** -
dominated by developer time (the audit's 20-30 developer days for R0-R5), not GPU.

Exit criteria: the R1-R4 closures implemented, each with an on/off record whose signs match or
whose mismatch is written up as a finding; a 33 um reference plateau under the combined closure
with acceptance (b) passing on the *corrected* ledger; its 25 um rung verdict; the ext-val
comparison recorded (agreement or a recorded miss); the record contract frozen at a version. Only
then is "a 2-D label" defined.

### Phase 1 - 3-D PIC: design + verification (placeholder; the design doc is forthcoming)

The 3-D design itself is a separate document (`modern/docs/pic3d-design.md`, not yet written).
This section fixes **what "verify it works" must mean** before any 3-D result is used for
anything, the resolution / cost reality on our grids, and the decomposition the design must
support. The audit's section 6 (iii) is the seed of the cost estimate; the numbers below re-derive
it from the v4 record.

**Geometry fact that shapes the design**: our channel is axisymmetric (bore radius 2 mm to z =
18 mm, cone to 3 mm at z = 24 mm; masks from the P2 mesh regions). The 3-D mesh is therefore the
2-D mask x theta; the masks, the dielectric, the anode and the far plane do not depend on theta.
Consequently the Poisson operator separates in theta: an FFT in theta turns the 3-D solve into
`N_theta / 2 + 1` independent 2-D `(r, z)` solves of `(1/r d/dr r d/dr + d^2/dz^2 - m^2 / r^2) phi_m
= -rho_m / eps_0` with the *existing* masks and boundary rows; the `m = 0` problem *is* today's
solver, and every other `m` differs only by a positive diagonal term (better conditioned). The
block-Thomas column factorisation and the GMG both extend per mode; the RUB code in the LANDMARK
r-theta benchmark used exactly "FFT in the azimuthal direction and a tridiagonal solver in the
radial direction for each of the azimuthal harmonics" (Villafana et al. 2021, section 3). The
particle side is the 3-D Boris push our `orbit_mc` already performs (3-D positions and velocities
in the axisymmetric field), a theta coordinate on every particle (today: `v_theta` without
theta), 8-node CIC in `(r, theta, z)` with the Verboncoeur axis volumes, and a periodic theta
(full annulus or a `2 pi / n` sector).

**What "verify it works" must mean** (every item preregistered with a pass band before the run):

1. *Manufactured solutions.* 3-D Poisson on the masked cylinder with `phi = J_m(k r) cos(m theta)
   sin(k_z z)` for several `m`, second-order convergence per mode and exact reproduction of the
   2-D solver at `m = 0` (bitwise: the same factorisation). Boris push against `orbit_mc` orbits in
   the P2 field (the 1e-10 energy gate the orbit code already passes), including orbits with
   `v_theta != 0` that cross sector boundaries. Charge conservation of the 8-node deposit to
   round-off; Gauss law with the surface charge on the theta-extruded wall nodes.
2. *Axisymmetric-limit replay.* Start from the theta-uniform 2-D plateau checkpoint (reference,
   33 um), `N_theta` small (8-16, azimuthal cell 0.8-1.6 mm >> any unstable wavelength), run
   0.5-1 us. Pass band: theta-averaged `I_d`, `S`, `n_g`, peak `n_e`, `T_e,peak` inside the 2-D
   particle-resolution band (5.7 / 4.6 / 4.0 / 11.9 / 9.3 %); the `m != 0` field energy stays at
   the particle-noise floor (no numerical azimuthal instability of the scheme). Cost 5-10 GPU-h.
3. *LANDMARK benchmarks in the code's slab limit* (both are in the repository's literature review
   with DOIs; parameters verified against the papers 2026-09-05). (a) **Axial-azimuthal**, Charoy
   et al. 2019 (doi:10.1088/1361-6595/ab46c5): 2.5 cm x 1.28 cm, 500 x 256 cells of 50 um,
   `dt` 5 ps, 4e6 steps = 20 us, 200 V, prescribed cosine ionisation source `S_0 = 5.23e23
   m^-3 s^-1`, collisionless, 75 / 150 / 300 ppc; seven codes agreed within ~5 % on the
   time-averaged profiles; their computing times were 2.5-21 days on 32-448 CPU cores or 9-14
   days on one or two V100s (2019 hardware). (b) **Radial-azimuthal**, Villafana et al. 2021
   (doi:10.1088/1361-6595/ac0a4a): 1.28 x 1.28 cm, 256 x 256 cells of 50 um, `dt` 15 ps, 30 us,
   `B_z` 200 G, virtual axial `E_x` 10 kV/m, grounded walls, 100 ppc initial (~212 at steady
   state), collisionless, ECDI + MTSI; seven codes within 5 %; 12-306 h on CPU codes, hours on
   the GPU code. (Note for the brief: the *r-theta* case is Villafana 2021; Charoy 2019 is
   *z-theta*.) Pass band: time-averaged profiles inside the published seven-code envelope; the
   dominant azimuthal wavelength and frequency inside the reported spread; the ppc convergence
   trend reproduced. Our projected cost at ~1e9 particle-steps per second per H100 (the audit's
   measured anchor, consistent with the v4 record: 4.5 M particles at 3.5 ms/step): (a) 20-80 M
   particles x 4e6 steps = 1-3 GPU-days per case, (b) 14-28 M x 2e6 = 8-16 GPU-h per case;
   **~100-150 GPU-h for both with one ppc-convergence pair each**.
4. *Sector convergence on our own channel.* One design (reference), 90 deg -> 180 deg -> 360 deg
   at fixed `r dtheta`, same seed family; pass band: theta-averaged plateau scalars inside the
   particle band, the `m`-spectrum of the fluctuations converged in its resolved range. This is
   the check that a sector is admissible for labels.
5. *Measured scaling.* ms/step vs `N_theta` and vs GPU count on the real reference field before any
   preregistered 3-D run (the L1b v1 and mini-sweep lessons: preflight at production load, budget
   at 1.5x).
6. *Record contract for 3-D.* The 2-D contract plus: `theta_cells`, `sector_fraction`,
   theta-averaged maps (the same keys as `maps.npz`, so a 3-D record's `m = 0` slice ingests as a
   2-D record does), the azimuthal mode spectra of `n_e`, `phi`, `E_theta` per `(r, z)` band, the
   `m != 0` energy fraction, the instability-driven cross-field current (the Reynolds-stress /
   `<n E_theta>` term) as a map - this last quantity is the effective anomalous mobility the 2-D
   alpha closure will be calibrated against.

**Resolution / cost table for our grids** (reference channel, `dr = dz` 33.3 um, 90 x 720 =
64 800 cells, ~4.5 M macro-particles at parity in 2-D, `dt` 1.4 ps; anchors: v4 record
`0d228ad2` - 5 142 858 steps = 3 transits, 1 714 286 steps = 1 transit; 3.5 ms/step; ~1e9
particle-steps/s/GPU; 64 B per particle). "Full" = 360 deg; the azimuthal cell `r dtheta` is
what the Debye gate sees, and it is smallest at the peak-density radius `r ~ 0.7 mm` (window
peak `lambda_D` 15.5 um at 1.29e18 m^-3 / 5.6 eV) and largest at the lip `r = 3 mm`.

| `N_theta` (full annulus) | `r dtheta` at r = 0.7 / 2 / 3 mm | `r dtheta / lambda_D` at the peak (hard gate pi, soft 2.5) | cells | particles (parity) | memory | ms/step, 1 GPU (projected) | GPU-h: warm 1 transit / cold 3 transits | wall on 8 H100 (ideal decomposition) | USD (3/GPU-h) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 64 | 69 / 196 / 295 um | **4.4 - inadmissible** | 4.1 M | 290 M | 18 GB | ~220 | 105 / 315 | 13 / 39 h | 315 / 945 |
| 128 | 34 / 98 / 147 um | 2.2 (soft margin ok) | 8.3 M | 580 M | 37 GB | ~440 | 210 / 630 | 26 / 79 h | 630 / 1,890 |
| 256 | 17 / 49 / 74 um | 1.1 | 16.6 M | 1.15 G | 74 GB (does not fit one 80 GB card with sort buffers) | ~870 | 415 / 1,240 | 52 / 155 h | 1,245 / 3,720 |
| **90 deg sector, 64 theta cells** (= the 256 row's `r dtheta`) | 17 / 49 / 74 um | 1.1 | 4.1 M | 290 M | 18 GB | ~220 | **105 / 315** | **13 / 39 h** | **315 / 945** |
| 45 deg sector, 64 theta cells (= a 512 row's `r dtheta`) | 8.6 / 25 / 37 um | 0.55 | 4.1 M | 290 M | 18 GB | ~220 | 105 / 315 | 13 / 39 h | 315 / 945 |

Reading: a full annulus at `N_theta = 64` is ruled out by the same finite-grid-heating gate that
ruled out 50 um in 2-D; `N_theta = 128` is the coarsest admissible full annulus and `256`
the comfortable one; a **90 deg periodic sector with 64 theta cells has the resolution of the 256
row at a quarter of its cost**, and it is the unit this plan costs a "3-D anchor" at: **~100-300
GPU-h, USD 300-950, 0.5-1.6 days wall on the 8-GPU box** - i.e. **10-80x a 2-D label** (4-11 GPU-h). The
sector is only admissible if item 4 above passes; the unstable azimuthal wavelengths at our
`B` (0.1-0.5 T, ten times the Hall-thruster benchmarks') are expected to be tens of um, i.e.
`lambda_D`-scale, so a 4.7 mm arc (90 deg at the lip) holds many wavelengths - a hypothesis the
sector-convergence run tests, not a result. "Warm 1 transit" assumes the run starts from the
theta-uniform 2-D plateau and only the azimuthal modes must saturate; whether the theta-averaged
plateau rule is then re-satisfied within one transit is itself a predeclared outcome. The audit's
independent estimate (18 M cells, ~550 M electrons, ~23 days on one H100 for 5 us at full annulus;
a 30 deg wedge 8x cheaper) agrees with the 256 row within a factor 2. The acceleration review's
2-3.5x would scale every GPU-h figure down by the same factor once *measured* on the 3-D code.

**Multi-GPU decomposition needs** (design requirements, not decisions): theta-slab decomposition
of particles (each GPU owns a contiguous theta range; migration per step is a small fraction - the
azimuthal drift `E / B ~ 1e4-1e6 m/s` moves an electron 0.01-1 um per step against 17-74 um cells)
with ghost-node deposit exchange; the Poisson FFT in theta needs an all-to-all transpose of the
charge array (65 611 nodes x `N_theta` x 8 B = 34-134 MB per step; sub-millisecond over NVLink on
the H100 SXM box, tens of ms over PCIe / MPS-shared cards - the box choice matters); mode-parallel
2-D solves (each GPU owns a range of `m`); the diagnostics that today run inside one CUDA graph
(window accumulators, ledgers, peak-Debye statistics) need a reduction across GPUs at the record
cadence only. The single-GPU path must remain the reference (bitwise replay of the 2-D code at
`m = 0` is the regression test). Memory: the 128 row fits one card only just; 256 needs two.

Cost of Phase 1 (GPU): manufactured + axisymmetric-limit checks 10-20 GPU-h; LANDMARK pair with
ppc convergence 100-150 GPU-h; scaling measurements 20-50 GPU-h; sector-convergence triplet on the
reference (90 / 180 / 360 deg, warm start) 100 + 200 + 400 = ~700 GPU-h if all three are run at 64
theta cells per 90 deg (or ~350 if the 360 deg leg is skipped after 90 / 180 agree); one first
3-D reference anchor at the chosen sector 100-300 GPU-h. **Total ~600-1,200 GPU-h, USD 1.8-3.6k,
2-4 weeks wall on the 8-GPU box after the code exists** - the code (design doc, kernels,
decomposition, tests) is weeks of developer time on top and is the real cost.

Exit criteria: items 1-5 recorded with their pass bands met (or a recorded miss); a 3-D record
contract; **one 3-D anchor of the reference design** with its theta-averaged plateau, its
`m`-spectrum and its effective cross-field current map; and the number that decides Phase 3:
**the 3-D-vs-2-D shift of `I_d`, `S`, utilisation, peak `n_e` on the reference, in units of the
2-D particle band.**

### Phase 2 - dataset schema, ingestion, noise floor (from 2-D and 3-D records)

* Build `cft_revival/pic_dataset` (or `surrogates/pic_records.py`): the record registry (semantic
  ids for every `summary.json` scalar, `maps.npz` array, closure target, ledger sidecar, 3-D
  spectrum; unit, support, disposition), the ingestion, the label ledger (one row per record x
  quantity: value, band, rung, **closure id incl. alpha**, seed, W, **code = 2-D / 3-D, sector**,
  verdict, `label_status` in {qualified, provisional_ledger, classifier_only}), the dataset
  manifest with lineage hashes, the canonical-grid resampler with its error record, split
  assignment by design id **and design family** (4.5). Tests: fail-closed on a tampered sidecar,
  an unknown key, a mixed rung, a mixed closure, a design straddling splits, a 3-D record whose
  `m = 0` slice does not ingest as a 2-D record.
* Noise-floor study on what exists after Phase 0: seed / W / cross-GPU replication pairs
  (reference), the 056 seed replicate, the alpha-series (a *closure* band, reported separately
  from the particle band); per-quantity band table in log space; split-half reliability where two
  replicates exist. The 3-D anchor's theta-averaged scalars enter as a **fidelity**, not as noise.
* Scalar baseline: GP + ridge + trees + mean on the labelled designs, **`noise_floor_only` /
  plumbing evidence**; leave-one-design-out error against the band. **POD-GP on the maps of the
  same designs from this phase on** (1.6 change 1): pointwise band-normalised error and the
  scalar-from-field consistency check, also `noise_floor_only`.
* Cost: CPU only, 1-2 weeks of work; 0 GPU-h.
* Exit: the manifest reproduces byte-for-byte from the records; the floor table is recorded; the
  campaign generator knows which quantities need replicates; the challenge family is chosen and
  sealed.

### Phase 3 - AI campaign (labels, then models)

**Which simulator generates the labels** is decided by Phase 1's exit number, not in advance.
Three options, with the decision rule and the cost of each:

| option | labels | when it is the right choice | cost for target A (24-30 designs) / target B (70-100) of 3.5 |
| --- | --- | --- | --- |
| **A. 2-D labels, calibrated closure, 3-D anchors as the caveat** | 2-D 33 um plateaus under the Phase-0 combined closure with alpha *calibrated on the 3-D anchors* (a constant, or an `(r, z)` map from the anchors' effective cross-field current); 3-5 3-D anchors across rho (reference + 047 + 056 + one operating point) | the 3-D-vs-2-D shift on the reference is within the 2-D particle band **or** a one-parameter / one-map alpha closure reproduces the anchors' theta-averaged `I_d`, `S`, peak `n_e` within the band on a held-out anchor | 2-D 120-250 / 400-800 GPU-h + anchors 3-5 x 100-300 = 300-1,500 -> **A: 420-1,750 GPU-h (USD 1.3-5.3k); B: 700-2,300 GPU-h (USD 2.1-6.9k)** |
| **B. 3-D labels at scaled resolution for every design** | 90 deg sector, 64 theta cells, warm-started from a 2-D plateau; 1 transit | the shift exceeds the band **and** no closure reproduces it **and** the budget allows | 24-30 x 100-300 = **2,400-9,000 GPU-h (USD 7-27k)** for A; 70-100 x 100-300 = **7,000-30,000 GPU-h (USD 21-90k)** for B - **not affordable at today's rate**; only if the 3-D step measures >= 5x faster than projected |
| **C. multi-fidelity: 2-D low, 3-D high** | 70-100 2-D labels (low fidelity) + 8-12 3-D labels (high fidelity) in the AR1 / multi-task residual (`TwoFidelityAR1`, BoTorch `MultiTaskGP`, the weld `MultiFidelitySurrogate` idea); the surrogate's *target* fidelity is 3-D | the shift exceeds the band and the closure does not reproduce it, but the shift is *smooth in the design features* (the residual is learnable from ~10 points) | 400-800 + 8-12 x 100-300 = **1,200-4,400 GPU-h (USD 3.6-13k)**; the field-level model is then a 2-D model plus a scalar 3-D correction, and the transformer core's 3-D tokens are trained on 8-12 records - i.e. not trained |

Decision rule, in order: (1) run the reference anchor (Phase 1 exit) and two more anchors (047
low-rho, 056 HEMP-like: 200-600 GPU-h); (2) compute the shifts in band units; (3) fit the alpha
closure on two anchors, test on the third; (4) choose A if the test passes, C if it fails and the
residual is smooth, and declare B unaffordable unless the measured 3-D rate says otherwise. The
choice, the three anchors and the test are one preregistered experiment. Whatever the choice, the
**label is what the chosen simulator says under its declared closure at its declared rung and
sector**, and the surrogate is "of that model" - never of the thruster.

Then the campaign as before, with the label source fixed:

* Sampler: scrambled Sobol on the declared box (4-cusp family first; then N stages, `V`, feed),
  the whole-set feasibility screen of 3.4 before the prereg commit, boundary challenges, replicate
  allocation from Phase 2's floor; the challenge family excluded from every batch until the final
  evaluation.
* Per batch (8-16 designs + 2-4 replicates): one preregistration (protocol binding the sealed
  per-design protocols, preflight, shakedown on ONE real design through run -> finalize -> assess
  -> targets, MPS replay), launched through `tools/cloud/schedule.py` at MPS-4 per GPU; results
  committed from the job worktrees; the Phase-2 ingestion runs on each batch as it lands. Never
  kill a client under MPS; budgets at 1.5x the measured early rate.
* Models, in ladder order and gated as 4.5: the scalar GP (noise floor + acquisition) from batch 1;
  POD-GP on the maps from batch 1; the 2-D FNO and the tokenised transformer-core operator only
  once >= 100 map sets exist (option A / C target B), pretrained on the cheap fidelities (P2 field
  maps of the 224 catalogued designs, orbit_mc wall-hit maps, 50 um screening maps) and compared
  at fixed compute on the held-out family; the field-supervised vs scalar-only ablation (1.6.3)
  reported on the challenge family. Training on the local 5090 or one H100-hour; never beside a
  preregistered run.
* After batch 1 (16 + 4 runs, 80-160 GPU-h, USD 250-500): the first *gated* scalar attempt and
  the first POD-GP field verdict (target A). After batches 2-4 (cumulative 70-100 labels): target
  B, the network rungs, the ablation.
* Cost: option A (recommended if its test passes): **700-2,300 GPU-h, USD 2.1-6.9k** through
  target B, plus 5-20 GPU-h of training; option C: 1,200-4,400 GPU-h.
* Exit: the acquisition loop runs from the label ledger with the stopping policy's diagnostics
  visible; each rung has a verdict under the 4.5 gates on the held-out designs *and* the challenge
  family; the field-supervision ablation is recorded.

### Phase 4 - optimisation with solver-in-the-loop verification

* The campaign of 4.6 on the accepted F2 source: a constrained multi-objective acquisition
  proposes 8-12 candidates per round; each round is a preregistered PIC batch (F3, the label
  simulator of Phase 3); disagreement beyond the band feeds the next round; two rounds minimum
  before any front is drawn. **The final shortlist (2-3 designs) is verified in 3-D** at the
  Phase-1 sector regardless of which option Phase 3 chose - that is the solver-in-the-loop rule
  applied to the highest fidelity we have, and it is where an option-A closure is caught if it
  drifted off the anchors.
* Plume objectives only if Phase 0's plume qualification succeeded and target D of 3.5 has been
  paid for; otherwise the channel-quantity objective set, declared as such.
* Cost: 2-3 rounds x 10 designs x 4-11 GPU-h = 100-300 GPU-h (USD 300-900) for 2-D channel
  confirmation; 3-D shortlist verification 2-3 x 100-300 = **200-900 GPU-h (USD 0.6-2.7k)**; plume
  confirmation 20-50 GPU-h per point on top.
* Exit: an F3-verified Pareto set with bands (and its 3-D verification of the shortlist), the
  surrogate-vs-PIC disagreement table, and the claim envelope; the paper admission would be a
  `numerical-campaign` gate with the surrogate as a *tool* of the campaign, not as a finding.

### Cost summary

| phase | GPU-h | USD (3/GPU-h) | wall (8 H100 box, USD 24/h) | dominant cost |
| --- | --- | --- | --- | --- |
| 0 - 2-D physics | 170-290 | 0.5-0.9k | 4-8 weeks | developer time (audit R0-R5: 20-30 days) |
| 1 - 3-D design + verification | 600-1,200 | 1.8-3.6k | 2-4 weeks GPU after the code exists | developer time (design doc, kernels, decomposition) |
| 2 - data plane + noise floor | 0 | 0 | 1-2 weeks CPU | - |
| 3 - AI campaign, option A (2-D labels + 3-5 3-D anchors) through target B | 700-2,300 | 2.1-6.9k | 2-4 weeks | GPU |
| 3 - option C instead | 1,200-4,400 | 3.6-13k | 4-8 weeks | GPU |
| 3 - option B (all 3-D) | 7,000-30,000 | 21-90k | months | **unaffordable at today's rate** |
| 4 - optimisation + 2-D F3 + 3-D shortlist | 300-1,200 | 0.9-3.6k | 1-2 weeks | GPU |
| 4 (+ plume) | +400-1,200 | +1.2-3.6k | +1-2 weeks | GPU |
| **first useful loop (0 -> 4, option A, channel)** | **~1,800-5,000** | **~5.4-15k** | **~3-5 months** | developer time in 0-1, GPU in 3-4 |

Against the previous revision (400-1,100 GPU-h, 2-4 weeks): the increase is the user's order -
the 2-D physics has to be finished and the 3-D code built and verified *before* a single label is
paid for - plus the 3-D anchors that make a 2-D label defensible. The 2-D-label part alone
(Phase 3's 2-D campaign 400-800 GPU-h + Phase 4's 2-D confirmation 100-300 GPU-h = 500-1,100
GPU-h) is unchanged from the previous revision; everything else is the price of the 3-D anchors
and of verifying the shortlist at the highest fidelity we will have.

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
| the 3-D-vs-2-D shift is large and not reproducible by any alpha closure (option C or B forced) | audit section 6: 2-D reduced models "significantly overestimate" the mobility; no 3-D PIC of a cusped-field thruster exists to bound it | the three-anchor preregistered test decides before any 2-D campaign is paid for; option C keeps the 3-D count at 8-12; option B is declared unaffordable rather than attempted piecemeal |
| the 3-D sector is inadmissible (m-spectrum not converged at 90 deg) | hypothesis only; the unstable wavelengths at 0.1-0.5 T are estimated, not measured | Phase 1 item 4 (90 / 180 / 360 deg) before any 3-D anchor is used as a label or a calibration point |
| the 3-D step is slower than the ~1e9 particle-steps/s projection (8-node CIC, theta sort, transposes) | 2-D perf audit: the step is latency-bound; MPS inflates tiny kernels 7-10x | Phase 1 item 5 measures solo before any prereg; budgets at 1.5x; the cost table is re-issued from the measurement |
| the transformer rung over-fits or is selected by taste | Reality-Simulator/ai's own risk sentence; our v1/v2 selection instability | the 1.5 criteria verbatim: same data / splits / compute, held-out family, beat the rung below or it is not adopted; POD-GP remains the model otherwise |
| Arena's field-supervision result does not transfer to tens of designs | their ablation is at 250 k designs | our own ablation (field-supervised vs scalar-only at equal design count) is a Phase-3 deliverable on the challenge family; fields-first is a design choice, not a claimed gain |

### What must NOT be claimed (at any phase)

* Thruster performance - thrust, `Isp`, efficiency, discharge power - from any model trained on
  channel-only records; nor from plume records until a plume plateau under an accepted residual
  exists. Any "development" thrust number with > 5 % residual power is non-quotable.
* "Converged" for any grid rung before the ladder says so; the surrogate inherits the rung's
  verdict and says "at 33 um, <verdict>".
* Experimental validity: the PIC is not externally validated (ext-val v0 inconclusive); the
  surrogate is a surrogate of *this* model under *this* closure (electrostatic, axisymmetric, no
  anomalous transport by construction, 0-D neutral inventory, no SEE) until Phase 0 changes the
  closure - and then of *that* closure, with its alpha.
* A self-consistent anomalous mobility, azimuthal structure or instability heating from any 2-D
  label (audit section 6); "3-D-verified" for anything but the designs actually run in 3-D at the
  recorded sector and resolution; a 3-D number from a sector whose convergence (Phase 1 item 4)
  was not recorded.
* Anything Arena Physica claims about Heaviside as if it were established: the figures in 1.6 are
  company statements on a company benchmark; the only checkable object is the public EMVal split.
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
* Physics completeness (Phase 0): `modern/docs/pic2d-physics-completeness-audit.md` (`0901138a`); ledger correction v2.0.6 `modern/src/cft_revival/pic2d/ledger_recompute.py` and the `ledger-corrected.json` sidecars under each `results/`; `modern/docs/literature/pic-mcc-blockers.md` (Charoy 2019 ref. 19, Villafana 2021 ref. 110, Zhao and Zhao 2026 ref. 116).
* 3-D design (Phase 1): `modern/docs/pic3d-design.md` - **forthcoming, not yet written**; the 3-D orbit integrator that verifies the pusher: `modern/src/cft_revival/orbit_mc/`.

## 7. File references (`Reality-Simulator/ai`, read-only)

`README.md`; `IMPLEMENTATION_STATUS.md`; `TODO.md`; `docs/README.md` (normative order); `docs/ARCHITECTURE.md`;
`docs/ARCHITECTURE_DETAIL.md` (sections 4, 4.1, 12); `docs/FULL_PHYSICS_SURROGATE_PLAN.md` (sections 1-3, 7.1, 13.4,
14.2-14.3, 15, 33, 42.4); `docs/SURROGATE_ARCHITECTURES.md`; `docs/LITERATURE_REVIEW.md`; `docs/DATA_PLANE.md`;
`docs/EXPORTER_AND_HETEROGENEOUS_STATE.md`; `docs/PLANNING_AND_REALTIME.md`; `docs/TRAINING_PLAN.md`;
`docs/VALIDATION_PLAN.md`; `docs/DATASET_GENERATION_CAMPAIGN.md`; `paper/latex/sections/01_introduction_related.tex`,
`02_export_representation_architecture.tex`, `03_training_optimisation_control.tex`,
`04_protocol_status_conclusion.tex`, `05_appendix_references.tex`; `paper/ARCHITECTURE_PAPER_DRAFT.md`;
`configs/surrogate/{baseline.yaml, model_ladder.yaml}`;
`configs/datasets/{production.yaml, s355_gmaw_tfillet_pilot_campaign.json}`; `configs/optimisation/default.yaml`;
`src/reality_ai/surrogate/{model.py, fno.py, unet.py, tfno.py, ufno.py, backbone_factory.py, configurable_model.py,
control_encoder.py, objective_decoder.py, query_decoder.py, field_decoder.py, ensemble.py, multifidelity.py,
uncertainty.py, geometry_encoder.py, material_encoder.py, multiphysics_fusion.py (CodomainAttentionFusion3d),
heterogeneous_encoder.py, latent_dynamics.py}`;
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

Added in the 2026-09-05 revision (sections 1.5, 1.6, Phase 1):

* Transformer / latent-token cores cited by `Reality-Simulator/ai`: Alkin et al., "Universal
  Physics Transformers", arXiv:2402.12365; Herde et al., "Poseidon: Efficient Foundation Models for
  PDEs", arXiv:2405.19101; Rahman et al., CoDA-NO, arXiv:2403.12553 (already above).
* Arena Physica (company publications, no peer review; accessed 2026-09-05): Bryant et al.,
  "Introducing Atlas RF Studio: Toward a Foundation Model for Electromagnetics", 2026-03-31,
  <https://www.arenaphysica.com/publications/rf-studio>; Bryant et al., "Introducing Heaviside-1",
  2026-09-01, <https://www.arenaphysica.com/publications/heaviside-1>; dataset card
  `ArenaPhysica/EMVal` (v0 public split, 500 boards), <https://huggingface.co/datasets/ArenaPhysica/EMVal>.
* LANDMARK benchmarks for the 3-D verification (both already in `pic-mcc-blockers.md`): Charoy et
  al., "2D axial-azimuthal particle-in-cell benchmark for low-temperature partially magnetized
  plasmas", Plasma Sources Sci. Technol. 28, 105010 (2019), doi:10.1088/1361-6595/ab46c5;
  Villafana et al., "2D radial-azimuthal particle-in-cell benchmark for E x B discharges", Plasma
  Sources Sci. Technol. 30, 075002 (2021), doi:10.1088/1361-6595/ac0a4a. Parameters quoted in
  Phase 1 were read from the published tables (Table 1 of each) on 2026-09-05.
