# CFT full-orbit wall-loss campaign v4 (shakedown-gated)

This directory preregisters nine explicit collisionless prescribed-field
test-particle wall-loss cases using the independently
`NUMERICAL_P2_QUALIFIED` divergent-exit P2 evidence, exactly as v3 did, and
adds the one thing v1, v2 and v3 lacked: a disclosed **non-evidentiary
shakedown of the complete production path on the real P2 fields** that must
pass before the preregistration can be frozen.

v1 (`prebundle_failure`), v2 (`runtime_failure`) and v3 (`runtime_failure`)
each passed a synthetic preflight and then died on code, never physics, the
first time the real field met the production path inside the one-shot run.
Their exact failure strings and root causes are in
`protocol.json#prior_campaign_disclosure`.

## Lifecycle

From `modern/` (PowerShell):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
python -m experiments.cft_orbit_wall_loss_v4.run shakedown   # any HEAD, dirty allowed, BEFORE prepare
python -m experiments.cft_orbit_wall_loss_v4.run prepare     # refuses without a valid shakedown.json
# commit and push the preregistration, then from a clean detached worktree:
python -m experiments.cft_orbit_wall_loss_v4.run execute
python -m experiments.cft_orbit_wall_loss_v4.run validate
```

### `shakedown` (non-evidentiary)

Builds the real P2 → ψ adapter for all three roles exactly as production does,
generates a **disjoint** launch design (separate campaign prefix
`cft-orbit-wall-loss-v4-shakedown`, RNG positions in the seed namespace
`cft-orbit-wall-loss-v4:shakedown`, gyrophase offset 17π/96, 2 launches per
stratum = 64 per case, 8 batches of 8, partial checkpoint at 4) and drives the
full production path for all nine cases through the shared `ExperimentRuntime`
into a **temporary** result root: `preflight_campaign` → integration →
every checkpoint/witness/artifact validator → partial/resumed/final checkpoint
chain → estimators and stratum summaries → gate report (evaluated,
informational, non-binding) → sealed artifacts with deterministic replay,
verified reload and the export payload → `validate_bundle`.

It writes `shakedown.json` into this directory: git HEAD and dirtiness,
the orbit_mc package version and source hash (all `orbit_mc/*.py` +
`spec/orbit_mc/*.json`, LF bytes), the protocol semantic hash, per-case
termination counts, validator pass/fail counts and messages, tolerance-close
event counts and event-resolution counts, timing, maximum relative energy
error and the count of orbits with non-zero energy error, the count of orbits
whose final velocity equals the witnessed event velocity, the μ diagnostic
(min/median/max), the informational gate results, the P2 and label access
records, the shakedown design hash and a disjointness proof (zero overlap of
launch IDs, seeds, positions and (energy, pitch, direction, gyrophase) tuples)
against the evidentiary v4 design and v1/v2/v3. Shakedown outcomes never
enter any estimand.

### `prepare` (refuses unless the shakedown gate is open)

`prepare` refuses to freeze unless `shakedown.json` exists, `passed == true`
(all validators passed for all nine cases, bundle validated), its protocol
hash equals the current protocol hash, its orbit_mc source hash equals the
current orbit_mc source hash, its design hashes reproduce, and disjointness is
proven. `authorities.json` binds the orbit_mc package version
(`cft_revival.orbit_mc.__version__ == "1.6.0"`), the result/checkpoint/
validation-protocol/handoff schema versions, the orbit_mc source hash and the
shakedown file hash; `execute` re-verifies all of them at bind time and the
prebundle callback re-verifies them again inside the runtime.

### `execute` (one immutable attempt)

Identical to v3: clean detached pushed preregistration commit, Git-common lock,
shared `experiment_runtime` bundle with access-before-label records, no patch,
no rerun. Cases run in a spawn process pool **after** all nine label accesses
have been recorded in case order; each orbit is a pure function of (launch,
field, config), a main-process determinism sample re-integrates two launches
per case, and the deterministic replay validator re-integrates every case
before its artifact is sealed. Coupling is export-only and is emitted only
after every preregistered gate passes.

## Design

Each primary/refined/enlarged × N/2N/4N case has a case-equal
`ensemble_id`/`campaign_id`, 512 case-prefixed launch IDs, eight batches and
independent launch/batch/estimator/field/config/policy/case hashes. The 32
strata (4 cells × 2 energies × 2 pitches × 2 directions) with two position
repeats and eight gyrophases are fresh relative to v3: new radii (0.675 and
0.800 of the wall radius), new axial samples and a gyrophase grid rotated by
+π/12 relative to v3. The wall estimand is restricted to the straight
cylindrical dielectric (`r = 2 mm`, `1 ≤ z ≤ 18 mm`); radial exit in the
divergent section is domain escape.

## Gates and diagnostics under orbit_mc v1.6

All v3 numerical gates are kept and are **binding** in the evidentiary run,
including the 1e-10 relative energy gate. orbit_mc v1.6 (`3ab50ef5`) defines
the event velocity as the Boris state at the event time (a sub-push over
`event_fraction × step_dt` with the step's midpoint fields, recorded in the
witness as `event_velocity_m_per_s`, `step_magnetic_midpoint_t`,
`step_electric_midpoint_v_per_m`; failure witnesses carry zeros). In the
prescribed pure-B P2 field this makes the per-orbit relative energy drift
exactly `0.0`, so the gate is satisfiable; v1.5's chord-interpolated event
velocity carried up to ~1e-3 and could not meet it. v4 additionally requires
`final_velocity_m_per_s == event_velocity_m_per_s` exactly for every orbit
(`gates.require_final_velocity_equals_event_velocity`).

**Magnetic-moment variation (μ) is a diagnostic only.** Each case reports
min/median/max of `maximum_instantaneous_mu_relative_variation` and the count
of orbits above 0.1 and 0.5, under `diagnostics_not_gates` — never among the
gate checks, binding or informational. Electrons in the divergent-exit cusp
field are strongly non-adiabatic where the gyroradius approaches the field
scale length, so large μ variation is the measured physics of this
configuration, not a numerical defect; numerical fidelity is established by
the binding energy gate, the runtime rotation bound, the manufactured
convergence orders and the CPU/CUDA parity gate. See
`protocol.json#diagnostics.magnetic_moment_variation`.

No result is PIC, self-consistent plasma, experimental validation, hardware
validation, total thruster performance, or mirror-formula publication.
