# L1a multi-fidelity field surrogate v10

This prospective numerical-emulation experiment is based on accepted
`cft_revival.experiment_runtime` commit
`b46e263950f91530ea61710b5dcc9354fc63cf6c`. V1-v9 outcomes remain immutable;
only the frozen v9 development rejection informed this protocol. V10 does not claim material, plasma,
thermal, structural, propulsion, or hardware accuracy.

The protocol screens 1,024 fresh input rows and freezes 270 candidate plus 54
method, calibration, and assessment rows. Every role balances stage count,
input-only interpolation/boundary/OOD stratum, and both polarities. Candidate
budgets 162/216/270 are complete balanced maximin prefixes.
Every model requires observed coarse data.

The primary model directly interpolates complete lossless physical-grid
residual snapshots. It uses standardized geometry plus input-only coarse-field
descriptors, with pooled/stage-specific scope, 8/16 neighbours and fixed
Wendland-C2/inverse-distance kernels. Three true held-out candidate folds
exclude every validation row from fitting. Duplicate, compact-support, OOD,
and neighbour-spread behavior is deterministic. Independent bulk Br/Bz,
axis-Bz, and overlapping landmark-patch bases form a non-neural comparator.
All field QoIs derive from the reconstructed field; source error is input-only.

The preregistration is prepared before commit:

```powershell
cd modern
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m experiments.l1a_field_surrogate_v10.run prepare
pytest tests/experiments/l1a_field_surrogate_v10 -q
```

Preparation performs input-only geometry screening and executes the complete
synthetic scientific callback matrix through the shared runtime, covering all
five terminal states without real field labels. The only real execution is
then made from a fresh, clean detached checkout of the pushed preregistration:

```powershell
python -m experiments.l1a_field_surrogate_v10.run execute
```

Detached/commit/dependency/remote/global-clean verification and immutable
attestation happen before constructing the runtime. An atomic persistent
attempt claim in the Git common directory prevents a second worktree from
starting the v10 namespace. A runtime-phase drift check
then permits untracked entries only beneath the exact runtime-owned result and
cache roots while rejecting every tracked or staged change. The shared runtime owns result-root preflight, immutable locking, atomic
artifact pairs, cache cleanup, failures, and terminal publication. The result
commit must be the direct child of the preregistration; no patch or rerun is
permitted.
