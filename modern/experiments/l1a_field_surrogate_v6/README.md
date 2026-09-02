# L1a multi-fidelity field surrogate v6

This prospective numerical-emulation experiment is based on accepted
`cft_revival.experiment_runtime` commit
`231873d23fa242b15d0c085447a3d34ad55162a7`. V1-v5 outcomes remain immutable;
v5 failed in prebundle before solver or label access. V6 does not claim material, plasma,
thermal, structural, propulsion, or hardware accuracy.

The preregistration is prepared before commit:

```powershell
cd modern
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m experiments.l1a_field_surrogate_v6.run prepare
pytest tests/experiments/l1a_field_surrogate_v6 -q
```

Preparation performs input-only geometry screening and executes the complete
synthetic scientific callback matrix through the shared runtime, covering all
five terminal states without real field labels. The only real execution is
then made from a fresh, clean detached checkout of the pushed preregistration:

```powershell
python -m experiments.l1a_field_surrogate_v6.run execute
```

Detached/commit/dependency/remote/global-clean verification and immutable
attestation happen before constructing the runtime. A runtime-phase drift check
then permits untracked entries only beneath the exact runtime-owned result and
cache roots while rejecting every tracked or staged change. The shared runtime owns result-root preflight, immutable locking, atomic
artifact pairs, cache cleanup, failures, and terminal publication. The result
commit must be the direct child of the preregistration; no patch or rerun is
permitted.
