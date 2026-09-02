# L1a multi-fidelity field surrogate v5

This prospective numerical-emulation experiment is based on accepted
`cft_revival.experiment_runtime` commit
`231873d23fa242b15d0c085447a3d34ad55162a7`. V1-v4 outcomes remain immutable;
v4 failed before solver or label access. V5 does not claim material, plasma,
thermal, structural, propulsion, or hardware accuracy.

The preregistration is prepared before commit:

```powershell
cd modern
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m experiments.l1a_field_surrogate_v5.run prepare
pytest tests/experiments/l1a_field_surrogate_v5 -q
```

Preparation performs input-only geometry screening and executes the complete
synthetic scientific callback matrix through the shared runtime, covering all
five terminal states without real field labels. The only real execution is
then made from a fresh, clean detached checkout of the pushed preregistration:

```powershell
python -m experiments.l1a_field_surrogate_v5.run execute
```

The shared runtime owns result-root preflight, immutable locking, atomic
artifact pairs, cache cleanup, failures, and terminal publication. The result
commit must be the direct child of the preregistration; no patch or rerun is
permitted.
