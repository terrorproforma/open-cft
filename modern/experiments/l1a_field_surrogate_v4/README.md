# L1a multi-fidelity field surrogate v4

This is a prospective numerical-emulation experiment on
`exp/l1a-field-surrogate-v4`. It does not claim material, plasma, thermal,
structural, propulsion or hardware accuracy.

The preregistration is prepared before commit:

```powershell
cd modern
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m experiments.l1a_field_surrogate_v4.run prepare
pytest tests/experiments/l1a_field_surrogate_v4 -q
```

The only real execution is made from a fresh, clean detached checkout of the
pushed preregistration:

```powershell
python -m experiments.l1a_field_surrogate_v4.run execute
```

The result commit must be the direct child of the preregistration. The exclusive
lock is retained permanently and no patch or rerun is permitted.
