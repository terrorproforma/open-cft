# Preregistered L0 surrogate experiment v3

V3 inherits the hash-pinned v2 scientific protocol without changes. Its only
versioned delta is a root-confined atomic serializer plus a mandatory synthetic
preflight. V2 remains immutable failure evidence.

Before the freeze commit:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m experiments.l0_surrogate_v3.protocol partitions
python -m experiments.l0_surrogate_v3.protocol preflight --record
python -m pytest -q tests/experiments/l0_surrogate_v3
```

After that commit is pushed, execute exactly once with its full SHA:

```powershell
python -m experiments.l0_surrogate_v3.protocol execute `
  --preregistration-commit <40-character-sha>
```

Hash integrity is not physical validity. Any successful metrics apply only to
emulation of the deterministic L0 software sweep.
