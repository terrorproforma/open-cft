# Commit-bound L0 surrogate experiment v4

V4 preserves v2/v3 as immutable failures and changes only pre-execution Git
identity binding. Its scientific protocol and thresholds are unchanged.

Before preregistration:

```powershell
$env:PYTHONPATH="$PWD\src"
python -m experiments.l0_surrogate_v4.protocol partitions
python -m experiments.l0_surrogate_v4.protocol preflight --record
python -m pytest -q tests/experiments/l0_surrogate_v4
```

After the exact-path commit is pushed, execution discovers and verifies Git
identity itself:

```powershell
python -m experiments.l0_surrogate_v4.protocol execute
```

An optional `--expected-head <sha>` is only an additional equality check; it
cannot override the SHA observed from `git rev-parse HEAD`.
