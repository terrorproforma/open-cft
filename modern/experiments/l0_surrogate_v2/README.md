# Preregistered L0 surrogate experiment v2

This isolated experiment supersedes none of the accepted runtime. The separate
`l0_surrogate/` experiment remains failed development evidence: it improved
point errors but failed its preregistered coverage gate.

V2 freezes three input-only group partitions, fixed 96-row active and baseline
budgets, per-stratum split-conformal intervals, and independent per-stratum and
overall gates. Final assessment is unavailable to acquisition and is loaded
once only after selection, model, and calibration identities are frozen.

The first commit contains no assessment labels or results. Execution is
single-shot and requires the full preregistration commit SHA:

```powershell
$env:PYTHONPATH="$PWD\src"
python experiments/l0_surrogate_v2/protocol.py execute `
  --preregistration-commit <40-character-sha>
```

All claims are limited to deterministic L0 software emulation. Hashes establish
artifact integrity, not physical validity.
