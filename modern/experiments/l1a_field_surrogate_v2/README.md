# Preregistered L1a multi-fidelity field surrogate v2

V2 prospectively corrects v1's input-only geometry preparation. Before role
partitioning, every fresh raw candidate is passed through the actual accepted
geometry-v1.1 constructor and L1a preview. The requested divergent endpoint is
tried first; continuity failures follow a fixed `nextafter` sequence, with the
real constructor as the only acceptance oracle. Invalid raw rows are recorded,
then exactly 112 valid rows are frozen and rebuilt hash-identically.

The field experiment remains unchanged in spirit: 112 coarse 41x73 solves,
80 nested accepted 81x145 solves, high-only/AR1/coarse-discrepancy scalar GPs,
high-only/discrepancy POD fields, role-safe model selection, exact-rank group
conformal calibration, single-use assessment, topology/OOD rejection, and
complete coarse-plus-inference latency.

This is accepted-L1a numerical emulation only. It makes no material, plasma,
thermal, structural, propulsion, build, hardware, or physical-accuracy claim.

After pushing the preregistration commit:

```powershell
$env:PYTHONPATH="$PWD\modern\src;$PWD\modern"
python -m experiments.l1a_field_surrogate_v2.run execute
```

The Git-common-dir lock permanently forbids a second execution.
