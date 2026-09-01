# Preregistered L1a multi-fidelity field surrogate v1

This is a prospective, exactly-once numerical-emulation experiment. It learns
the discrepancy between a coarse 41x73 L1a solve and the accepted 81x145 L1a
solve over fresh geometry-v1.1 families. It does not emulate L0 algebra.

`predeclaration.json` fixes the 112 low-fidelity and 80 nested high-fidelity
budgets, disjoint development/calibration/single-use assessment roles,
candidate models, exact-rank group conformal intervals, numerical and
predictive gates, topology/OOD rejection policy, and end-to-end timing policy.

The high-fidelity target is an accepted L1a discretization—not physical truth.
No material, plasma, propulsion, thermal, structural, build, or hardware
accuracy is claimed. Accepted L1a-v2 is disclosed only as development evidence;
its coordinates and labels are excluded.

After the protocol commit is pushed, execution is:

```powershell
$env:PYTHONPATH="$PWD\modern\src;$PWD\modern"
python -m experiments.l1a_field_surrogate_v1.run execute
```

The retained Git-common-dir lock forbids a second run.
