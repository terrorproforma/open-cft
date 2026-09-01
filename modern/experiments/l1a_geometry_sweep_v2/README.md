# Preregistered L1a geometry sweep v2

This experiment is a one-execution, 96-case, deterministic shifted-Halton
screen of axisymmetric geometries using the accepted geometry v1.1
non-authoritative current-equivalent preview and accepted L1a Warp field
solver. `protocol.json` is the sealed preregistration authority.

The protocol fixes the sample, solver, QoIs, objective directions, five
representative roles, role coalescence, numerical replay tolerances, failure
taxonomy and seven terminal gates before execution. Zero case failures is an
additional mandatory acceptance condition. Representative fields are archived
from each selected case's primary CUDA solve; selected cases are never rerun
and role coalescence is never padded.

The result is field-only screening evidence. It does not calculate thrust,
efficiency, Isp, plasma, thermal or structural performance, and it must not be
described as hardware-valid. Wall-clock timing is an uncontrolled diagnostic
because other GPU work may be concurrent.

Run only after the preregistration commit is pushed:

```powershell
$env:PYTHONPATH="$PWD\modern\src;$PWD\modern"
python -m experiments.l1a_geometry_sweep_v2.run
```

The execution lock makes a second invocation fail before solving any case.
