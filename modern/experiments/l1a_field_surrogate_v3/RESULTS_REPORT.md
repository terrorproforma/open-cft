# L1a field-surrogate v3 prospective result

- Branch: `exp/l1a-field-surrogate-v3`
- Preregistration: `98bba6344b8422c918ab091eb593d09bd693b143`
- Raw geometries: 506 valid, 6 rejected, 64 constructor-corrected
- Frozen geometry rebuilds: 240/240 exact
- Candidate+method coarse solves: 144/144
- Candidate+method fine solves: 144/144
- Model-fit accesses: 1
- Calibration solver/label accesses: 0
- Assessment solver/label accesses: 0
- Terminal status: **failed execution during method selection**

The first field-model snapshot build raised
`NameError: name 'math' is not defined`. The exclusive execution lock had
already been claimed, so the preregistered no-patch/no-rerun rule forbids
repair or continuation.

No model family, POD rank, kernel length or training budget was selected.
Development accuracy, assessment, topology, group coverage and latency gates
were not evaluated. The strict failure bundle and its pre-solve 149-blob
dependency closure validate successfully.

This is a valid prospective software-execution failure. It provides no
material, plasma, propulsion, thermal, structural, hardware, physical-accuracy
or surrogate-performance evidence.
