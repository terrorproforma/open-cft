# L1a field-surrogate v1 prospective result

- Preregistration: `6e8b74f2cefc2f4ed4e4745e6e9bc91580a09af7`
- Device: NVIDIA GeForce RTX 5090 (`cuda:0`)
- Execution count: exactly one
- Terminal status: **REJECTED — case preparation failure**
- Failed case: method-selection index 69
- Failure: accepted geometry v1.1 rejected divergent wall slopes as noncontinuous
- Patch/rerun: none

The protocol budget was 112 coarse solves and 80 nested fine solves, with
high-fidelity model budgets 24 and 32. Execution stopped at the first failure,
before all labels were available. Consequently no surrogate was selected and
scalar, field, topology, conformal-coverage, OOD-safety, latency, and
representative-prediction gates were not evaluated. The mandatory zero-case-
failure numerical gate failed, so overall acceptance is false.

This is a valid disclosed prospective failure. It provides no material,
plasma, propulsion, thermal, structural, hardware, or physical-accuracy
evidence; it was intended only to emulate the accepted L1a discretization.
