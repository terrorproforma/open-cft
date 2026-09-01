# Verification, Validation, and Evidence Workstream Report

## Audit closure

Version 2 binds claims to source authority instead of trusting a record-selected
claim. Analytical/manufactured references can verify implementation;
independent code, simulations/PIC, and published model outputs can support only
cross-model agreement. Predictive-validity authority is reserved for actual
experiment records with facility, hardware, campaign, UTC acquisition,
instrument, raw-data, measured-truth provenance, and uncertainty metadata.

Context-of-use audits now match exact quantity names and SI units, independent
group counts, operating ranges, model/result contexts, and model/code
revisions. Independence is derived from immutable design, run, hardware,
campaign, and specimen identities, and calibration overlap is checked inside
the context audit. Conservation status is recomputed from raw bound residuals
and gate policy. Every convergence level binds its own evidence hash and raw
error/spacing observations, from which order is recomputed. PIC ensembles
require homogeneous identities, at least three unique seeds, and use a
two-sided conservative tabulated Student-t 95% mean interval.

Bundle parsing verifies integrity before enforcing a closed nested schema.
Reports recompute partition status from the exact registry identity and cannot
accept caller-supplied audit objects. Empty and otherwise insufficient
registries are `NOT_EVALUATED`; only an evaluated failed gate is `FAIL`.

## Published 2020 S1 evidence

The source-native labels and published values are preserved exactly:

- `MDO (original)`: 102.7 mN, 36.5%, 2131 s;
- `PIC`: 62.8 mN, 15.2%, 1333 s;
- `MDO (modified)`: 61.7 mN, 14.6%, 1280 s.

Source: Yeo et al. (2020), *Multiobjective Optimization and Particle-in-Cell
Simulation of Cusped Field Thrusters for Microsatellite Platforms*,
DOI [10.2514/1.A34584](https://doi.org/10.2514/1.A34584).

Descriptions such as “corrected low-fidelity,” “2D3V PIC-MCC,” and
“PIC-informed reduced model” are stored separately as editorial
interpretations. These records are published external model outputs for
cross-model benchmarking, not experiments, truth, calibration targets, or
acceptance tolerances.

## Integrity limitation

Canonical SHA-256 detects accidental changes and provides version identity. It
does not authenticate the source: a malicious party able to replace the payload
can recompute the hash. Citation checks and authenticated distribution remain
separate obligations. URI parsing requires a valid absolute HTTP(S) authority,
and DOI parsing enforces the standard registrant/suffix shape.
