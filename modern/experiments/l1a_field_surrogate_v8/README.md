# L1a multi-fidelity field surrogate v8

This prospective numerical-emulation experiment is based on accepted
`cft_revival.experiment_runtime` commit
`b46e263950f91530ea61710b5dcc9354fc63cf6c`. V1-v7 outcomes remain immutable;
only the frozen v7 development rejection informed this protocol. V8 does not claim material, plasma,
thermal, structural, propulsion, or hardware accuracy.

The protocol screens 1,024 fresh input rows and freezes 270 candidate plus 54
method, calibration, and assessment rows. Every role balances stage count,
input-only interpolation/boundary/OOD stratum, and both polarities. Candidate
budgets 162/216/270 are complete balanced maximin prefixes.
Every model requires observed coarse data.

Source representation error is calculated input-only on the high source grid.
Mirrors and gradients are derived from reconstructed fields. Remaining scalar
discrepancies are fitted only after candidate-only correlation qualification.
ARD lengths use deterministic candidate-only grouped CV and marginal
likelihood. Field emulators require a 0.5% alignment roundtrip and use
stage-local joint bases combining cylindrical-energy, unweighted axial-window,
and explicit axis-Bz channels. Projection field, energy, and topology gates
precede dedicated low-field-basis, whitened coefficient fitting.

The preregistration is prepared before commit:

```powershell
cd modern
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m experiments.l1a_field_surrogate_v8.run prepare
pytest tests/experiments/l1a_field_surrogate_v8 -q
```

Preparation performs input-only geometry screening and executes the complete
synthetic scientific callback matrix through the shared runtime, covering all
five terminal states without real field labels. The only real execution is
then made from a fresh, clean detached checkout of the pushed preregistration:

```powershell
python -m experiments.l1a_field_surrogate_v8.run execute
```

Detached/commit/dependency/remote/global-clean verification and immutable
attestation happen before constructing the runtime. An atomic persistent
attempt claim in the Git common directory prevents a second worktree from
starting the v8 namespace. A runtime-phase drift check
then permits untracked entries only beneath the exact runtime-owned result and
cache roots while rejecting every tracked or staged change. The shared runtime owns result-root preflight, immutable locking, atomic
artifact pairs, cache cleanup, failures, and terminal publication. The result
commit must be the direct child of the preregistration; no patch or rerun is
permitted.
