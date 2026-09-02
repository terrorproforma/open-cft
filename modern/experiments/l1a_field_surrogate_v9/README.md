# L1a multi-fidelity field surrogate v9

This prospective numerical-emulation experiment is based on accepted
`cft_revival.experiment_runtime` commit
`b46e263950f91530ea61710b5dcc9354fc63cf6c`. V1-v8 outcomes remain immutable;
only the frozen v8 development rejection informed this protocol. V9 does not claim material, plasma,
thermal, structural, propulsion, or hardware accuracy.

The protocol screens 1,024 fresh input rows and freezes 270 candidate plus 54
method, calibration, and assessment rows. Every role balances stage count,
input-only interpolation/boundary/OOD stratum, and both polarities. Candidate
budgets 162/216/270 are complete balanced maximin prefixes.
Every model requires observed coarse data.

The primary representation is the polarity-canonical physical-grid
fine-minus-prolonged-coarse residual. Its primary augmented block decodes
exactly without warping or unalignment. Auxiliary channels comprise
cylindrical energy, overlapping smooth tapered landmark windows, and axis-Bz
null/extrema derivatives. Candidate-only grouped physical gates select among
ranks 64/96/128/192 before the method oracle. All field QoIs derive from the
reconstructed physical field; source representation error remains input-only.
Qualified coefficient models use observed coarse low-basis coefficients,
whitening, candidate-only ARD and per-mode regularization.

The preregistration is prepared before commit:

```powershell
cd modern
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m experiments.l1a_field_surrogate_v9.run prepare
pytest tests/experiments/l1a_field_surrogate_v9 -q
```

Preparation performs input-only geometry screening and executes the complete
synthetic scientific callback matrix through the shared runtime, covering all
five terminal states without real field labels. The only real execution is
then made from a fresh, clean detached checkout of the pushed preregistration:

```powershell
python -m experiments.l1a_field_surrogate_v9.run execute
```

Detached/commit/dependency/remote/global-clean verification and immutable
attestation happen before constructing the runtime. An atomic persistent
attempt claim in the Git common directory prevents a second worktree from
starting the v9 namespace. A runtime-phase drift check
then permits untracked entries only beneath the exact runtime-owned result and
cache roots while rejecting every tracked or staged change. The shared runtime owns result-root preflight, immutable locking, atomic
artifact pairs, cache cleanup, failures, and terminal publication. The result
commit must be the direct child of the preregistration; no patch or rerun is
permitted.
