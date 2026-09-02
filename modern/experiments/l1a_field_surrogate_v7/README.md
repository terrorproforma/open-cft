# L1a multi-fidelity field surrogate v7

This prospective numerical-emulation experiment is based on accepted
`cft_revival.experiment_runtime` commit
`231873d23fa242b15d0c085447a3d34ad55162a7`. V1-v6 outcomes remain immutable;
only the frozen v6 development rejection informed this protocol. V7 does not claim material, plasma,
thermal, structural, propulsion, or hardware accuracy.

The protocol screens 768 fresh input rows and freezes 224 candidate plus 48
method, calibration, and assessment rows. Each later role contains 16
interpolation, boundary, and OOD rows. Candidate budgets are 128/176/224.
Every model requires observed coarse data.

Scalar emulators fit per-output standardized transformed high-minus-low
discrepancies using the relevant coarse QOI and ARD Mahalanobis Matérn-5/2
kernels. Mirror ratios use the preregistered regularized near-null transform;
source error additionally receives radial/axial grid-phase features. Field
emulators canonicalize polarity, align piecewise through every stage centre and
chamber landmark, and fit separate stage-count cylindrical-energy POD bases
normalized by coarse energy. Projection-oracle field, energy, and topology
gates must pass before coefficient regressors using observed low-field modal
coefficients are fitted.

The preregistration is prepared before commit:

```powershell
cd modern
$env:PYTHONDONTWRITEBYTECODE = "1"
python -m experiments.l1a_field_surrogate_v7.run prepare
pytest tests/experiments/l1a_field_surrogate_v7 -q
```

Preparation performs input-only geometry screening and executes the complete
synthetic scientific callback matrix through the shared runtime, covering all
five terminal states without real field labels. The only real execution is
then made from a fresh, clean detached checkout of the pushed preregistration:

```powershell
python -m experiments.l1a_field_surrogate_v7.run execute
```

Detached/commit/dependency/remote/global-clean verification and immutable
attestation happen before constructing the runtime. An atomic persistent
attempt claim in the Git common directory prevents a second worktree from
starting the v7 namespace. A runtime-phase drift check
then permits untracked entries only beneath the exact runtime-owned result and
cache roots while rejecting every tracked or staged change. The shared runtime owns result-root preflight, immutable locking, atomic
artifact pairs, cache cleanup, failures, and terminal publication. The result
commit must be the direct child of the preregistration; no patch or rerun is
permitted.
