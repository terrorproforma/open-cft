# Independent P2 FEM reference artifacts

From `modern/`:

```powershell
$env:PYTHONPATH='src'
python examples/fem_reference/run_reference_campaign.py
python examples/fem_reference/replay_artifacts.py
# Only after the 8-GiB free-RAM preflight passes:
python examples/fem_reference/run_reference_campaign.py `
  --design historical-envelope-baseline --allow-third-level
```

The campaign requests one graded body-fitted mesh plus two nested adaptive
levels for each hypothetical compact, divergent, and historical-envelope
geometry. Normal execution stops before a projected level exceeds `400,000`
P2 DOFs. The revised ceiling is `1,500,000`, but using it requires explicit
third-level opt-in, exactly one design, and at least `8 GiB` free physical RAM.
Each level also has a calibrated COO/CSR/solver peak estimate with a `1.75`
safety factor and scale-dependent allocator reserve capped at `256 MiB`.
Checkpoint parsing/decompression and serialization buffers are included. The
same typed guard rechecks RAM and the `1,500,000`-DOF/topology caps before all
heavy mesh, solve, assembly, artifact, replay, validation, and publication
phases. This is a resource-policy change,
not an accuracy-gate relaxation.
Low-RAM startup, level, topology, and cap failures all raise typed
`ResourceBlockedError` with status/reason `NOT_EVALUATED`.
Readiness work below both `100,000` P2 DOFs and `64 MiB` serialized state
remains cap- and topology-checked but does not invoke the live-RAM gate
reserved for heavy allocation.
Deterministic component-wise Dörfler marking covers residual,
flux-jump, and bore-window proxy indicators; strict `1.3` gradation closure is
never relaxed. The full P2 artifact and viewer projection use the finest
completed level. `manifest.json` records actual local mesh sizes, P2 DOFs,
quality, refinement ancestry, volume-average QoIs, observed orders, L1b
comparisons, resource diagnostics, and complete topology hashes.
Each completed level also writes a hash-sealed checkpoint under
`artifacts/checkpoints/`, so a bounded per-design run leaves deterministic
progress evidence even if a later solve is interrupted.
The sealed manifest anchors each checkpoint file and payload hash and chains
checkpoint-file and parent-mesh ancestry. Schema-1.3 checkpoints bind complete
mesh/topology/problem/solver/solution arrays in compressed binary sidecars,
allowing independent replay without giant duplicate JSON arrays. The ordered
chain also binds design/config/geometry/magnetics/code/problem and final
run/mesh identities, so an unrelated valid chain cannot be substituted.
Checkpoint metadata is limited to `8 MiB`; actual NPY header dimensions and
ZIP sizes are verified before allocation and compared with anchor counts.
Preliminary checkpoints use the same sidecar format, and finalization uses
bounded metadata parsing and header-guarded loading rather than `read_bytes()`.

These files are independent numerical-reference evidence. They are not
hardware validation. Interface/cell maxima are intentionally excluded from
the one-percent acceptance decision. Current large schema-1.1 artifacts are
legacy integrity-only screening evidence. New schema-1.3 artifacts recompute
acceptance quantities from bound mesh/solution/config/code evidence. A
phase-matched fixed-local-`h` domain-expansion study at padding
`0.5/1.0/1.5` is also required before acceptance; all extents and local
resolutions must be finite, positive where applicable, nested, and recomputed
from bound grids.

Completed explicit-opt-in qualification evidence is stored one design per
directory under `artifacts/third-level/`. The September 2026 campaign
labels `divergent-exit-stack` as `NUMERICAL_P2_QUALIFIED`;
`historical-envelope-baseline` and `compact-high-gradient-stack` are
`SCREENING_ONLY` because one or more actual-`h` observed orders were negative.
All three passed the unchanged sub-percent two-change and phase-matched domain
gates. Timing and memory values are `DIAGNOSTIC_ONLY`. No result represents
hardware validation.

If all adaptive and domain solves completed but publication was interrupted,
the guarded recovery command promotes the existing binary checkpoint chains
and reruns authoritative acceptance replay without rerunning the solve:

```powershell
python examples/fem_reference/recover_completed_qualification.py `
  --design historical-envelope-baseline
```
