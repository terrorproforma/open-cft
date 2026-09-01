# Geometry workstream devlog

## 2026-09-02 — foundation

- Restricted implementation to new `geometry/` paths owned by this workstream.
- Inspected the legacy FYP FEMM construction and separated fixed values,
  commented examples, broad optimizer bounds, and new assumptions.
- Implemented immutable typed SI geometry, strict cross-field validation,
  closed canonical JSON, and payload hashing.
- Implemented a PPM/pole generator with explicit axial polarity and a hard
  TWT/CFT physics boundary.
- Added historical-envelope, compact short-pitch, and divergent-exit
  hypothetical variants.
- Added exact geometry descriptors and stable optimizer-vector ordering. No
  performance quantities were introduced.
- Added solver-neutral, accepted material-aware magnetics, and explicit L1a
  preview adapters. Permanent-magnet representation authority is mutually
  exclusive.
- Added deterministic JSON, viewer-data JSON, SVG, raw-file hashes, and an
  artifact manifest generator. DXF was intentionally omitted.
- Added focused tests for validation, overlap, ordering, alternating polarity,
  tolerances, hashes, legacy mapping, adapters, descriptors, schemas, and SVG.

## Verification log

Commands and results:

```text
python -m pytest tests/geometry -q
# 18 passed
python -m pytest tests/geometry tests/magnetics tests/fields -q
# 112 passed
python -m compileall -q src/cft_revival/geometry examples/geometry tests/geometry
# passed
python examples/geometry/generate_reference_artifacts.py
# passed; second regeneration retained all 10 non-sidecar artifact hashes
git diff --no-index --stat -- FYP/FEMMrun.m modern/examples/geometry/artifacts/historical-envelope-baseline.json
# provenance comparison completed: 1 file, 1 insertion, 231 deletions
python -m ruff check ...
python -m mypy ...
# not run: optional tools are not installed; dependency installation was forbidden
```

The last command is expected to report textual differences because it compares
legacy MATLAB source with a new canonical JSON artifact; it is used only as a
human-visible provenance check and not as a zero-diff gate.

## 2026-09-02 — audit hardening

- Bumped geometry and artifact-manifest contracts to 1.1.0.
- Made material kind/ID ownership authoritative and changed the magnetics
  adapter to require an exact caller-supplied registry.
- Bound one permanent-magnet authority into a configuration-specific plan ID;
  removed call-time authority switching.
- Wrapped L1a current-equivalent output in a permanently non-authoritative
  preview type.
- Closed stage-to-region centers, duplicate references, role/material kinds,
  chamber coverage, divergent continuity, and region-graph invariants.
- Added complete segmented inner/outer/upstream/downstream interface topology,
  reciprocal adjacency, symmetry-axis semantics, and taper unit normals.
- Made nominal clearance strictly exceed thermal/minimum requirements after
  both-sided tolerances; removed descriptor snapping.
- Added finite/representable descriptor, density, volume, and mass-underflow
  publication checks.
- Restricted IDs and filenames to canonical safe forms and escaped SVG dynamic
  text/attributes.
- Standardized JSON artifacts on no trailing newline.
- Added a strict bundle loader covering duplicate keys, closed projections,
  file/payload hashes, sidecars, substitutions, paths, and extra files.
- Added a material_fields integration change notice.

Final audit verification:

```text
python -m pytest tests/geometry -q
# 36 passed
python -m pytest tests/geometry tests/magnetics tests/fields -q
# 130 passed
python -m compileall -q src/cft_revival/geometry examples/geometry tests/geometry
# passed
python examples/geometry/generate_reference_artifacts.py
# passed twice; all 20 files byte-stable
load_artifact_bundle(Path("examples/geometry/artifacts"))
# 3 strict geometry bundles loaded
git diff --no-index --stat -- FYP/FEMMrun.m modern/examples/geometry/artifacts/historical-envelope-baseline.json
# provenance comparison completed: 1 file, 1 insertion, 231 deletions
```

Optional Ruff and mypy packages remain uninstalled, as required.

## 2026-09-02 — v1.1 acceptance closure

- Added serialized stage axial envelopes and model-level chamber containment,
  magnet membership, pole-after adjacency, and connected-stack checks.
- Rejected non-rectangular permanent magnets until an exact L1b handoff exists.
- Replaced global PM clearance with exact endpoint evaluation over every
  axially overlapping dielectric segment.
- Added adapter invariants proving every authoritative PM region appears in
  the handoff and every equivalent-current PM has one source.
- Allowlists now bind artifact generator identity `generate_reference_artifacts.py`,
  generator version `1.1.0`, and the exact hypothetical/no-performance claim.
- Documented that sidecar hashes provide integrity, not publisher
  authenticity.
- Added the L1b migration notice and adversarial regressions for coherent
  shifted stacks, stage envelopes, pole adjacency, tapered PM endpoints,
  handoff counts, and consistently rehashed semantic substitution.

Final acceptance results:

```text
python -m pytest tests/geometry -q
# 41 passed
python -m pytest tests/geometry tests/magnetics tests/fields -q
# 135 passed
python -m compileall -q src/cft_revival/geometry examples/geometry tests/geometry
# passed
python examples/geometry/generate_reference_artifacts.py
# repeated regeneration retained all 20 hashes; 3 strict bundles loaded
git diff --no-index --stat -- FYP/FEMMrun.m modern/examples/geometry/artifacts/historical-envelope-baseline.json
# provenance comparison completed: 1 file, 1 insertion, 231 deletions
```

No dependency installation, benchmark, commit, or push was performed.
