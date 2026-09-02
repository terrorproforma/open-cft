# Offline P2 FEM qualification dashboard

From `modern/`:

```powershell
python examples/fem_reference/visualization/generate_dashboard.py
pytest tests/fem_reference_visualization/test_dashboard.py
```

Open `fem-reference-p2-qualification.html` directly in a browser. It is a
standalone file: all evidence projections, styles, and JavaScript are inline,
and no network request is made.

The generator pins the three accepted third-level manifests and verifies every
referenced artifact, viewer, checkpoint metadata file, and NumPy sidecar before
rendering. The dashboard labels only `divergent-exit-stack` as **NUMERICAL P2
QUALIFIED**. Historical and compact evidence remain **SCREENING ONLY** because
one or more actual-local-h observed orders are non-positive.

This is independent numerical evidence, not hardware or experimental
validation. It makes no material-plasma or device-performance claim.
