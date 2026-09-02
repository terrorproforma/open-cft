# L1a geometry-sweep dashboard

`l1a-geometry-sweep.html` is a deterministic, self-contained offline viewer
for the reviewed 96-case L1a field-only sweep. Open the HTML file directly in a
modern browser; it has no fetch, CDN, font, image, or other network dependency.

Regenerate from `modern/`:

```powershell
python experiments/l1a_geometry_sweep/visualization/generate_dashboard.py
```

The generator fails closed unless the manifest, dataset, report, and every
representative geometry/full-field/downsampled-field artifact match both their
SHA-256 sidecars and manifest bindings. The reviewed manifest and dataset file
and canonical payload identities are pinned in the generator.

Run focused checks:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m pytest -q tests/experiments/l1a_geometry_sweep_visualization
python -m compileall -q experiments/l1a_geometry_sweep/visualization tests/experiments/l1a_geometry_sweep_visualization
```

The dashboard is a screening evidence viewer only. Equivalent-current L1a
results are not a material-aware permanent-magnet model, plasma solution,
propulsion result, or hardware validation.
