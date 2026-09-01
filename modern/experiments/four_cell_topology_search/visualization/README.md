# Four-cell topology-search dashboard

Generate the standalone offline dashboard from the repository root:

```powershell
python modern/experiments/four_cell_topology_search/visualization/generate_dashboard.py
```

The generator refuses modified dataset, manifest, sidecar, representative field,
or geometry evidence. It validates canonical payload hashes, portable manifest
paths, field dimensions and `|B| = hypot(Br, Bz)` before embedding data.

Open `four-cell-topology-search.html` directly in a modern browser. No server,
network connection, package installation, or runtime data fetch is required.

The page follows the corrected v1 semantics: six rank-22 residual roots, zero
identifiable states, and zero performance publications. The v1 search is
superseded development evidence because its gates were not preregistered and
coupling v2 used a deprecated same-z mirror proxy. No archived state-vector or
power data is embedded.
