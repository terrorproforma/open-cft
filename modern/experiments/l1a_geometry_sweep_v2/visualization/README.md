# Preregistered L1a sweep-v2 dashboard

`l1a-geometry-sweep-v2.html` is a deterministic self-contained offline viewer
for the committed preregistered sweep-v2 evidence.

From `modern/`:

```powershell
python experiments/l1a_geometry_sweep_v2/visualization/generate_dashboard.py
```

Generation fails closed unless all of the following remain exact:

- preregistration commit `092f5fae692ee7d6711e0c7e1c94dac6a345f37c`;
- its direct-child results commit
  `f30cb42ec4a8633bf634a3d32ffa5b11f66be97a`;
- committed authored times and execution-lock ordering;
- protocol, execution lock, raw results, summary and manifest file/payload
  SHA-256 identities;
- every manifest-listed deterministic file, sidecar, payload and representative
  binding;
- tolerance-based nondominated front, five roles/four unique representative
  coalescence, and all seven acceptance gates.

Focused checks:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m pytest -q tests/experiments/l1a_geometry_sweep_v2_visualization
python -m compileall -q experiments/l1a_geometry_sweep_v2/visualization tests/experiments/l1a_geometry_sweep_v2_visualization
```

The dashboard distinguishes hash-exact identities from tolerance-based CUDA
floating replay. It remains L1a field-only screening and provides no
material-aware permanent-magnet, plasma, propulsion or hardware validity.
