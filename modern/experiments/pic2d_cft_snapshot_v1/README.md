# PIC-2D CFT snapshot v1 (development / screening)

Status: `development_screening_not_preregistered`. This experiment exercises
`cft_revival.pic2d` on the qualified P2 `divergent-exit-stack` field at one
documented operating point. It is not preregistered, has no experimental
comparison, and produces no performance claim. Every simplification is listed
in `protocol.json` and repeated in each `summary.json` and in the dashboard.

From `modern/`:

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"
python -m experiments.pic2d_cft_snapshot_v1.run case coarse-w1e5 --max-wall-seconds 1800
python -m experiments.pic2d_cft_snapshot_v1.run case coarse-w5e4 --max-wall-seconds 1800
python -m experiments.pic2d_cft_snapshot_v1.run case fine-w5e4 --max-wall-seconds 2400
python -m experiments.pic2d_cft_snapshot_v1.run case fine-w2.5e4 --max-wall-seconds 2400
python -m experiments.pic2d_cft_snapshot_v1.run summarize
python visualization/generate_pic2d_cft_snapshot.py
```

Each case stops at `target_steps`, at its wall-clock budget, or when the
fail-closed runtime stability gate (`ω_pe Δt ≤ 0.2` on the peak node density,
one-cell Courant crossing) refuses to continue; the reason is recorded in
`summary.json`. Outputs per case: `summary.json`, `series.npz`, `maps.npz`
(time-averaged over the last complete or half-complete averaging window),
`checkpoint-final.{json,npz}`, all with `.sha256.json` sidecars. `summarize`
writes `results/manifest.json` with the case hashes and a between-case
convergence table (relative spread of window-averaged summaries).

Operating point (v1): anode 300 V, exit plane 0 V, uniform Xe background
5e20 m⁻³ at 300 K (≈ 0.075 mg/s effusive equivalent), 0.1 A of 2 eV electrons
injected at the exit plane, 5e16 m⁻³ / 5 eV quasi-neutral seed plasma,
Δt = 2 ps, coarse grid 30×240 (100 µm) and fine grid 60×480 (50 µm), two macro
weights per grid.
