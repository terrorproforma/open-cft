# First GPU L0 Results

## Result identity

This is the first checked execution of the conservation-based
`L0-conservation-reduced-performance` model on the repository's RTX 5090 Warp
CUDA path. It is a numerical model result, not measured-thruster validation,
plasma-solver output, a fitted 2020 case, or a reproduction of the flawed 2017
objective outputs.

Run on 1 September 2026:

```powershell
cd modern
$env:PYTHONPATH = "$PWD\src"
$out = Join-Path $env:TEMP "cft-l0-first-cuda.json"
python -m cft_revival l0-sweep config/l0-deterministic-sweep.json `
  --device cuda:0 --output $out
```

The checked config uses 8,192 deterministic prime-base radical-inverse points
with seed `20260901`. Inputs span:

- discharge voltage: 150–500 V;
- xenon mass flow: `2e-7`–`2e-6 kg/s`;
- ionized number fraction: 0.65–0.98;
- Xe2+ share of ions: 0–0.15;
- beam-current/anode-current fraction: 0.75–0.98;
- axial/ion-momentum fraction: 0.75–0.98;
- cathode input: 5–25 W;
- assumed PPU efficiency used to construct an admissible boundary: 0.82–0.95.

All bounds are physically plausible but explicitly hypothetical. They are not
calibration or uncertainty intervals.

## Hardware and runtime

- OS: Windows 11 build 26200, 64-bit.
- Python: 3.12.10.
- Warp: 1.14.0; bundled CUDA Toolkit 12.9; driver CUDA 13.2.
- Device: NVIDIA GeForce RTX 5090, 32,607 MiB, `sm_120`.
- NVIDIA driver: 595.97.
- Observed GPU utilization immediately before/after: 7% / 4%;
  temperature: 50 / 51 °C.
- Warp CUDA end-to-end evaluation: 0.634302 s, 12,914.99 points/s.
- Separate Python reference construction: 0.141245 s.

The timing is **uncontrolled**. It includes Python preprocessing, allocations,
host/device transfers, synchronization, and construction of every Python result
record. It was a single diagnostic run with no controlled clocks, repeated
trials, or kernel-only interval. The lower Python time is also not a benchmark:
no speedup or slowdown claim is made.

## Numerical outputs

Across all 8,192 CUDA results:

- axial thrust: `0.00188384225`–`0.0513183291 N`;
- specific impulse: `799.268670`–`2726.81617 s`;
- beam current: `0.100578251`–`1.64624848 A`;
- anode input: `21.5739213`–`910.314451 W`;
- beam kinetic power: `17.2688569`–`786.015592 W`;
- PPU-input-to-beam efficiency: `0.379828910`–`0.908991983`.

Selected deterministic cases:

- index 0: `485.047280 V`, `9.82608045e-7 kg/s`, ionized fraction
  `0.843437893`, Xe2+ fraction `0.0692342479`; axial thrust
  `0.0186785330 N`, Isp `1938.39273 s`, beam power `319.667811 W`;
- index 4096: `485.090005 V`, `1.49106712e-6 kg/s`, ionized fraction
  `0.895308613`, Xe2+ fraction `0.0843043805`; axial thrust
  `0.0310332192 N`, Isp `2122.31068 s`, beam power `520.707438 W`;
- index 8191: `310.068642 V`, `1.40578541e-6 kg/s`, ionized fraction
  `0.878519226`, Xe2+ fraction `0.0930025505`; axial thrust
  `0.0226805104 N`, Isp `1645.17885 s`, beam power `311.206846 W`.

## Conservation and parity

The full CUDA batch was compared with the dependency-free Python reference,
not only a subset. All 26 published numeric fields were checked and all 8,192
records passed the documented binary64 tolerances.

- maximum physical-output differences: thrust `1.38778e-17 N`,
  axial thrust `6.93889e-18 N`; all reported speeds, rates, currents, powers,
  and efficiencies otherwise had zero maximum difference;
- maximum CUDA conservation residuals: `2048 particles/s`,
  `3.41407e-22 kg/s`, `4.44089e-16 A`, and `2.27374e-13 W`;
- corresponding maximum relative residuals: `2.36837e-16`,
  `3.24198e-16`, `4.10551e-16`, and `4.43207e-16`;
- maximum CUDA-versus-reference residual differences reflect binary64
  reduction order: `1024 particles/s`, `4.21228e-22 kg/s`,
  `2.22045e-16 A`, and `0 W`;
- failed or rejected points: 0.

## Limits

L0 supplies charge-state mix, beam-current fraction, divergence, cathode power,
and PPU behavior as external inputs. It omits ionization/excitation, wall and
thermal losses, cathode-plasma coupling, magnetic topology, facility effects,
erosion, uncertainty calibration, and experimental comparison. Numerical
closure and CPU/CUDA parity establish implementation consistency only; they do
not establish physical predictive accuracy.

These outputs are not wired into the optimization objectives. In particular,
the campaign's historical `total_efficiency` objective is not silently equated
with any of L0's three explicitly bounded efficiency definitions.
