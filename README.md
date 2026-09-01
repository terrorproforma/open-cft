# Open CFT

Open CFT preserves and cautiously modernizes research code for a small-scale
cusped-field thruster (CFT/HEMP) design study.

## Repository layout

- [`FYP/`](FYP/) is the historical 2017 MATLAB source snapshot. It is preserved
  as research evidence and is not presented as a corrected or reproducible
  solver.
- [`modern/`](modern/) is the 2026 modernization. It contains typed Python
  models and configuration, FEMM-export readers, a Python implementation of the
  independently checkable cusp loss-cone kernel, an equivalent C++17 kernel,
  and an optional NVIDIA Warp implementation for CPU/CUDA batch execution.

The Python, C++17, and Warp CPU/CUDA implementations of the cusp kernel have
correctness tests covering endpoints, tiny ratios, signed zero, subnormal
values, invalid inputs, and cross-backend parity. This is a verified numerical
kernel, not evidence of GPU acceleration for the complete thruster model.

## Current limitations

Open CFT does **not** yet provide a validated complete plasma solver,
magnetostatic field solver, optimizer, or end-to-end reproduction of the 2017
results. The historical MATLAB model depends on absent optimizer/surrogate
libraries, FEMM, archived run data, and equations not fully specified in the
available publication. Confirmed source defects and publication/snapshot
differences are documented in [`modern/docs/AUDIT.md`](modern/docs/AUDIT.md).

Do not use this repository for hardware design or physical performance
predictions without independent model reconstruction, validation data, and
domain review.

## Verification

The commands below use only dependencies and toolchains already present on the
machine. Warp tests skip cleanly when Warp or a requested device is unavailable.

```powershell
cd modern
python -m pytest
python -m compileall -q src tests

cmake -S . -B build -DCFT_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure

# Focused Warp CPU/CUDA correctness smoke (no performance benchmark)
python -m pytest tests/test_warp_backend.py -k "matches_analytic_reference or preserves_tiny_ratios"
```

For a dependency-free scalar check:

```powershell
cd modern
$env:PYTHONPATH = "$PWD\src"
python -m cft_revival cusp-probability --low-t 0.2 --high-t 1.0
```

See [`modern/README.md`](modern/README.md) for package details and
[`modern/docs/ARCHITECTURE.md`](modern/docs/ARCHITECTURE.md) for the staged
validation plan.

## Publications

- Angus Muffatti and Hideaki Ogawa, “Multi-objective Design Optimisation of a
  Small Scale Cusped Field Thruster for Micro-satellite Platforms,” 31st
  International Symposium on Space Technology and Science, ISTS 2017-b-32
  ([official archive](https://archive.ists.ne.jp/upload_pdf/2017-b-32.pdf)).
- Thomas Fahey, Angus Muffatti, and Hideaki Ogawa, “High Fidelity
  Multi-Objective Design Optimization of a Downscaled Cusped Field Thruster,”
  *Aerospace* 4(4), 55 (2017)
  ([doi:10.3390/aerospace4040055](https://doi.org/10.3390/aerospace4040055)).

The papers provide research context but do not validate every executable
equation in the preserved MATLAB snapshot. Detailed provenance and
cross-source discrepancies are recorded in
[`modern/docs/REFERENCES.md`](modern/docs/REFERENCES.md).

## Citation and license

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). The repository
is released under the [MIT License](LICENSE). Publication copyrights remain
with their respective rights holders.
