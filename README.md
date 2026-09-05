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
  an optional NVIDIA Warp implementation for CPU/CUDA batch execution, an
  SI-explicit Xe/Xe+/Xe2+ L0 conservation model, and a dependency-free
  multi-fidelity optimization campaign foundation.

Accepted 2026 foundations also include the linear-vacuum **L1a** axisymmetric
FDM field solver and checked result bundle; corrected, source-ledgered global
plasma numerics; magnetic material/source contracts; field-to-plasma coupling;
prescribed-field hybrid and reduced electrostatic PIC kernels; surrogate and
active-learning infrastructure; and validation/evidence contracts. These are
independently verified foundations, not a single predictive CFT model.

The browser-tested, self-contained viewers are
[`first-results.html`](modern/visualization/first-results.html),
[`geometry-designs.html`](modern/visualization/geometry-designs.html), and
[`axisymmetric-results.html`](modern/visualization/axisymmetric-results.html).
The last displays the three accepted L1a artifacts and their limitations.
Where the whole programme stands is
[`roadmap-status.html`](modern/visualization/roadmap-status.html): the
eight-rung evidence ladder over every workstream (specified -> code -> tests ->
real inputs -> preregistered run -> accepted -> in the paper -> externally
validated, with a merged flag and a citation behind every rung), built offline
from the roadmap canvas by
[`modern/visualization/roadmap-status/`](modern/visualization/roadmap-status/README.md).

The Python, C++17, and Warp CPU/CUDA implementations of the cusp kernel have
correctness tests covering endpoints, tiny ratios, signed zero, subnormal
values, invalid inputs, and cross-backend parity. This is a verified numerical
kernel, not evidence of GPU acceleration for the complete thruster model.

The L0 model now has checked point/sweep JSON workflows and a first 8,192-point
RTX 5090 result with full CPU-reference parity. See
[`modern/docs/FIRST_RESULTS.md`](modern/docs/FIRST_RESULTS.md). Its charge-state,
beam, divergence, cathode, and PPU quantities are external hypothetical inputs,
so numerical closure is not measured-performance accuracy. The optimization
package supplies immutable records, constrained Pareto logic, async replay,
budgets/retries, guardrails, and shifted-Halton designs; it does not silently
turn L0 outputs into validated campaign objectives.

## Current limitations

Open CFT does **not** yet provide a validated material-aware production field
solver, predictive L2/L3 CFT model, fitted surrogate optimizer, or end-to-end
reproduction of the 2017 results. L1a is finite-box, constant-permeability,
equivalent-current FDM—not FEM. The global plasma, hybrid, and PIC foundations
have numerical verification but no accepted predictive or experimental-validity
claim. The held-out surrogate quality benchmark failed its acceptance criteria.
The historical MATLAB model depends on absent optimizer/surrogate
libraries, FEMM, archived run data, and equations not fully specified in the
available publication. Confirmed source defects and publication/snapshot
differences are documented in [`modern/docs/AUDIT.md`](modern/docs/AUDIT.md).

Do not use this repository for hardware design or physical performance
predictions without independent model reconstruction, validation data, and
domain review.

## Verification

Run these commands from the repository root. They use only dependencies and
toolchains already present on the machine. Warp tests skip cleanly when Warp or
a requested device is unavailable.

```powershell
cd modern
$env:PYTHONPATH = "$PWD\src"
python -m pytest
python -m compileall -q src tests
python -m cft_revival l0-evaluate config/l0-representative-point.json
python -m cft_revival validate-campaign-spec spec/optimization/campaign-v1.json
python -m cft_revival generate-initial-design spec/optimization/campaign-v1.json --count 32 --seed 7
python -m cft_revival validate-axisymmetric-results examples/axisymmetric/results/manifest-l1a-v1.json

cmake -S . -B build -DCFT_BUILD_PYTHON=OFF -DBUILD_TESTING=ON
cmake --build build
ctest --test-dir build --output-on-failure

# Focused Warp CPU/CUDA correctness smoke (no performance benchmark)
python -m pytest tests/test_warp_backend.py -k "matches_analytic_reference or preserves_tiny_ratios"
```

No package installation is required for those core CLI commands. The
`PYTHONPATH` assignment is required because the repository uses a `src/`
layout. On POSIX shells, use `export PYTHONPATH="$PWD/src"` after
`cd modern`.

If the optional Warp package and CUDA device are already available, the L0
GPU command is:

```powershell
cd modern
$env:PYTHONPATH = "$PWD\src"
python -m cft_revival l0-sweep config/l0-deterministic-sweep.json `
  --device cuda:0 --output (Join-Path $env:TEMP "cft-l0-sweep.json")
```

See [`modern/README.md`](modern/README.md) for package details and
[`modern/docs/ARCHITECTURE.md`](modern/docs/ARCHITECTURE.md) for the staged
validation plan. Repository milestones are tracked in [`ROADMAP.md`](ROADMAP.md).
The evidence-gated manuscript source and deterministic checks are under
[`paper/`](paper/); L1, L2, and L3 paper result gates remain closed.

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
