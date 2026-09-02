# Learning scratchpad

## Predeclared decisions

- Use the accepted shifted-Halton sampler and preserve its honest name.
- Use geometry v1.1 generators and the explicitly non-authoritative L1a
  thin-volume current-equivalent preview.
- Scale preview ampere-turns as an experiment variable without reinterpreting
  the scale as a magnet grade or measured remanence.
- Keep the domain and solver configuration fixed across all cases.
- Treat geometry, source, and run configuration as separate hash anchors.
- Keep failed evaluations typed and outside objective ranking; do not invent
  penalties.
- Use local exact constrained dominance because accepted optimization fidelity
  labels are not semantically compatible with a field-only L1a solve.
- Separate uncontrolled wall-time diagnostics from deterministic scientific
  payloads.

## Claim guardrails

- Sampled axis nulls are not continuous critical-point proofs.
- Boundary truncation, source transfer, residual, topology confidence, and
  manufacturability margins are explicit screening gates.
- This work produces no thrust, efficiency, Isp, plasma, thermal, structural,
  procurement, build, or hardware-valid claim.

## Findings

- All 96 bounded samples passed strict geometry and manufacturability after
  representing a few requested pitch/taper values at adjacent binary64 values
  needed by geometry v1.1's ULP-continuity contract; both requested and
  represented values are recorded.
- The fixed padded domain kept boundary/peak ratios between about 0.00179 and
  0.0117, below the predeclared 0.05 screening gate.
- Source representation error remained about 0.00213–0.0175 and topology
  confidence about 0.912–0.958.
- Six distributed CPU/CUDA comparisons passed by orders of magnitude; maximum
  component scale-relative differences remained at binary64 rounding scale.
- Exact four-objective constrained ranking produced 25 nondominated cases.
- Early dry runs exposed one experiment-local strict-zip indexing defect and
  three sub-minimum divergent lengths; both were corrected before the sealed
  successful run. Failed dry-run outputs were overwritten and are not evidence.
