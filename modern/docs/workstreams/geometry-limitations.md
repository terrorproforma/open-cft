# Geometry workstream limitations

- The three configurations are hypothetical screening geometries. They are
  neither optimized nor evidence that hardware is buildable.
- “Compact high-gradient” names a short-pitch design intent. No field solution
  in this workstream establishes a gradient.
- TWT and CFT devices can share a periodic permanent-magnet/pole mechanical
  pattern. TWT RF slow-wave amplification physics is not CFT plasma propulsion
  physics; no TWT RF analogy is used as a performance model.
- The historical baseline exactly maps only legacy values visible in
  `FYP/FEMMrun.m`: 2 mm bore radius, 21 mm envelope, and magnet spans 0–4,
  5–15, and 16–20 mm. Commented radial examples, clearance, pole fill, and the
  13 mm yoke radius are assumptions. Its magnet inner radius is now 3.40 mm:
  a 0.40 mm nominal gap retaining more than 0.25 mm after radial tolerances.
- The old optimization admitted each radial variable independently over
  2–50 mm and did not encode radial ordering in `FYP/CFTOpt.m`. Those ranges
  are not interpreted as valid hardware dimensions.
- Material definitions are magnetic/geometry placeholders. SmCo, BN, iron,
  aluminium, and copper densities or permeability values are not vendor-grade
  qualification data.
- The linear soft-iron `mu_r` is unsuitable for saturation, hysteresis, loss,
  or thermal design. Use an accepted traceable B-H material contract before
  material-aware field conclusions.
- Thermal clearance is a scalar geometry allowance, not a thermal expansion,
  heat-transfer, stress, fracture, outgassing, erosion, or insulation model.
- Cathode and neutralizer placement is metadata because the expected hardware
  is external and not necessarily axisymmetric.
- The L1a current-equivalent preview omits recoil permeability contrast, pole
  saturation, exact current sheets, and demagnetization. It is not a permanent
  magnet material solve and is structurally non-authoritative.
- The accepted magnetics v1 region contract is rectangular. Exact tapered
  ownership is present in the solver-neutral topology, but tapered regions are
  not projected into that rectangular handoff.
- Non-rectangular permanent magnets are rejected outright in geometry v1.1.
  L1b must provide exact polygonal/frustum PM material and magnetization
  support before tapered PM studies are accepted.
- A nominal alternating sequence suggests inter-stage cusps; actual null count
  and topology must come from the accepted field solver and topology audit.
- SVG is a deterministic technical cross-section, not a CAD manufacturing
  drawing. No DXF is emitted because a rigorously dimensioned/tested DXF
  implementation was not needed for this workstream.
- No propulsion performance prediction is made.
- SHA-256 sidecars are integrity anchors, not signatures or proof of publisher
  authenticity. Generator/claim allowlists prevent semantic substitution
  within accepted bundles but still assume a trusted loader and distribution
  channel.
