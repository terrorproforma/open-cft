# Preregistered L1a multi-fidelity field surrogate v3

V3 runs on the isolated `exp/l1a-field-surrogate-v3` branch. Its protocol and
result commits must be direct parent/child regardless of activity on the main
development branch.

The experiment freezes 240 geometry-v1.1-valid rows from a fresh 512-row input
pool: 128 candidate, 16 method, 48 calibration, and 48 single-use assessment
rows. Calibration and assessment each contain 16 interpolation, boundary, and
OOD cases. Fine labels are solved only when their stage becomes accessible.

Both 41x73 and 81x145 fidelities conservatively integrate the same physical
0.8 mm source bands. Scalar discrepancy models predict log high/coarse ratios
while using observed coarse values directly. Field models use input-only
landmark alignment and cylindrical weighted, coarse-normalized residual POD.

The high target is the accepted L1a numerical discretization, not physical
truth. No material, plasma, thermal, structural, propulsion, build, or
hardware accuracy is claimed.
