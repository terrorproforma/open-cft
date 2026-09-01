# Axisymmetric Field Learning Report

## Evidence acquired

- Warp 1.14.0 is installed with a working CPU device and RTX 5090 `cuda:0`.
- Warp's bundled FEM diffusion example can assemble scalar weighted weak forms;
  its magnetostatics example is a 3D H(curl) curl-curl model. Neither example
  verifies the axis-singular `1/r` scalar form or regular axis trace needed
  here.
- The flux variable `psi=r A_phi` exposes both a positive weighted operator
  and exact divergence-free field representation:
  `-div(r^-1 grad psi)=mu J_phi`.
- The regular-axis expansion is a numerical contract, not merely a boundary
  value. `psi(0,z)=0` alone does not define `B_z(0,z)`; the `r²` coefficient
  must be recovered.

## Decisions and reasons

- Use a conservative matrix-free structured-grid FDM for L1a. It is smaller
  and more auditable than introducing an unverified weighted FEM path, and its
  second-order behaviour is directly proven by a manufactured solution.
- Keep the Python implementation dependency-free and canonical. Warp performs
  the same operator, vector updates, Jacobi preconditioning, and reductions
  in real CPU/CUDA kernels; Python independently recomputes the final residual.
- Represent sources as signed ampere-turns smeared over annular/axial
  cross-sections. This enables geometry comparison now without inventing a
  permanent-magnet constitutive mapping.
- Treat homogeneous outer `psi` as a finite-box truncation only. Do not call it
  an open boundary.
- Require source bands to remain in the interior dual-cell support and span at
  least two spacings. Exact overlap averaging preserves requested current
  without disguising unresolved or boundary-clipped geometry.
- Publish downsampled maps but retain full one-dimensional profiles, exact
  input/provenance, source-transfer error, closed schemas, canonical payload
  hashes, and filename-bound file hashes.

## Numerical guardrails learned

- PCG recurrence residuals can differ from a reapplied-operator residual;
  publication status must use the latter. A false recurrence crossing should
  restart from the true residual rather than discard remaining iterations.
- The Jacobi-preconditioned residual may rise initially while still converging;
  monotonic residual history is not a valid requirement for PCG.
- Recovering both field components from one scalar flux makes centred mixed
  derivatives commute. The resulting near-roundoff metric is a
  flux-reconstruction identity check, not independent divergence/PDE
  validation.
- A global `|B|min` on a homogeneous finite box is commonly a corner boundary
  zero. Cusp/null reporting must search and label interior topology separately.
- Zero and near-zero maps are degenerate topologies, not fields containing an
  arbitrary isolated null at every sample. Sign changes, isolated samples,
  plateaus, and boundary minima need distinct machine-readable states.
- Embedded file hashes are recursively ambiguous. Canonical payload integrity
  excludes only the integrity object; raw file integrity belongs in a
  filename-bound sidecar and is anchored by the non-recursive manifest payload.
- A raw `width < 2*spacing` comparison is not a geometry policy: decimal-looking
  endpoints can subtract one rounding step low. Summed half-ULP uncertainty for
  both endpoints and the target admits the exact intended boundary while still
  rejecting the next representably narrower endpoint.
- Python JSON integers are unbounded but binary64 artifact scalars are not.
  Every numeric conversion boundary must wrap `float()` overflow and decoder
  digit-limit failures in the same typed artifact-validation error.
- Backend agreement proves implementation parity only. It does not test the
  equivalent-current approximation, material model, boundary truncation, or
  agreement with hardware.

## Open work before L1

- Source and verify the mapping from real magnet remanence/magnetization to
  equivalent surface currents.
- Add discontinuous material interfaces and nonlinear iron B-H data.
- Verify infinite/open boundaries and domain-size convergence.
- Run geometry-aware mesh convergence for centreline, wall, cusp, and peak
  field quantities.
- Compare complete profiles with FEMM and preferably an independent MFEM
  implementation.
- Compare to measured maps with uncertainty before predictive claims.
