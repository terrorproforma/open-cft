# FEM reference learning ledger

## Constraints carried through

- Work only in the five FEM-reference path families.
- Treat geometry, magnetics, fields, material-fields, shared code, FYP, and Git
  state as read-only concurrent work.
- Install nothing and retain a required CPU implementation.
- Publish an independent numerical reference, never hardware validation.

The requested `learning-scratchpad-loop` and `devlog-loop` skill definitions
were not present in the available skill set or repository. Their material
effect is implemented directly here and in the continuously updated
`fem-reference-devlog.md`: decisions, failed assumptions, verification
evidence, and remaining gaps are recorded during the work rather than after it.

## What changed the numerical design

1. A naive P2 space directly for \(\psi\), constrained only by \(\psi=0\) on
   the axis, admits local \(O(r)\) modes. Their
   \(\int (1/r)|\nabla\psi|^2\) energy is singular. Using P2 for \(A_\phi\)
   and representing \(\psi=rA_\phi\) enforces the required \(O(r^2)\)
   regularity without multipoint derivative constraints.
2. No safe general-purpose mesher is installed. A fake raster-to-triangle
   conversion would not diagnose oblique interfaces. The implemented strip
   arrangement preserves all accepted rectangle/taper sides as constraints
   and rejects crossing tracks or interface-crossing triangle interiors.
3. Global uniform refinement alone does not necessarily resolve a small bore
   inside a large padded domain. Geometry breakpoints guarantee interface
   conformity, but field-gradient convergence still needs enough interior
   radial lines. This is why the resource-bounded design campaign can fail the
   one-percent gate even when residual and energy identities are excellent.
4. P2 gradients are discontinuous at vertices. Axis values at a mesh-level
   junction must average all adjacent axis-edge traces; selecting the first
   containing triangle creates an avoidable orientation-dependent QoI.
5. Recoil remanence and equivalent bound-current sheets become the same
   discrete action only when sheet lines are complete conforming edges and the
   handoff's recoil-permeability scaling is retained.
6. Nonzero manufactured Dirichlet data carries boundary work. The simple
   \(x^TKx=x^Tf\) action check is therefore an acceptance diagnostic for the
   homogeneous physical solves, not those manufactured cases.
7. The old `bore-average` was not a volume average: it integrated only
   \(B_z(R,z)\). The corrected quantity integrates the piecewise P2
   \(\psi(R,z)\) trace and uses
   \(\int_0^R B_zr\,dr=\psi(R,z)\). The old quantity remains explicitly named
   `bore_wall_line_average`.
8. Making the bore radius a forced edge is harmful in the divergent geometry:
   it and the taper enclose a `9.46°` wedge. Piecewise trace integration keeps
   the QoI coordinate exact without contaminating physical mesh quality.
9. Binary midpoint refinement changes area-equivalent \(h\) by \(\sqrt2\).
   Therefore a strict adjacent growth limit of `1.3` can force refinement
   propagation beyond the Dörfler set. In the historical campaign, `764`
   marked cells propagated to every one of the `33,224` parents. The completed
   meshes satisfy the limit, while levels exceeding the explicit DOF bound are
   not claimed.
10. Applying Dörfler marking only to a summed indicator does not guarantee
    bulk capture for each differently scaled component. The deterministic
    union of residual, flux-jump, and QoI component bulk sets makes all three
    `theta=0.5` guarantees explicit before conformity/gradation closure.
11. The edge jump term needs two powers of length: one from \(h_e\), one from
    \(ds\). Midpoint sampling with only `length * jump^2` is dimensionally
    incomplete and changes relative marking across graded edges.
12. Python row dictionaries hid sparse-assembly memory in object overhead.
    A topology-key pass followed by direct CSR accumulation cut the measured
    probe peak by `57.2%`, even though binary-search accumulation costs more
    time on the small case.
13. An artifact payload hash proves internal integrity only. Acceptance needs
    an external authority chain plus recomputation from mesh, solution,
    controls, code identity, QoI windows, and prior-level evidence.
14. Warm-start interpolation must use refinement ancestry, not repeated point
    location. Child `element_parent_ids` make all six local P2 evaluations
    bounded work, and four successive refinements show stable time per child
    element.
15. A startup-only RAM check does not protect assembly. The allocation model
    must include COO sort/unique workspace, masks, CSR, Krylov vectors, and
    allocator overhead, then be checked again at the allocation boundary. The
    4,959-DOF probe measured `2,914,469` tracemalloc bytes and a `675,840`-byte
    retained RSS increase; the calibrated bound was `478,307,723` bytes.
16. Level summaries are claims, not authority. Schema 1.3 accepts only
    manifest-anchored checkpoints containing complete replayable bound
    artifacts. Domain evidence rejects nonfinite, nonnested, or
    phase-mismatched grids before evaluating a change gate.
17. A DOF cap enforced only by the campaign is not a resource boundary.
    Initial mesh, refinement, assembly, solve, artifact, checkpoint, replay,
    and validation entrypoints must independently reject non-integer
    dimensions, more than `1,500,000` P2 DOFs, and impossible projected
    topology before allocating.
18. A valid checkpoint can still be unrelated evidence. Authority requires
    binding the exact ordered chain to schema/classification, design,
    geometry, magnetics, config, code, base problem, parent, and final
    run/mesh identities. Domain chains share the same authority root.
19. Keeping complete arrays inside checkpoint JSON multiplies peak memory.
    Small JSON metadata plus hash-addressed compressed NumPy sidecars permits
    streaming file hashing and lazy per-array decompression while preserving
    little-endian dtype, shape, C-order, and byte hashes.
20. Manifest dimensions are claims and cannot size a safety guard. ZIP central
    directory records and bounded NPY headers reveal actual shapes/dtypes and
    uncompressed bytes before allocation; only those verified values may drive
    the guard. Anchor dimensions must match those headers exactly before
    decompression.
21. Internal artifact validity is necessary but insufficient. Each loaded
    bound artifact must directly match the enclosing chain's design, geometry,
    magnetics, config, normalized problem, run, implementation, and acceptance
    code. The same rule applies to domain-study chains.
22. Preliminary recovery files must already use bounded metadata and binary
    sidecars. Otherwise finalization recreates the giant-JSON memory defect.
    Legacy files above the metadata bound require a maximum-topology resource
    gate before streaming parse and migration.
23. Sub-percent successive changes do not imply asymptotic convergence.
    Historical stage 3 and compact stages 1/5 had negative observed order even
    though both changes were far below one percent. Keeping the positive-order
    gate prevented false qualification.
24. Domain truncation was not the limiting gate in this campaign. All three
    designs passed phase-matched padding `0.5/1.0/1.5`; the largest first and
    second expansion changes were `2.665e-3` and `3.289e-4`, both below one
    percent.
25. The divergent design was the only case with positive order for every bore
    QoI (`1.348–1.395`) and therefore the only numerical P2 qualification.
    Historical and compact remain screening evidence, not failed hardware.

## Verified observations

- Smooth axis-regular manufactured sampled errors converge at approximately
  third order in integrated axisymmetric L2 and second order in energy, as
  expected for P2.
- Both an aligned and an oblique \(11{:}1\) reluctivity jump reproduce an
  exactly representable continuous-flux solution to near algebraic tolerance.
- The analytic dipole verifies the sign and the missing \(-2/r\) radial term
  in the Robin coefficient.
- Recoil and equivalent-sheet PM authorities agree to roundoff, polarity
  reversal negates the full solution, and physical solve energy/source actions
  agree to roundoff.
- A uniform-medium current-band case agrees with the independent L1a solver at
  fixed axis points to below 0.3% at the checked resolutions.

## Remaining gaps

- No nonlinear B-H, saturation, hysteresis, irreversible demagnetization, or
  plasma response.
- The local dipole Robin condition is not an exact nonlocal exterior map.
  Phase-matched `0.5/1.0/1.5` padding evidence is now required and not yet run.
- Strict `1.3` adjacent area-size growth converts local binary bisection into
  global red refinement for these connected meshes because one midpoint split
  has discrete ratio \(\sqrt2\). The policy ceiling is now `1,500,000` DOFs,
  but the live host failed the `8 GiB` free-RAM preflight, so two-change/order
  convergence remains unresolved.
- No verified GPU sparse assembly/solve path was added.
- Interface field maxima remain mesh-side traces and are screening-only.
