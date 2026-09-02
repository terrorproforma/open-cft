# Independent axisymmetric P2 FEM reference

## Claim boundary

This workstream is an independent numerical reference for diagnosing L1b
interface discretization. It is not calibration, hardware validation, a
nonlinear magnetic design tool, or evidence that a hypothetical configuration
is build-qualified. Local element/interface maxima are screening-only.

## Reproducible implementation choice

The inspected environment provides Python 3.12, NumPy 2.5.2, and Warp. SciPy,
Gmsh, meshio, and triangle are unavailable and were not installed. The CPU
reference therefore uses:

- a deterministic in-package constrained strip triangulator;
- continuous six-node quadratic Lagrange triangles;
- native deterministic CSR assembly;
- an IC(0)-preconditioned conjugate-gradient solve for small systems and
  vectorized CSR/Jacobi-PCG for the larger adaptive references in binary64
  NumPy;
- no GPU path, because an optional path would not strengthen the independent
  CPU reference without a separately verified sparse backend.

Every accepted linear geometry side is inserted as a complete sequence of
triangle edges. Horizontal region endpoints are global mesh levels. Between
levels, non-crossing inner/outer region tracks partition the domain and each
partition is triangulated independently. Triangle centroids receive strict
region tags and a post-check rejects interface-crossing interiors. This covers
rectangular annuli and the accepted linearly tapered divergent-exit polygons.

## Axisymmetric equation

With azimuthal vector potential \(A_\phi\), define

\[
\psi = r A_\phi,\qquad
B_r=-\frac{1}{r}\partial_z\psi,\qquad
B_z=\frac{1}{r}\partial_r\psi.
\]

For piecewise reluctivity \(\nu=1/\mu\), recoil remanence
\(\mathbf B_r^\mathrm{rem}=(B_{r,r},B_{r,z})\), and free azimuthal current,

\[
-\nabla_{rz}\cdot\left(\frac{\nu}{r}\nabla_{rz}\psi\right)
=J_{\phi,\mathrm{free}}
-\nabla_{rz}\cdot(\nu B_{r,z},-\nu B_{r,r}).
\]

After removing the common \(2\pi\) factor, the weak problem is

\[
\int_\Omega\frac{\nu}{r}\nabla\psi\cdot\nabla v\,dr\,dz
+\int_{\Gamma_R}\frac{\nu}{r}\alpha\psi v\,ds
=\int_\Omega J_\phi v\,dr\,dz
+\int_\Omega\nu(B_{r,z}\partial_rv-B_{r,r}\partial_zv)\,dr\,dz
+\int_{\Gamma_K}K_\phi v\,ds.
\]

The discrete unknown is \(A_\phi\in P_2\), with represented test functions
\(v=rN_i\). All P2 nodes on the axis have \(A_\phi=0\), so
\(\psi=rA_\phi=O(r^2)\). This removes the false finite-energy \(O(r)\) mode
that a naive P2 discretization of \(\psi\) can admit near \(r=0\).

## Interfaces and permanent magnets

The mesh is conforming and \(\psi\) is single-valued across each material
interface. Elementwise integration of the weak form supplies the natural
continuity of

\[
\mathbf n\cdot\left[
\frac{\nu}{r}\nabla\psi-(\nu B_{r,z},-\nu B_{r,r})
\right],
\]

which is the scalar-potential form of continuous normal \(B\) and tangential
\(H\) when free surface current is zero. Bound PM current is represented
either by the recoil-remanence volume action or by conforming line-sheet
actions, never both. The two accepted authorities are numerically equivalent
to roundoff in the verification case.

## Corrected dipole Robin boundary

For the far field

\[
\psi_d=C\frac{r^2}{(r^2+(z-z_c)^2)^{3/2}},
\]

the local condition is \(\partial_n\psi+\alpha\psi=0\), where

\[
\alpha_{r=R}=\frac{3R}{R^2+(z-z_c)^2}-\frac{2}{R},
\qquad
\alpha_{z=z_\pm}=\frac{3|z_\pm-z_c|}{r^2+(z_\pm-z_c)^2}.
\]

The assembled boundary coefficient is \((\nu/r)\alpha\), not merely
\(\alpha\). This condition is still a finite-domain approximation and does not
replace domain-expansion evidence.

## Numerical details

- Seven-point positive Dunavant degree-five triangle quadrature.
- Three-point Gauss-Legendre line quadrature for Robin and sheet terms.
- Global edge identity creates one midpoint DOF per edge.
- The initial grid uses the geometry/material/QoI support scale in the hardware
  region and a deterministic radial exterior size field with geometric factor
  `1.01`, capped at `1.3`. The strip mesher subdivides between physical tracks
  rather than inserting the bore QoI radius as an edge. Geometry and stage
  endpoints remain exact.
- Residual, normal-flux-jump, and bore-window proxy indicators drive
  separate deterministic Dörfler bulk sets with \(\theta=0.5\); their union
  prevents one numerically larger component from suppressing another.
- A parent-level closure promotes coarse neighbors from longest-edge to red
  bisection until every adjacent area-equivalent size ratio is at most `1.3`.
  The topology is then built once and replayed against the same gate.
- A midpoint bisection changes area-equivalent size by \(\sqrt{2}\), so a
  local one-level transition does **not** satisfy the `1.3` gate. The current
  third-level protocol therefore retains verified global red closure; it does
  not claim local `1.3` refinement.
- Essential values are eliminated before deterministic sparse construction.
  Pass one writes preallocated topology-only COO keys and compresses the exact
  CSR pattern; pass two accumulates numerical element/Robin actions directly
  into that pattern. No Python row dictionaries or floating COO copies remain.
- A direct-parent P2 field is prolonged in \(O(N)\) element work: each child
  element uses `element_parent_ids`, maps its six DOFs to parent barycentric
  coordinates, and evaluates that parent's P2 basis. There is no global
  coarse-element location scan. The field is supplied as the PCG initial guess.
  Relative acceptance remains referenced to the RHS norm, not the smaller
  warm-start residual.
- True residuals are recomputed from the assembled CSR operator.
- The homogeneous-data magnetic and source stationarity actions are
  \(\pi x^TKx\) and \(\pi x^Tf\).
- Mesh, solution, run, artifact, and viewer identities use canonical SHA-256.

The interior-edge estimator is

\[
\eta_e^2=h_e\int_e [[\mathbf q\cdot\mathbf n]]^2\,ds,
\qquad
\mathbf q=\frac{\nu}{r}\nabla\psi-(\nu B_{r,z}^{rem},-\nu B_{r,r}^{rem}).
\]

Three-point Gauss quadrature resolves the P2 flux trace. Since both \(h_e\)
and \(ds\) carry length, a constant normal-flux jump contributes
\(h_e^2[[q_n]]^2\); the previous single-length midpoint expression was
dimensionally wrong.

## Acceptance QoIs and observed order

The L1b-compatible bore quantity is the axisymmetric volume average

\[
\overline{B_z}
=\frac{2}{R^2\Delta z}\int_{z_0}^{z_1}\int_0^R B_z r\,dr\,dz
=\frac{2}{R^2\Delta z}\int_{z_0}^{z_1}\psi(R,z)\,dz.
\]

The vertical trace is split at every triangle crossing and integrated with
Gauss quadrature on each P2 piece. It therefore does not require an artificial
vertical mesh constraint; such a constraint creates a geometrically unavoidable
`9.46°` wedge where the divergent bore radius meets the taper. The former wall
sample is retained only as `bore_wall_line_average`. Axis values use weighted
quadratic patch recovery and are diagnostic; finite-radius, finite-length
volume averages are the acceptance quantities.

The L1b comparisons use the same physical fixed QoIs: axis values at each
stage center and axisymmetric bore-volume averages over each stage window.
The FEM axis value is patch recovered while L1b uses structured-grid
interpolation; these are different numerical evaluations of the same QoI, and
the artifact records both methods explicitly.

Manufactured errors are integrated axisymmetric L2 and energy norms. Their
orders use measured local \(h\), while design studies estimate order from
successive differences and the actual bore-region \(h\). Acceptance requires
both successive volume-QoI changes below one percent and positive observed
order. Energy stationarity remains diagnostic and cannot override a failed
true-residual or QoI gate.

The topology hash covers vertices, triangles, all P2 nodes and element DOFs,
sorted edges, midpoint ownership and coordinates, boundary/interface ownership,
parent IDs, refinement ancestry, protected coordinates, shapes, ranges, and
quality replay. For schema `1.3`, acceptance additionally rebuilds the problem
and matrix from bound evidence and recomputes QoIs, true residual, magnetic
and source actions, FEM/L1b arithmetic, changes, actual-\(h\) orders, and
gates. Complete checkpoints bind source-quadrature values as well as all
mesh/topology/configuration/solution arrays. Every binary64/int64 array has an
explicit little-endian dtype, shape, C-order, and SHA-256 descriptor. The L1b
source artifact is file-hash bound and its fixed-QoI values are
reloaded during comparison replay. Rehashed QoI/status claims fail. Existing
schema `1.1` campaign files
remain explicitly legacy integrity-only screening evidence.

Every adaptive level emits hash-sealed JSON metadata plus a compressed NumPy
array sidecar containing the complete replayable state. Sidecar loading is
streamed/hash-checked before lazy array decompression. Metadata is bounded to
`8 MiB`; larger legacy preliminary JSON is guarded at maximum topology before
it may be parsed and migrated. ZIP central-directory records and NPY headers
provide actual dtype, shape, and byte counts before allocation. Those verified
counts—not anchor claims—drive the resource guard and must match the anchors.
The campaign manifest
anchors every checkpoint metadata/payload/sidecar identity and the exact
ordered chain. Each checkpoint binds artifact schema, classification, design,
geometry, magnetics, config, code, base problem, parent checkpoint/mesh, and
final run/mesh identities; unrelated internally valid chains are rejected.
Hydrated bound artifacts are then directly compared with that authority for
schema/classification, design, geometry, magnetics, config, normalized problem,
run, implementation, and acceptance-code identity. Domain chains share the
same authority root and bind each expanded problem independently.

For the three design studies, strict `1.3` closure promotes the first local
marking set to global red refinement. The resource policy now permits at most
`1,500,000` P2 DOFs, but this is only an execution-policy revision. A third
level requires explicit opt-in, exactly one selected design, and at least
`8 GiB` currently free physical RAM. Every level computes a calibrated peak
allocation bound for COO keys, sort/unique masks and temporaries, CSR, solver
vectors, checkpoint parsing/decompression/serialization buffers, and
Python/NumPy overhead. It applies a `1.75` safety factor plus a scale-dependent
reserve capped at `256 MiB`, and rechecks physical RAM before every guarded
mesh, solve, assembly, artifact, checkpoint, replay, and validation phase.
The `1,500,000`-DOF and projected topology caps are independently enforced at
each entrypoint. Failure returns typed `NOT_EVALUATED` and writes no partial
publication; campaign-level failure records a resource-abort checkpoint.
Startup failure below the strict `8 GiB` third-level floor uses the same typed
`ResourceBlockedError(status="NOT_EVALUATED")` as level and cap failures.
Work below both `100,000` P2 DOFs and `64 MiB` serialized state is classified
as readiness-scale: it still enforces all dimension/DOF/topology limits but
does not require the live-RAM gate used by heavy allocation.
Without those conditions the prior
`400,000`-DOF execution bound remains active. Accuracy gates are unchanged.

Robin truncation is a separate fail-closed gate. Padding factors `0.5`, `1.0`,
and `1.5` must be run with phase-matched fixed source/QoI-region \(h\); their
finite positive extents and local \(h\) values are recomputed from their bound
grids, and their successive QoI changes must be below one percent. NaN or
unbound summary evidence fails closed. This evidence is not yet run,
so the artifacts remain screening-only regardless of mesh convergence.

The machine-readable equation ledger is
`spec/fem_reference/equation-ledger-v1.json`.
