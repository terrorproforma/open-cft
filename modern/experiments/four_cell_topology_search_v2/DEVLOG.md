# Four-cell topology search v2 development log

## 2026-09-02 preregistration phase

- Audited the accepted coupling v3 baseline at
  `f80a360fd740a30017cdac1874cedbfa2806874a`; the current branch was exactly at
  that commit and accepted coupling, fields, geometry, magnetics, optimization,
  plasma, plasma_network, and specification paths had no local changes.
- Added an experiment-local immutable protocol for 128 deterministic rounded
  shifted-Halton candidates and three independent Warp maps per candidate:
  56x168 primary, 40x120 lower-resolution evidence, and 64x192 on a 1.25x
  enlarged domain.
- Declared exact four-cusp count, geometry-slot registration, three-cell
  endpoint exclusion, 0.8 mm cross-map shift, finite-domain boundary
  comparison, map residual/source/flux gates, three flux quantiles, reject-on-
  saddle connectivity, complete covered field uncertainty, 100 eV isotropic
  screening electrons, and rho_e/L_B <= 0.1.
- Implemented a direct v3 L1a artifact adapter that binds full psi/B maps,
  source, geometry, material, mesh, domain, artifact, protocol, code closure,
  backend, and runtime identities. No v2 proxy or archived experiment is
  imported.
- Implemented full four-dimensional probability-box propagation into
  geometry-derived N=4 `plasma_network` inputs at three fixed operating points
  with nine deterministic starts. The full-rank publication policy suppresses
  all state/power/performance objects.
- Added exclusive deterministic execution locking, detached-clean enforcement,
  conservative transitive Git-blob closure, tolerance-based GPU replay,
  representative exact artifacts, strict JSON hashes, sidecars, manifests, and
  pre/post-run lifecycle validation.
- Preflight: eight experiment-local tests passed, including all 128 strict
  geometries and all 384 role-domain source-support contracts, source pairing,
  synthetic connected contours, exact-saddle rejection, probability-box
  completeness, accepted-blob identity, prohibited imports, and the pre-run
  lifecycle. The preflight fixed the protocol's stack-start lower bound and
  current-sheet smear before preregistration; no candidate field was solved.
