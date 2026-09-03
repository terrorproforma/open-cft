# Learning scratchpad — mdo_l0_campaign_v1

- [self] Probe the candidate evaluation chain on the real code before writing
  the protocol: the corrected Kornfeld solver looked like the natural cusp ->
  performance bridge and turned out never to close for nonzero cusp
  probabilities (and to cost 3 s per solve). Twenty minutes of probing saved
  a protocol built on an unusable chain.
- [self] A design-independent uncertain input that enters every objective
  multiplicatively preserves Pareto dominance; the robust optimisation only
  differs from the nominal one through the feasible set. State such
  degeneracies as predeclared expectations and check them, do not hide them.
- [self] "Front invariance" checks must be restricted to the common feasible
  set; otherwise the moving feasibility boundary masquerades as a front
  change (first shakedown: Jaccard 0.02 at p = 0 purely from feasibility).
- [tool] BoTorch 0.18.1: `optimize_acqf` is lazily exported from
  `botorch.optim`, `optimize_acqf_list` only from `botorch.optim.optimize`.
- [tool] qLogNEHVI caches Cholesky base samples at construction; assigning
  `.sampler` afterwards raises a shape error in `_update_base_samples`. Pass
  the sampler to the constructor.
- [tool] Under a saturated GPU (another agent's PIC run at ~92 % util), tiny
  GP fits and acquisition optimisation on cuda:0 are 20-40x slower than cpu.
  Measure before declaring the device in a protocol.
- [tool] pymoo `eliminate_duplicates=True` silently reduces the number of
  evaluated offspring; exact budgets need `False`.
- [tool] The repository's standard test interpreter has numpy but no scipy;
  a scipy-Sobol baseline would have made every campaign test error there.
  Baselines and shared initial designs use the stdlib RNG (LHS) instead.
- [self] The 4-D hypervolume slicing recursion is fast enough only with the
  nondominated pre-filter and a vectorised 2-D sweep; the pure-Python version
  was O(n^4) on the dense front.
- [self] Exact front-set equality is fragile: L0 recomputes anode power as
  `Ua * I_beam / beam_fraction`, one ulp off `Ua * Ia` depending on theta,
  so a design tied in power under one prior lost dominance under another.
  Compare fronts with roundoff-aware dominance and report both.
- [self] A worst-case constraint over a finite QMC sample enforces the worst
  sampled case, not the worst case of the prior's support (max sampled S was
  0.704, not 1). Declare the sampled support explicitly, or add the support
  vertex to the sample.
- [self] Canonical typed JSON round trips break re-hashing: tuples become
  `__cft_type__` tagged objects that `canonical_value` refuses on re-parse.
  Records that must be re-hashed after loading must be written from plain
  lists/dicts only.
- [tool] PowerShell `Start-Process -RedirectStandardError` delivers the log
  with multi-minute lag; read the runtime's checkpoints or the terminal
  output file instead of trusting the stderr tail.
- [tool] Headless Edge clamps `--window-size` widths below ~500 px; a 390 px
  iframe host page gives a true narrow viewport (as the v4 dashboard notes).
