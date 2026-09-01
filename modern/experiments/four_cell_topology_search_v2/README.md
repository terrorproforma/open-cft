# Four-cell topology search v2

This directory contains the preregistered successor to the archived v1 proxy
search. It is bound to coupling v3 commit
`f80a360fd740a30017cdac1874cedbfa2806874a` and uses only accepted geometry
v1.1, L1a fields, connected constant-psi coupling v3, and `plasma_network`.

`protocol.json` is the immutable before-run declaration. It fixes 128
deterministic candidates, three independent maps per candidate, four-cusp
stability and geometry-registration gates, all-quantile contour acceptance,
covered field uncertainty, electron adiabaticity, interval propagation, the
failure taxonomy, ranking, replay, and publication policy.

The run must start from a clean detached preregistration commit:

```powershell
$env:PYTHONPATH="$PWD\modern\src;$PWD\modern"
python -m experiments.four_cell_topology_search_v2.run
```

The executable refuses a branch checkout, a dirty worktree, or an existing
results path. It creates the result lock exclusively before the first solve.
Zero accepted candidates is a valid immutable result and must not trigger a
patch or rerun.

Plasma residual roots are not performance. The fixed full-rank policy publishes
no state, power, or performance object for the structurally rank-deficient N=4
network. Only residual, conservation, rank, and nullity diagnostics are retained.
