# CFT wall-cusp validation v1

This directory contains the first preregistered, disjoint held-out numerical
validation of the frozen coupling-v4 wall-cusp criterion and schema 4.1. It is
bound to coupling commit `f10d8213117fbafd8c2b69bdc103b6ef7b5d6d8c`.

The 56-case topology characterization remains development evidence only. This
validation uses 24 new offset-geometry cases spanning three stage counts, two
pitches, two chamber radii, and both first polarities. Every case requires
accepted primary, refined, and enlarged L1a maps, an all-case GPU replay gate,
complete v4 map fingerprints, wall-connected two-direction paths, same-line
extrema, uncertainty bounds, and three explicit electron energy/pitch samples.

X/O/null and closed-island results are recorded separately as diagnostics.
They do not define a v4 wall cusp or inter-cusp cell.

Promotion means only that the criterion is numerically stable and
source-consistent across this held-out family. It is not experimental,
hardware, plasma-performance, or flight validation.

Execution is deliberately one-shot:

1. commit the protocol, implementation, and manufactured tests;
2. use a clean detached worktree at that commit;
3. acquire `results/execution-lock.json` exclusively;
4. execute once with `python -m experiments.cft_wall_cusp_validation_v1.run`;
5. commit the immutable results without patching or rerunning.

