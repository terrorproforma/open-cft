# CFT wall-cusp validation v2

This directory contains the audit-corrected preregistered held-out numerical
validation of the frozen coupling-v4 wall-cusp criterion and schema 4.1,
bound to commit `f10d8213117fbafd8c2b69bdc103b6ef7b5d6d8c`.

The 24-case v2 family uses stages 5/7/9, pitches 5.6/7.5 mm, chamber radii
8.2/11.1 mm, and both polarities. These coordinates and IDs are disjoint from
the frozen 56-case development family and from every v1-accessed case. In
particular, `wcval-f1-s04-p0-r0-neg` was accessed during v1 and is explicitly
not held out for v2.

The manufactured preflight exercises finite-boundary, interior, and empty-null
serialization plus nested production v4/v3 records before held-out access.
The orbit verifier cryptographically resolves each path to its exact accepted
map, samples B along that path, computes `mu = m v_perp^2/(2B)`, evolves
energy/pitch/mirroring over nested timesteps, and reports polyline convergence
as a separate metric. Thresholds were fixed from manufactured/development-only
evidence and were not tuned from v1.

Every phase checkpoints dependency closure, access events, map bytes,
prerecords, replay, diagnostics, and typed failure evidence. Candidate and
resolved cell/path/orbit counts remain distinct.

X/O/null and closed-island results are recorded separately as diagnostics.
They do not define a v4 wall cusp or inter-cusp cell.

Promotion means only that the criterion is numerically stable and
source-consistent across this held-out family. It is not experimental,
hardware, plasma-performance, or flight validation.

Execution is deliberately one-shot:

1. commit the protocol, implementation, and manufactured tests;
2. use a clean detached worktree at that commit;
3. acquire `results/execution-lock.json` exclusively;
4. execute once with `python -m experiments.cft_wall_cusp_validation_v2.run`;
5. commit the immutable results without patching or rerunning.

