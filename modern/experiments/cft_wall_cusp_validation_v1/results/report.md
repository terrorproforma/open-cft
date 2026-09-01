# Coupling v4 wall-cusp held-out validation v1 — failed immutable run

The sole preregistered run was executed from clean detached commit
`18fe790e2b35aaa4a2f173ed55568e683bcef927`, bound to accepted coupling
commit `f10d8213117fbafd8c2b69bdc103b6ef7b5d6d8c`.

The run failed during topology-diagnostic serialization after materializing
the primary, refined, and enlarged accepted L1a maps for the first case,
`wcval-f1-s04-p0-r0-neg`. A typed `BoundaryNullDiagnostic` was passed to
`list(point)`, raising `TypeError: 'BoundaryNullDiagnostic' object is not
iterable`.

Per the preregistered lifecycle, the implementation was not patched and the
experiment was not rerun.

## Coverage and gates

- Declared held-out cases/maps: 24/72
- Attempted cases: 1
- Completed held-out outcomes: 0
- Materialized accepted L1a maps: 3
- V4 records / opaque projections: 0/0
- Persisted GPU replay outcomes: 0
- Field residual/boundary/source/reconstruction gates for the three
  materialized maps: 3/3 passed
- Numerically promoted: no
- Ready for search v3: no
- Ready for plasma coupling: no

The exact three maps carry complete map hashes and v4 evidence fingerprints
in `failure.json`. Their relative residuals were
`9.6275975721539e-11`, `9.74710841930744e-11`, and
`9.810443289560541e-11`.

## Read-only diagnostic reconstruction

Reapplying the frozen criterion to the exact persisted maps, without solving
or rerunning the experiment, found:

- cusp counts: 5/5/5 across primary/refined/enlarged;
- candidate cell counts: 4/4/4;
- stable wall cusps: 15/15 across all maps;
- wall-terminated paths: 24/24;
- resolved cells: 0/12;
- resolved orbit samples: 24/72;
- nonadiabatic orbit samples: 48/72; and
- v4 status: ambiguous.

This reconstruction is diagnostic only. It is not a held-out outcome, accepted
projection, or substitute for complete all-case validation.

## Claim boundary

The 56-case characterization remains frozen development evidence only. This
failed run provides no experimental or hardware truth and makes no plasma
performance claim.

## Repository verification

- Focused manufactured preregistration/coupling tests: 45 passed.
- Python compileall: passed.
- Native C++ build/ctest: 1 passed.
- Full Python suite with importlib isolation: 1040 passed, 13 failed,
  36 errors, 2 skipped; failures are existing result/visualization artifact
  state outside this new path.
- FYP Git content: unchanged from the accepted commit.
- FYP MATLAB/Octave execution: not run because neither runtime is installed.

