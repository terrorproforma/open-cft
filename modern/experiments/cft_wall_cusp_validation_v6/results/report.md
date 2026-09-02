# Coupling v4.2 wall-cusp held-out validation v6

Preregistered held-out numerical validation of coupling schema 4.2.
This is numerical/source-consistency evidence, not hardware or experimental validation.

- Preregistration commit: `942f9e7d5f9936e572fadd1f98e388024cbcfab4`
- Accepted coupling commit: `b46e263950f91530ea61710b5dcc9354fc63cf6c`
- Cases/maps: 8/24
- Stable cusps/cells: 0/0
- Wall-connected paths: 240/240
- Resolved orbit samples: 0/720
- GPU replay: 8/8
- Criterion numerically promoted: false
- Search v3 ready: false
- Plasma coupling ready: false

## Gates

- Three-map field gates: 8/8
- Cross-map cusp/cell stability: 0/8
- Opaque projection acceptance: 0/8

## Failure taxonomy

- `AXIAL_METRIC_DRIFT`: 0
- `CELL_BOUND_SHIFT`: 0
- `CELL_COUNT_DISAGREEMENT`: 0
- `CUSP_ASSIGNMENT_SHIFT`: 0
- `CUSP_COUNT_DISAGREEMENT`: 0
- `CUSP_STRENGTH_DRIFT`: 0
- `FIELD_ENLARGED_INVALID`: 0
- `FIELD_PRIMARY_INVALID`: 0
- `FIELD_REFINED_INVALID`: 0
- `GEOMETRY_INVALID`: 0
- `GPU_FIELD_REPLAY_FAILURE`: 0
- `GPU_RESIDUAL_REPLAY_FAILURE`: 0
- `HELD_OUT_MEMBERSHIP_INVALID`: 0
- `OPAQUE_PROJECTION_REJECTED`: 0
- `ORBIT_NONADIABATIC`: 0
- `ORBIT_UNVERIFIED`: 8
- `PATH_EXTREMA_INVALID`: 0
- `PATH_LENGTH_CONVERGENCE_FAILURE`: 0
- `PATH_NOT_WALL_CONNECTED`: 0
- `PATH_PSI_DRIFT`: 0
- `SOURCE_CONSISTENCY_FAILURE`: 0
- `THREE_MAP_FINGERPRINT_FAILURE`: 0
- `UNCERTAINTY_BOUNDS_INVALID`: 8
- `UNCERTAINTY_DOMINATED`: 0
- `WALL_CUSP_UNRESOLVED`: 8
- `WALL_ENDPOINT_SHIFT`: 0

X/O/null and closed-island outputs are diagnostics only; they do not define
or promote a wall cusp. No experimental truth, plasma performance, or
hardware qualification is claimed.

