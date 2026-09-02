# Coupling v4 wall-cusp held-out validation v2 - immutable failure

The sole clean-detached attempt failed before any held-out access. It was not patched or rerun.

- Failure: the exclusive lock payload contained a raw UTC `datetime`, which canonical `json.dumps` rejected.
- The exclusive filename was created, but the original lock artifact is zero bytes and preserved separately.
- The launcher finalizer then failed while parsing that empty lock; both worker and launcher logs are retained.
- Attempted cases/maps: 0/0 of 24/72.
- Candidate/resolved cells: 0/0.
- Candidate/resolved paths: 0/0.
- Candidate/resolved orbits: 0/0.
- GPU replay: 0.
- Opaque projections: 0.
- Criterion numerically promoted: false.
- Search v3/plasma coupling ready: false/false.

The v1 first case is disclosed as accessed and is excluded from v2. No experimental or hardware truth is claimed.
