# L0 surrogate v3 — immutable results and provenance failure

## Scientific status

V3 is **invalid and not accepted**. The one technical execution completed, but
its invocation recorded
`658fe5c6b3be8db3337e1d5d90e69ca9b2171831` instead of the actual pushed
preregistration commit
`658fe5c34dd6b669d1e426d8aa162782d7809ca4`. The supplied value is not a Git
object. This failure was detected after execution.

No file was patched and no execution was repeated. The generated artifacts are
preserved exactly. Their numerical metrics are diagnostic failed-run evidence,
not valid preregistered results.

Provenance-failure hash:
`ec6d262bf4b1c54a198501b53d2cc7742fed21c377e01518d1f6024d47672af0`.

## Technical numerical outcome

The technical run also failed the frozen numerical acceptance rule: no active
or fixed-baseline replicate passed every output/scope gate.

### Group split 2026090201

- Active overall thrust: NRMSE 2.164%, coverage 0.849, worst error 8.757% range.
- Active overall Isp: NRMSE 3.620%, coverage 0.821, worst error 10.649% range.
- Active interpolation passed; boundary coverage failed for both outputs; OOD
  coverage failed high for thrust and low for Isp.
- Baseline overall thrust/Isp NRMSE: 5.420%/8.305%; both point-error gates
  failed despite overall coverage 0.883/0.901.

### Group split 2026090202

- Active overall thrust: NRMSE 2.410%, coverage 0.834, worst error 13.078% range.
- Active overall Isp: NRMSE 2.764%, coverage 0.932, worst error 9.051% range.
- Active Isp and both OOD outputs passed; thrust boundary and overall coverage
  failed.
- Baseline overall thrust/Isp NRMSE: 6.209%/5.440%; coverage 0.971/0.890.

### Group split 2026090203

- Active overall thrust: NRMSE 2.221%, coverage 0.915, worst error 13.821% range.
- Active overall Isp: NRMSE 3.438%, coverage 0.841, worst error 10.605% range.
- Boundary thrust, interpolation coverage, and OOD coverage failed; OOD Isp
  also narrowly failed the 5% NRMSE gate at 5.072%.
- Baseline overall thrust/Isp NRMSE: 7.086%/10.277%; coverage 0.824/0.797.

The active campaign materially reduced point errors relative to the fixed
baseline, but split-conformal coverage was not stable across strata.

## Serialization and verification

The v3 serializer fixed the v2 directory failure:

- all 12 final models were written through missing/deep parent directories,
  atomically reloaded, and hash checked;
- all three selection/model/calibration/freeze/assessment bundles completed;
- no atomic temporary files remained;
- runtime was 196.875 s, diagnostic only.

The pre-execution suite covered missing/deep parents, read-only/permission
failure, cleanup, three synthetic replicates, a tiny full execution, and
single-use assessment without accessing real v3 assessment labels.

## Required successor

Any successor must validate the supplied commit with `git cat-file -e
<sha>^{commit}` and require exact equality to the expected pushed
preregistration SHA before creating its execution lock. V2 remains immutable
serialization-failure evidence; v3 remains immutable provenance-failure
evidence. Neither is physical-accuracy evidence.
