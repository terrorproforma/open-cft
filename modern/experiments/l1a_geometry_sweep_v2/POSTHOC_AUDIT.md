# L1a geometry sweep v2 posthoc audit (protocol sidecar end-of-line)

## Verdict

Evidence **ACCEPTED** as `L1a_field_only_design_space_screening`
(terminal status `ACCEPTED`, 96/96 cases, 0 failures, 25 nondominated, four
unique representative artifacts). Nothing about the numerical result changes.

The single defect found is a **portability defect in the preregistration
recording layer**, not in the evidence: `protocol.json.sha256` was frozen at
the preregistration commit on a Windows checkout with `core.autocrlf=true`,
so it records the SHA-256 of the **CRLF working-tree bytes** of
`protocol.json`, and the one authorised execution copied that digest into the
immutable bundle as `protocol_file_sha256`. Git has always stored the LF form
(`37d455a9…`, identical at the preregistration commit and at `HEAD`), and
since the repo-wide `* text=auto eol=lf` pin (`fab0eccc`, 2026-09-03) every
checkout receives LF bytes whose SHA-256 differs. The content of the protocol
is untouched: its canonical payload digest (EOL-independent, `da319f22…`)
recomputes and is the digest every bundle file binds. Every file under
`results/` is byte-exact. No file under `results/`, no frozen preregistration
file and no test asserting the run's numbers is modified by this overlay.

## Immutable bindings

- preregistration commit: `092f5fae692ee7d6711e0c7e1c94dac6a345f37c`
  (`preregister L1a geometry sweep v2`, 2026-09-02T03:03:16+10:00);
- result commit: `f30cb42ec4a8633bf634a3d32ffa5b11f66be97a`
  (`record L1a geometry sweep v2 results`, 2026-09-02T03:06:19+10:00; direct
  child of the preregistration commit; adds 34 files under `results/` and
  nothing else);
- `results/` tree: `de85a158a01aa4113154ef256c9d11032bdf6538` (identical at
  `f30cb42e` and at this overlay's `HEAD`);
- `results/manifest.json` SHA-256:
  `768b345e946a45e623f83aaa18e01f8ec5bc7f823e81858a0a8c3a3e2e448754`;
- frozen inputs at `092f5fae`: `protocol.json` blob
  `37d455a952306d9a6fe36456a1c0a3c6fd4c747a`, `protocol.json.sha256` blob
  `270da0c4c0939ed727b0ceb1c4ad9cc9cfb762c1` (both unchanged at `HEAD`);
- protocol canonical payload SHA-256 (`integrity.payload_sha256`,
  `json-sort-keys-compact-utf8-v1`):
  `da319f2271d56b0d0c883b76d3106b094359a608b560d58ac7801de1293ecbc8`, bound
  as `protocol_payload_sha256` by `manifest.json`, `raw-results.json`,
  `summary.json` and `execution-lock.json`;
- execution lock `started_at_utc` `2026-09-01T17:04:35.202481+00:00`, between
  the two commit times.

## 1. Finding: the protocol sidecar differs from the checkout by end-of-line only

`protocol.json` is 134 lines. On this LF checkout it is 7790 bytes with no
`\r`; the sidecar records the digest of 7924 bytes, i.e. the same text with
one extra byte per line. Replacing `\n` by `\r\n` reproduces the recorded
digest exactly:

| path | checkout bytes | recorded bytes | checkout sha256 | recorded sha256 | CRLF-recomputed sha256 | match |
| --- | ---: | ---: | --- | --- | --- | --- |
| `protocol.json` | 7790 | 7924 | `2a5ba9e46c777225384539a4c453a43aa3298c956b32b022cc5ddeac72ba874c` | `64b2c58c3cecb2ea1836d2bf48e23ff83dffb114866bf21e7135b411beaa2b2c` | `64b2c58c3cecb2ea1836d2bf48e23ff83dffb114866bf21e7135b411beaa2b2c` | CRLF == recorded |

The recording layer was internally consistent: `run.py` hashed the bytes it
actually read on the producing checkout (`verify_sidecar(PROTOCOL_PATH)`) and
wrote that digest into `execution-lock.json`, `raw-results.json` and
`manifest.json` (`protocol_file_sha256 = 64b2c58c…`). Those three bindings
agree with the sidecar. The bytes were simply not the bytes Git preserves.

### What the finding does not touch

- The sealed protocol payload: `sha256(canonical_json(protocol − integrity))`
  recomputes to `da319f22…` on this checkout and equals the
  `protocol_payload_sha256` recorded by all four bundle files. Canonical JSON
  carries no end-of-line bytes, so the preregistered content (96-case sample,
  solver, QoIs, objectives, roles, seven gates, replay contract) is exactly
  what executed.
- Every file under `results/` is byte-exact on this LF checkout: the 16
  entries of `manifest.deterministic_files` (lock, raw results, summary,
  report, 4 × geometry / full field / downsampled field) match their recorded
  `file_sha256`, and all 17 `*.sha256` sidecars (those 16 plus
  `manifest.json.sha256`) attest the LF bytes exactly. No file under
  `results/` contains a `\r`. `run.py` wrote its outputs with `write_bytes`
  and its sidecars with `newline="\n"`, so the run's own recording layer was
  already EOL-portable; only the hand-frozen preregistration sidecar was not.
- `validate_bundle` (`validate.py`) and the committed dashboard
  (`visualization/generate_dashboard.py`) reproduce the front (25), the five
  roles, the four unique representatives and all seven gates from the bundle.

## 2. Resolution (bound tolerance, no evidence edited)

Verification code accepts the audited alternate digest for **exactly this one
file** and fails on anything else:

- `protocol.py` — `EOL_AUDITED_SIDECARS` lists `protocol.json` with the LF
  digest `2a5ba9e4…` and the recorded digest `64b2c58c…`. `verify_sidecar`
  first requires the ordinary byte-exact sidecar; only when that fails and
  the path is the audited file does it accept the sidecar iff the checkout
  bytes contain no `\r`, hash to the audited LF digest, **and** their CRLF
  transform hashes to the recorded digest. It then returns the recorded
  digest, which is the identity the immutable bundle binds. Any other byte
  difference, any other file, or a sidecar naming any other digest still
  raises `invalid SHA-256 sidecar`.
- `visualization/generate_dashboard.py` — `_verify_file` applies the same
  rule for `protocol.json` only (`AUDITED_PROTOCOL_LF_SHA256`), so the
  committed dashboard's `protocol_file_sha256` identity (`64b2c58c…`) is
  unchanged and the rendered HTML is byte-identical to the committed file.
- `audit_sidecar_eol.py` recomputes everything above from the bytes on disk
  and additionally asserts that the tolerance constants in `protocol.py`
  equal the audited digests (`tolerance_bound_to_audit`).

## 3. Disclosures

1. **Sidecar EOL defect.** Described in §1. Root cause: the preregistration
   sidecar was produced by hashing `protocol.json` as it lay in a
   `core.autocrlf=true` working tree (CRLF), before the repository pinned
   `eol=lf`. The downstream experiments `l1a_field_surrogate_v1` and
   `l1a_field_surrogate_v2` import `experiments.l1a_geometry_sweep_v2` (which
   verifies the sidecar at import time) and therefore failed to collect for
   the same single cause; their own frozen sidecars are byte-exact on this
   checkout and need no tolerance.
2. **`results/` is masked by the ignore rule.** Root `.gitignore` `Results/`
   matches the directory case-insensitively on this filesystem; the 34 result
   files were nevertheless committed at `f30cb42e` and are all tracked (the
   tree hash above is what reviewers compare).
3. **Frozen files not edited.** `protocol.json`, `protocol.json.sha256`,
   `DEVLOG.md` and `LEARNING_SCRATCHPAD.md` are preregistered paths
   (`run.py: PREREGISTERED_PATHS`) and are left untouched; this audit is
   recorded here and in `modern/docs/workstreams/test-health-devlog.md`
   rather than in the experiment devlog.

## 4. Reviewer reproduction

From `modern/` (PowerShell shown; the script is read-only):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
# must print nothing: protocol and bundle are LF on disk
git ls-files --eol -- experiments/l1a_geometry_sweep_v2 | Select-String 'w/crlf'
# the table in section 1
python -m experiments.l1a_geometry_sweep_v2.audit_sidecar_eol --table
# full JSON report; exit code 0 iff passed
python -m experiments.l1a_geometry_sweep_v2.audit_sidecar_eol
python -m pytest tests/experiments/l1a_geometry_sweep_v2 tests/experiments/l1a_geometry_sweep_v2_visualization -q
```

The report must show `counts = {byte_exact: 33, eol_only: 1, mismatch: 0}`,
`eol_only_paths_are_exactly_expected: true`,
`bundle_binds_recorded_protocol_file_digest: true`,
`bundle_binds_protocol_payload_digest: true`, `tolerance_bound_to_audit: true`
and `passed: true`. The test module `test_posthoc_audit.py` re-derives the
table live, proves the script leaves the experiment directory byte-identical,
shows that a CRLF-restored scratch copy verifies through the ordinary
byte-exact path with the same digest, and shows that the tolerance does not
apply to any other file or to any other byte change.

## 5. Overlay contents

This audit adds `POSTHOC_AUDIT.md`, `audit_sidecar_eol.py` and
`tests/experiments/l1a_geometry_sweep_v2/test_posthoc_audit.py`, and changes
verification code only in `protocol.py` (`EOL_AUDITED_SIDECARS`,
`eol_equivalent_digest`, `verify_sidecar`) and
`visualization/generate_dashboard.py` (`AUDITED_PROTOCOL_LF_SHA256`,
`_eol_audited_digest`, `_verify_file`). The results tree hash and the
preregistered blobs are asserted unchanged by the test.
