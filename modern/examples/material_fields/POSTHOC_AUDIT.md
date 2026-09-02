# Material-fields v1.4 example artifacts posthoc audit (implementation digests)

## Verdict

The three L1b v1.4 reduced-screening artifacts under `artifacts/` remain
`SCREENING_NOT_ACCEPTED` / `STRUCTURED_GRID_L1B_INSUFFICIENT` with all 18
publication gates `NOT_EVALUATED`; nothing about their numerical content or
their status changes.

The single defect found is a **portability defect in the source-identity
binding**, not in the evidence and not in the solver: every raw run binds the
SHA-256 of the `cft_revival.material_fields` source files
(`implementation_sha256`, `evidence_implementation_sha256`;
`numerics._implementation_sha256` hashes the files' bytes). The artifacts were
generated at commit `8603a905` on a Windows checkout with
`core.autocrlf=true`, so the three recorded digests are hashes of the **CRLF
working-tree bytes**. Git stores the LF form, and since the repo-wide
`* text=auto eol=lf` pin (`fab0eccc`, 2026-09-03) every checkout receives LF
bytes whose digests differ. Strict validation
(`acceptance._validate_raw_runs`, `replay.replay_raw_run`) compares the
recorded digests with the live source and therefore refused every artifact
with `raw run hash binding failed` on any LF checkout, failing four tests in
`tests/material_fields/test_spec_ledgers.py`.

## Immutable bindings

- generating commit: `8603a905f8b19873e9a91c1afd237864e8b31aff`
  (`record L1b v1.4 reduced screening evidence`, 2026-09-02T09:02:47+10:00);
  it is the last commit touching both `artifacts/` and
  `modern/src/cft_revival/material_fields/`, so the source blobs at `8603a905`
  are the blobs at this overlay's `HEAD` (`git diff --quiet 8603a905 HEAD --
  modern/src/cft_revival/material_fields` is empty);
- CRLF-era payload digests (`integrity.payload_sha256`) at `8603a905`:
  compact `7579f1602c75cdf6773b24279dd7621dd9c290dd9c2205b875da0962e5c7ed67`,
  divergent `da7ef3f3660f1b2e6f6ea3ac9840bed23b1f177efc9168f75d3c90f5c5f12966`,
  historical `d91f4dd8b86251ec4294948b7df4e8a700ec362e45dc920752fca4592039a860`,
  manifest `32ce64983a03fc7278be1ceaa7bf20fb73e9b04926786530a1801148893ba134`
  (these are the digests recorded in `docs/workstreams/material-fields-devlog.md`
  for the v1.4 closure);
- CRLF-era implementation digests recorded 105 times across the six artifact
  and viewer files at `8603a905` (36 × evidence, 63 × warp, 6 × python) and
  nowhere else: see the table in §1.

## 1. Finding: the recorded digests are the CRLF hashes of unchanged source

All seven bound source files are CR-free on this checkout. Hashing the Git
blobs of `8603a905` (identical to `HEAD`) exactly as
`numerics._implementation_sha256` does, but with `\n` replaced by `\r\n`,
reproduces every recorded digest:

| digest role | files | LF sha256 (blob) | CRLF-recomputed sha256 | recorded at 8603a905 | match |
| --- | --- | --- | --- | --- | --- |
| `evidence` | `acceptance.py`, `adapters.py`, `artifacts.py`, `models.py`, `numerics.py`, `replay.py`, `warp_solver.py` | `ef17d1618a934bd1038e24ead341a519dcf365f8f54f1d360afbfedf4bf908db` | `d229f62d7ba6289646291d925f404785ab879b91f59185a91a90c327e92966b8` | `d229f62d7ba6289646291d925f404785ab879b91f59185a91a90c327e92966b8` | CRLF == recorded |
| `warp` | `adapters.py`, `models.py`, `numerics.py`, `warp_solver.py` | `6ced73daca60f883440d9f1a4287549ecd2cb8335c138e0fb121b319c0038d2f` | `dc988f4b01648e825ac7a1934b8ddca88ad53d1fa5859c8471e1dfcec745cd0b` | `dc988f4b01648e825ac7a1934b8ddca88ad53d1fa5859c8471e1dfcec745cd0b` | CRLF == recorded |
| `python` | `adapters.py`, `models.py`, `numerics.py` | `2ce98ebd46cab554fc38e81a099935eafcf4a93cf1059a3d86c64fb498fcdd61` | `734cff6aabe3964690ee6ccfa3bc5c3f9f88f2bc7184ffc9390a06b5b903e6b5` | `734cff6aabe3964690ee6ccfa3bc5c3f9f88f2bc7184ffc9390a06b5b903e6b5` | CRLF == recorded |

Before this overlay every artifact, viewer and the manifest was byte-exact
against its `.sha256` sidecar and every sealed payload recomputed; the
recording layer was internally consistent. The bytes hashed for the source
identity were simply not the bytes Git preserves.

### What the finding does not touch

- Raw solutions, rasterised problems, solver diagnostics (other than the
  recorded implementation digest), the 18 gates, the reduced-resource
  measurements and every summary value are identical before and after this
  overlay (verified field-by-field by the structural diff described in §2).
- No source file in `cft_revival.material_fields` is modified: the live
  package is the code that produced the evidence, byte-for-byte modulo EOL.

## 2. Resolution (sanctioned re-bind; tolerance is structurally impossible here)

The pattern used for the orbit wall-loss v4 and geometry-sweep v2 sidecars —
tolerate the audited CRLF digest in verification code — cannot be applied to
a **source-bytes** binding: the comparison sites (`acceptance.py`,
`replay.py`) and the hash function (`numerics.py`) are themselves among the
hashed files, so any tolerance added there changes the bytes being hashed and
falsifies the very equivalence it would tolerate. The workstream instead
provides `refresh_artifact_metadata.py` for exactly this situation
("Numerical simulation arrays and gate thresholds were unchanged; only
evidence metadata and hashes were regenerated", material-fields devlog,
2026-09-02): it re-derives every binding under the live implementation,
re-runs `assess_publication` from the embedded raw runs, and calls
`validate_artifact`, which replays all 30 raw runs and rejects any migration
that would change a bound numerical result.

That script was run once on this LF checkout. The structural diff of the
seven JSON files before/after shows changes only at these leaves:
`implementation_sha256` (69 occurrences: raw run, its diagnostics, anchors,
artifact diagnostics), `evidence_implementation_sha256` (36), `run_sha256`
(60: raw runs and studies), `base_run_sha256` (6), `cpu_run_sha256` /
`cuda_run_sha256` (3 each), `integrity.payload_sha256` (7),
`artifact_payload_sha256` (6: viewers and manifest) and
`artifact_file_sha256` (3: manifest). Raw `solution`, `problem`, all other
diagnostics, `gates` and `summary` are equal. A second run of the script
reproduces the same bytes (deterministic).

Re-bound digests (LF):

| file | file sha256 | payload sha256 |
| --- | --- | --- |
| `compact-high-gradient-stack.material-field.json` | `7c30a39150e4bdd233a38760f11413def91f58a177db34ffeaa94dc936062c8f` | `a6b69d03bde3626f31858b2b914ef8b6d2d9bdc26a3925371b7914a300f60da0` |
| `divergent-exit-stack.material-field.json` | `5a70664bbcda76a289dccc6e77aa38bc36609d916e8f5681e5da7cc26779b9e9` | `cf703753108f36a0d175e0540262b367d034080141634a062c8820c582194a06` |
| `historical-envelope-baseline.material-field.json` | `23f26357efd01500d18c31ae825a1a84205338ff0cef0d898eea38fb66a67ef0` | `dc1ab5ed462fd34271db64866fb45097767ce64c0f0129bcae69591a68244dcf` |
| `manifest.json` | `8e6ca2eff914b19b612572b0d63f400f4cda203996e104fab2baa63d32f87f34` | `eba362d8b18f46f8e5254eceecb4092c0e12b35673a011014a3751c43113ae7c` |

The implementation digests now recorded are exactly the LF blob digests in
the table of §1 (`ef17d161…`, `6ced73da…`, `2ce98ebd…`), so the CRLF path is
closed: any future difference between recorded and live digests is a real
implementation change and must again be resolved by the refresh script.

## 3. Disclosures

1. **Source-identity EOL defect.** Described in §1. Root cause:
   `numerics._implementation_sha256` hashes raw file bytes, which depended on
   the checkout's `core.autocrlf` until the repository pinned `eol=lf`.
2. **Refresh script wrote CRLF sidecars.** `refresh_artifact_metadata.write`
   wrote `.sha256` sidecars through the text layer without `newline="\n"`,
   i.e. CRLF on Windows (the same pattern fixed in orbit_mc 1.7). Fixed in
   this overlay (`newline="\n"`); the first refresh run's CRLF sidecars were
   replaced by the second, identical run. The digests inside the sidecars are
   unaffected (they name the artifacts' bytes, not their own).
3. **`generate_design_results.py` and `artifacts.write_json` are unchanged.**
   The generator is only used for a full re-solve and was not run; the
   library writer was not audited here because changing `artifacts.py` would
   itself change the evidence implementation digest.
4. **This is not preregistered experiment evidence.** The artifacts live under
   `examples/`, are screening-only (`require_accepted=False`), and have been
   regenerated by the workstream before; no `results/` directory and no frozen
   preregistration file is involved.

## 4. Reviewer reproduction

From `modern/` (PowerShell shown; the audit script is read-only):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
# must print nothing: source and artifacts are LF on disk
git ls-files --eol -- src/cft_revival/material_fields examples/material_fields | Select-String 'w/crlf'
# the table in section 1 (anchored to the Git blobs of 8603a905)
python examples/material_fields/audit_implementation_eol.py --table
# full JSON report; exit code 0 iff passed
python examples/material_fields/audit_implementation_eol.py
python -m pytest tests/material_fields -q
```

The report must show `source_is_lf: true`,
`history.all_crlf_reproduce_recorded: true`,
`history.era_artifacts_recorded_exactly_the_three: true`,
`artifacts.all_files_byte_exact: true`,
`artifacts.counts = {byte_exact: 3, eol_only: 0, mismatch: 0}`,
`live_state: "rebound_lf"` and `passed: true`.
`tests/material_fields/test_posthoc_eol_audit.py` re-derives the table live,
proves the script is read-only, and checks that the live source is the
`8603a905` source and that the recorded digests are its LF digests.

## 5. Overlay contents

This overlay adds `POSTHOC_AUDIT.md`, `audit_implementation_eol.py` and
`tests/material_fields/test_posthoc_eol_audit.py`, fixes the sidecar newline
in `refresh_artifact_metadata.py`, and re-binds the seven artifact JSON files
and their seven sidecars under `artifacts/` by running that script. No file
under `src/cft_revival/material_fields/` is modified.
