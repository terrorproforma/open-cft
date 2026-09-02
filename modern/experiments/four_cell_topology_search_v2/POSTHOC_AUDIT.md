# Four-cell topology search v2 posthoc audit (protocol copy end-of-line)

## Verdict

The recorded result stands as a **preregistered null**: 128 of 128
candidates evaluated, 128 three-map field acceptances, 0 candidates stable
under the frozen four-cusp/cell definition (`TOPOLOGY_COUNT` and
`TOPOLOGY_UNSTABLE` on every candidate), 0 coupled, 0 plasma states, 0
performance publications. Nothing about the numerical result changes.

The single end-of-line defect found is a **portability defect in the
recording layer**, not in the evidence: `experiment.py` hashed `protocol.json`
as it lay in a Windows checkout with `core.autocrlf=true`, so the one
authorised execution recorded the SHA-256 of the **CRLF working-tree bytes**
as `protocol_sha256` (`manifest.json`, `dataset.json`, `execution-lock.json`),
as the `sha256`/`bytes` of the manifest artifact entry
`preregistered-protocol.json`, and in the sidecar
`results/preregistered-protocol.json.sha256`. Git has always stored the LF
form (blob `9fda58fa…`, identical for `protocol.json` at the preregistration
commit, for `protocol.json` at `HEAD` and for `results/preregistered-protocol.json`
at both the result commit and `HEAD`), and since the repo-wide
`* text=auto eol=lf` pin (`fab0eccc`, 2026-09-03) every checkout receives LF
bytes whose SHA-256 differs. The content of the protocol is untouched: its
canonical payload digest (EOL-independent, `bd522269…`) recomputes on this
checkout and is the digest `dataset.json` binds as `protocol_payload_sha256`.
Every other file under `results/` is byte-exact. No file under `results/`, no
frozen preregistration file and no test asserting the run's numbers is
modified by this overlay.

## Immutable bindings

- preregistration commit: `d6317910703de91ca6dc25c4d4d855e36cc3b14d`
  (`preregister physical four-cell topology search v2`,
  2026-09-02T05:08:10+10:00);
- result commit: `7120e8edcb74c02c1df968c730d1f93b3758b4e1`
  (`record four-cell v2 null terminal-gate result`, 2026-09-02T05:14:51+10:00;
  direct child of the preregistration commit);
- `results/` tree: `56b41d451d94e0fde1f86bd4d3a40b7fbc2470b2` (identical at
  `7120e8ed` and at this overlay's `HEAD`; 28 tracked files);
- `results/manifest.json` SHA-256:
  `f5e26373d72bd13aa5631516009797567e0f66812941d993d5fad534be07240a`
  (sealed; canonical payload `f444dee5…` recomputes over the recorded CRLF
  digest, so the manifest is internally consistent);
- frozen protocol at `d6317910`: `protocol.json` blob
  `9fda58faa70e49da8e17a94a478329fd6d408f3c` (unchanged at `HEAD`; the
  results copy `results/preregistered-protocol.json` is the same blob);
- results sidecar blob `results/preregistered-protocol.json.sha256`:
  `dc73a9d384b7ad17bd39f4b1def43e189b4e8529` (unchanged since `7120e8ed`);
- protocol canonical payload SHA-256 (`json-sort-keys-compact-utf8-v1` over
  the whole protocol object):
  `bd522269b87e555fee279bd669d34e2b6a98a31540c6f4687cfcf51b40614c33`, bound
  as `protocol_payload_sha256` by `dataset.json`;
- accepted coupling v3 baseline: `f80a360fd740a30017cdac1874cedbfa2806874a`.

## 1. Finding: the protocol copy differs from the checkout by end-of-line only

`protocol.json` is 231 lines. On this LF checkout it is 10580 bytes with no
`\r`; the bundle records the digest and length of 10811 bytes, i.e. the same
text with one extra byte per line. Replacing `\n` by `\r\n` reproduces the
recorded digest exactly:

| path | checkout bytes | recorded bytes | checkout sha256 | recorded sha256 | CRLF-recomputed sha256 | match |
| --- | ---: | ---: | --- | --- | --- | --- |
| `preregistered-protocol.json` | 10580 | 10811 | `5c195119c7a3c3c7e8b2c2d58e2e9836ac0ece6e000e52b0fd86c4718446c1b4` | `ec2e9a732b7d0e909ff742ebbbb0215e1102909c148b812306df6f0759f48e49` | `ec2e9a732b7d0e909ff742ebbbb0215e1102909c148b812306df6f0759f48e49` | CRLF == recorded |

The recording layer was internally consistent: `experiment.py` computed
`PROTOCOL_SHA256 = sha256(protocol.json bytes)` on the producing checkout,
wrote the same bytes to `results/preregistered-protocol.json` with a sidecar
of that digest, and copied the digest into the lock, the dataset and the
manifest. Those bindings agree with each other. The bytes were simply not the
bytes Git preserves.

### What the finding does not touch

- The preregistered content: `sha256(canonical_json(protocol))` recomputes to
  `bd522269…` on this checkout and equals `dataset.protocol_payload_sha256`.
  Canonical JSON carries no end-of-line bytes, so the preregistered content
  (128-candidate shifted-Halton sample, three maps per candidate, the
  four-cusp count/registration/stability gates, the failure taxonomy, ranking,
  replay and publication policy) is exactly what executed.
- Every other file under `results/` is byte-exact on this LF checkout: the
  other 12 manifest artifact entries (dataset, runtime, report, lock, eight
  representative geometry/field files) match their recorded `sha256` and
  `bytes`, and all 13 other `*.sha256` sidecars (those 12 plus
  `manifest.json.sha256`) attest the LF bytes exactly. No file under
  `results/` contains a `\r`. `experiment.py` wrote its own outputs with
  `write_bytes` and its sidecars with `newline="\n"`, so the run's recording
  layer was EOL-portable for everything it generated itself; only the copied
  protocol inherited the checkout's line endings.
- The committed dashboard `modern/visualization/plasma-topology-results.html`
  (`generate_plasma_topology_dashboard.py::_four_cell_v2`) verifies the
  manifest, dataset, report and representative artifacts byte-exactly and does
  not read the protocol copy; it is unaffected.

## 2. Resolution (bound tolerance, no evidence edited)

`audit_sidecar_eol.py` (standard library only; imports nothing from the
experiment package) recomputes everything above from the bytes on disk.
`eol_equivalent_digest(path, data)` returns the recorded digest **only** for
`results/preregistered-protocol.json`, only when the bytes contain no `\r`,
hash to the audited LF digest `5c195119…` **and** their CRLF transform hashes
to the recorded digest `ec2e9a73…`; any other byte difference, any other file,
or any other digest returns `None`. Consumers that must bind the recorded
identity (the paper's typed manifest and `check_paper.py`) apply exactly this
rule for exactly this file and fail closed on everything else.

## 3. Disclosures

1. **Protocol EOL defect.** Described in §1. Root cause: `PROTOCOL_SHA256` was
   computed from the working-tree bytes of `protocol.json` in a
   `core.autocrlf=true` checkout before the repository pinned `eol=lf`.
2. **`validate_results` refuses the recorded bundle on an LF checkout.**
   `experiment.py::validate_results` compares the bundle's recorded
   `protocol_sha256` with the live `PROTOCOL_SHA256` of the checkout, which is
   now the LF digest, and raises `result protocol/schema/baseline identity
   mismatch`. The experiment-local lifecycle test
   (`tests/test_search_v2.py::test_result_lifecycle_before_or_after_single_run`)
   therefore fails on LF checkouts; it is not collected by `modern/tests`.
   This overlay does not change `experiment.py`; the audited rule above is
   the reviewer path.
3. **GPU replay diagnostic tolerance.** Independently of end-of-line bytes,
   the bundle's own `summary` records `gpu_replay_pass_count: 2` of
   `gpu_replay_required_count: 4`. All four replay candidates reproduced the
   field components within the preregistered tolerances (`|ΔB| ≤ 4.4e-15` T,
   `|Δψ| ≤ 2.3e-20` Wb against limits of 2e-8 T and 2e-10 Wb), but the
   residual-diagnostic relative difference exceeded its `5e-6` limit for
   `v2-031` (`9.42e-6`) and `v2-063` (`6.50e-6`). `validate_results` would
   also refuse the bundle on `required tolerance-based GPU replay did not
   pass`. The topology null (all 128 candidates failing `TOPOLOGY_COUNT` and
   `TOPOLOGY_UNSTABLE` on maps that passed every field gate) does not depend
   on the replay diagnostic; any consumer must report the replay outcome as
   recorded.
4. **Frozen files not edited.** `protocol.json`, `DEVLOG.md`,
   `LEARNING_SCRATCHPAD.md` and every file under `results/` are left
   untouched; this audit is recorded here and in
   `modern/docs/workstreams/paper-devlog.md`.

## 4. Reviewer reproduction

From `modern/` (PowerShell shown; the script is read-only):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
# must print nothing: protocol and bundle are LF on disk
git ls-files --eol -- experiments/four_cell_topology_search_v2 | Select-String 'w/crlf'
# the table in section 1
python -m experiments.four_cell_topology_search_v2.audit_sidecar_eol --table
# full JSON report; exit code 0 iff passed
python -m experiments.four_cell_topology_search_v2.audit_sidecar_eol
python -m pytest tests/experiments/four_cell_topology_search_v2 -q
```

The report must show `counts = {byte_exact: 25, eol_only: 2, mismatch: 0}`
(27 entries: 13 manifest artifacts and 14 sidecars; the protocol copy appears
once in each list), `eol_only_paths_are_exactly_expected: true`,
`bundle_binds_recorded_protocol_file_digest: true`,
`bundle_binds_protocol_payload_digest: true` and `passed: true`. The test
module `test_posthoc_audit.py` re-derives the table live, proves the script
leaves the experiment directory byte-identical, shows that a CRLF-restored
scratch copy hashes to the recorded digest, and shows that the tolerance does
not apply to any other file or to any other byte change.

## 5. Overlay contents

This audit adds `POSTHOC_AUDIT.md`, `audit_sidecar_eol.py` and
`modern/tests/experiments/four_cell_topology_search_v2/test_posthoc_audit.py`
and changes no other file. The results tree hash and the preregistered blobs
are asserted unchanged by the test.
