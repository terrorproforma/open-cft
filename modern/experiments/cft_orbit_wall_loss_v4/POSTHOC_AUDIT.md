# CFT full-orbit wall-loss v4 posthoc audit

## Verdict

Evidence **ACCEPTED** as
`collisionless_prescribed_field_test_particle_wall_loss_not_pic`.

The single defect found is a **portability defect in the recording layer**,
not in the evidence: orbit_mc 1.6.0 wrote the nine per-case
`artifacts/orbits/<case>.json.sha256` sidecars with the platform end-of-line
(CRLF on the producing Windows host), and the bundle recorded those CRLF byte
hashes, while Git (`* text=auto eol=lf`) stores and checks out the LF form.
Every byte of every orbit artifact, checkpoint, summary, gate report,
transition, counter and access record is reproduced exactly by this checkout;
for the nine sidecars the only difference is one `\r` byte per file. Fixed in
orbit_mc 1.7.0 (`newline="\n"`), which changes no artifact byte and no hash of
any artifact. Nothing under `results/` is altered by this overlay.

## Immutable bindings

- preregistration commit: `757e365f9f667620c7610663574294c3b71e1f51`
  (`preregister CFT full-orbit wall-loss v4`, 2026-09-03T01:02:45+10:00);
- result commit: `6922a3cf97d261735266aa1a5a0c0c9683e021ca`
  (`record CFT full-orbit wall-loss v4 result`; adds 388 files under
  `results/` and nothing else);
- `results/` tree: `447a5cf79024b85cabbaeb033d719c6d21ab28c0`
  (identical at `6922a3cf` and at this overlay's `HEAD`);
- `results/manifest.json` blob / SHA-256:
  `ccb9e85cc48b72bdb3dde44ae9c8e3187c3ac0f2` /
  `ef3863b0a3ba0a1d74187b05daf81d5d94d3838a7e33ecf82c485dccd162929f`;
- `manifest.lock_byte_sha256`
  `6232324add3ecfa3b4b400026dc15825e5df89ee9388e1ee0f3cca961f493952`,
  `terminal_byte_sha256`
  `6d23c4af9d1645aa8983054b9d8bc3b40d773ae088f7b1987d77cf77736921f1`,
  `transition_log_sha256`
  `477d2c552ba077dd507735f4f4ad505bdf45287855e5ffafe4b99d391b0a18cf`;
- frozen inputs at `757e365f`: `protocol.json` blob
  `a9ed08f3358aadf8a56184410db42a4d43c8f48d`, `authorities.json` blob
  `e954c371218b91485fc2c6a9e72d6a63ec8f68ff`, `shakedown.json` blob
  `7bac8ffebbf66c24f4391b241a7f81db457e686e`;
- executed orbit_mc: package `1.6.0`, source SHA-256 (LF)
  `007c2d51a44d74f989dae6938d10538454886f2e4970f9a9867aaeac8346aa43`,
  `code_identity()` `ab2acba9dd21709477bee6b61f37de881ac47f233118f9d43f607d99ad39e6b4`
  (carried by all nine sealed artifacts and by
  `results/artifacts/orbit-mc-contract.json`).

## 1. Finding: nine sidecars differ from the manifest by end-of-line only

`results/manifest.json` lists 407 entries (20 required directories, 387
files). On this LF checkout 378 files are byte-exact (length and SHA-256).
The nine that are not are exactly the orbit_mc sidecars; for each, the
recorded length is one byte longer and the recorded hash is the hash of the
checkout bytes with `\n` replaced by `\r\n`:

| path | checkout bytes | recorded bytes | checkout sha256 | recorded sha256 | CRLF-recomputed sha256 | match |
| --- | ---: | ---: | --- | --- | --- | --- |
| `artifacts/orbits/enlarged-2N.json.sha256` | 89 | 90 | `2c67ed582bacda0471e29b4674c8cc739e297a5f6bcef9b8b8b936917b0f1eaf` | `ca619dc0852734871ad73f78d82ff8be62d53aba086803ce5b0b819d77eeb773` | `ca619dc0852734871ad73f78d82ff8be62d53aba086803ce5b0b819d77eeb773` | CRLF == recorded |
| `artifacts/orbits/enlarged-4N.json.sha256` | 89 | 90 | `c839bf8e60bd581aedfa564ab90e17d40dc93914ca08d11728e123fa760c0675` | `7fec310bb3394997b48ad1bae0be4af5fb9e5410f3134f61e99801ac70e2f903` | `7fec310bb3394997b48ad1bae0be4af5fb9e5410f3134f61e99801ac70e2f903` | CRLF == recorded |
| `artifacts/orbits/enlarged-N.json.sha256` | 88 | 89 | `1eb388eb0a9e260f95978070ea236eff3f9ed9cc85b3cecce70a39d9acde5773` | `5580b85747dfecd185420c81794c50a47aed5fea6a7092efa94527d5aed679ba` | `5580b85747dfecd185420c81794c50a47aed5fea6a7092efa94527d5aed679ba` | CRLF == recorded |
| `artifacts/orbits/primary-2N.json.sha256` | 88 | 89 | `d3969cfe015bbb7c55e540f61cc048eabcedd6e9dc84fdc05889b0d5205a25ff` | `c82e53f7200669c4c162a45baa21bcedb066bdf05783bd6ff21694d44f91a5a9` | `c82e53f7200669c4c162a45baa21bcedb066bdf05783bd6ff21694d44f91a5a9` | CRLF == recorded |
| `artifacts/orbits/primary-4N.json.sha256` | 88 | 89 | `0d8a4c92feac45946a3a7208bc0e5a943c80394846b5ba713ae554c1da674cfa` | `10fb27fd4b3c0574d7bbaf93afb9fc646b7676d1e12f9b04c8fbd06b25f07dc5` | `10fb27fd4b3c0574d7bbaf93afb9fc646b7676d1e12f9b04c8fbd06b25f07dc5` | CRLF == recorded |
| `artifacts/orbits/primary-N.json.sha256` | 87 | 88 | `425a0bde91dc5c0ca1fc783de98283028c173b1bccc021103e409c26eb3a308c` | `eee592a82536f4d28dd9de766f0b71903c7e68a6fd79c4df9de3a230d4c6614d` | `eee592a82536f4d28dd9de766f0b71903c7e68a6fd79c4df9de3a230d4c6614d` | CRLF == recorded |
| `artifacts/orbits/refined-2N.json.sha256` | 88 | 89 | `d01dbdd0c300aa9e9d23f0ecbdc24fb24f5d4431cd466d4ae1287da1390f0f88` | `45df01e097bb7bd89796c3aac414d9119641fd915f22afc2618daa7d678395c8` | `45df01e097bb7bd89796c3aac414d9119641fd915f22afc2618daa7d678395c8` | CRLF == recorded |
| `artifacts/orbits/refined-4N.json.sha256` | 88 | 89 | `af4e3c8cf8931ad723f326d5e4f5484438e01e3b8880fc3624ab889a615749bf` | `036791dc36ef9b50773020d519bb6c3c869c57da1e114fe156aa641bb238920f` | `036791dc36ef9b50773020d519bb6c3c869c57da1e114fe156aa641bb238920f` | CRLF == recorded |
| `artifacts/orbits/refined-N.json.sha256` | 87 | 88 | `fa720ccc0fdda3c2f79d13072f7e673eeecc01467a8cddf0ce0fbc4e4ac82851` | `d69304226282d65bcff9a20403ee02679d1a735adae88b7edf48d46bdaa265f8` | `d69304226282d65bcff9a20403ee02679d1a735adae88b7edf48d46bdaa265f8` | CRLF == recorded |

Each sidecar is one line, `<64 hex>  <case>-orbit.json` + EOL (87–89 bytes
LF, 88–90 bytes CRLF). The runtime sidecar-of-sidecar
`<path>.sha256.json` records the same CRLF length and hash as the manifest,
so the recording layer hashed the bytes it actually wrote and is internally
consistent; the bytes were simply not the bytes Git preserves.

### What the finding does not touch

- The nine `artifacts/orbits/<case>.json.gz` orbit artifacts (canonical
  compact JSON, no EOL bytes, gzip `mtime=0`) are byte-exact against the
  manifest. Decompressed, each equals `canonical_bytes(json)` exactly, its
  SHA-256 equals the digest stated inside the affected sidecar, and its
  `integrity.payload_sha256` recomputes. The evidence and its content hashes
  are untouched.
- Every other manifest entry is byte-exact on this LF checkout: 378 files =
  the lock, the terminal record, 9 transitions, 17 counters, 16 access
  records, 3 phase records, 9 `.json.gz` orbit artifacts, 81 checkpoints
  (8 batches + 1 partial per case), 9 summaries, 18 case manifests, 3 field
  maps, 3 field-evidence records, 15 top-level artifacts (gates, convergence,
  contract, coupling export, plan, result, authorities, preflights, runtime)
  and the 193 `.sha256.json` runtime sidecars (including the nine that
  describe the affected orbit_mc sidecars; those records are themselves
  byte-exact).
- Restoring CRLF on exactly those nine files in a scratch copy makes the full
  `experiment_runtime` inventory (including every sidecar pair check) equal
  the manifest verbatim; `validate_bundle` then proceeds to the
  `root_identity` check, which binds the bundle to the producing directory
  (volume, file id, final path) by design and is not expected to pass on any
  other checkout. On this checkout without that restoration the first
  refusal is `artifact sidecar schema mismatch:
  artifacts/orbits/enlarged-2N.json.sha256` (the alphabetically first of the
  nine).

## 2. Run-time facts (from the immutable bundle)

- Single attempt: `execution-lock.json` `attempt: 1`, acquired
  `2026-09-02T15:03:40.072889Z` at commit
  `757e365f9f667620c7610663574294c3b71e1f51`, host `DESKTOP-31AD96J`,
  command `python -m experiments.cft_orbit_wall_loss_v4.run execute`,
  `clean_worktree_attested: true`. Terminal counters: `attempt_count 1`,
  `label_access_count 9`, one prebundle/development/assessment access each,
  `expensive_operation_count 13`.
- Transitions 1–9 contiguous: lock-acquired 15:03:40.076Z → cache-prepared →
  prebundle-started → prebundle-completed 15:03:41.004Z →
  development-started → development-accepted 15:03:47.151Z →
  assessment-started 15:03:47.180Z → assessment-accepted 15:14:47.585Z →
  terminal `accepted_result` 15:14:47.662Z. Lock to terminal: 667.6 s
  (assessment 660.4 s: integration 212.6 s, export 440.0 s; 9 parallel
  workers on 24 CPUs; Windows-11-10.0.26200, Python 3.12.10, numpy 2.5.2).
- Gates: all 15 binding checks `true` (`campaign_preflight`,
  `cross_map_probability_convergence`, `earliest_event`, `energy`,
  `field_adapter`, `field_map_convergence`,
  `final_velocity_equals_event_velocity`, `independent_repeats`,
  `manufactured`, `material_quarantine`, `relativistic_phase`,
  `runtime_rotation`, `timestep_probability_convergence`, `wall_endpoint`,
  `zero_incomplete_or_numerical_failures`); `exact_authority_replay` 9/9;
  `passed: true`.
- Validators: 289 invocations, 0 failures.
- Orbits: 9 cases × 512 = 4608; pooled 2962 `wall_hit`, 1646
  `domain_escape`, 0 `reflected`, 0 incomplete or numerical failures
  (`step_limit`, `time_timeout`, `path_timeout`, `field_failure`,
  `nonfinite_state`, `extreme_relativity`, `initial_state_invalid` all 0).
- Per-case wall-hit probability (Wilson 95 %): primary-N 329/512 =
  0.642578 [0.6001, 0.6829]; primary-2N and primary-4N 330/512 = 0.644531
  [0.6021, 0.6848]; refined-N 329/512 = 0.642578; refined-2N and refined-4N
  330/512 = 0.644531; enlarged-N, -2N, -4N 328/512 = 0.640625
  [0.5982, 0.6810].
- Convergence: successive probability changes across N→2N→4N and across
  primary→refined→enlarged are 0.0, 1/512 = 0.001953125 or 2/512 =
  0.00390625 (≈ 0.0039, two orbits) against the preregistered 0.01 gate;
  every adjacent Wilson pair overlaps.
- Energy: `maximum_relative_energy_error` identically `0.0`, 0 of 4608
  orbits above the 1e-10 gate; `final_velocity_m_per_s ==
  event_velocity_m_per_s` for all 4608 orbits (0 mismatches); maximum wall
  endpoint error 4.34e-19 m.
- Magnetic-moment variation (diagnostic, not a gate): min 0.026098, median
  0.140562, max 0.628631 over 4608 orbits; 2786 (60.5 %) above 0.1, 209
  above 0.5.

## 3. Disclosures

1. **Sidecar EOL defect.** Described in §1. Root cause: orbit_mc 1.6.0
   `write_artifact` used `Path.write_text(..., encoding="ascii")` without
   `newline="\n"`. Fixed in orbit_mc 1.7.0
   (`fix(orbit-mc): v1.7 write sidecars with LF newlines for byte
   portability`), together with the same pattern in
   `cft_revival.fields.artifacts._write_canonical_bytes`, a fail-closed
   newline lint over the hash-bound packages, and tests that a written
   artifact and sidecar re-validate byte-exactly after LF normalisation.
   Schema versions are unchanged because no artifact byte changed.
2. **Duplicated `execute` invocation (no side effects).** The agent harness
   re-issued the `execute` command about ten minutes after the real
   invocation. The second process was refused by the Git-common lock
   (`.git/cft-orbit-wall-loss-v4.execution.lock`, created with
   `O_CREAT | O_EXCL` at 2026-09-02T15:03:40.038Z, 34 ms before the runtime
   lock, content `757e365f…`) before any `ExperimentRuntime` was
   constructed, so it wrote nothing. Only the first process (operator log:
   PID 484) executed. This is an operator-disclosed run-time fact; the bundle
   corroborates it with `attempt_count 1`, a single lock record and a
   contiguous nine-event transition log.
3. **`results/` is masked by the ignore rule.** Root `.gitignore` lines 47–48
   (`results/`, `Results/`) match the directory case-insensitively on this
   filesystem (`git check-ignore -v` reports `.gitignore:48:Results/`), so
   the bundle was committed with `git add -f`, exactly as the v3 bundle was
   on its own branch (`09256fb1 record CFT full-orbit wall-loss v3 result`).
   All 388 files are tracked; the tree hash above is what reviewers compare.
4. **Shakedown bundle is non-evidentiary.** The pre-freeze shakedown ran the
   production path on the real P2 fields with a disjoint design into a
   temporary result root
   (`%TEMP%/cft-orbit-wall-loss-v4-shakedown-3ab50ef5c31c-20260902T145753Z-…`),
   recorded `evidentiary: false`, `outcomes_enter_estimand: false`, and its
   outcomes enter no estimand. Only its file hash and design hash are bound
   in `authorities.json`.

## 4. Reviewer reproduction

From `modern/` (PowerShell shown; the script is read-only on `results/`):

```powershell
$env:PYTHONPATH="$PWD\src;$PWD"; $env:PYTHONDONTWRITEBYTECODE='1'
# must print nothing: the bundle is LF on disk
git ls-files --eol -- experiments/cft_orbit_wall_loss_v4/results | Select-String 'w/crlf'
# the table in section 1
python -m experiments.cft_orbit_wall_loss_v4.audit_sidecar_eol --table
# full JSON report; exit code 0 iff passed
python -m experiments.cft_orbit_wall_loss_v4.audit_sidecar_eol
python -m pytest tests/experiments/cft_orbit_wall_loss_v4/test_posthoc_audit.py -q
```

The report must show `counts = {byte_exact: 378, eol_only: 9, mismatch: 0}`,
`eol_only_paths_are_exactly_expected: true`,
`runtime_sidecars_agree_with_manifest: true`, `orbit_evidence_intact: true`
and `passed: true`. The test module additionally re-derives every §2 figure
from the bundle, checks the §1 table against a live recomputation, proves the
script left `results/` byte-identical, and performs the scratch-copy CRLF
restoration to show the full inventory then equals the manifest.

## 5. Physics interpretation (interpretation, not evidence)

*This section interprets the accepted numbers; it is not part of the
preregistered estimand and carries no evidentiary weight.* Zero reflections
in all 4608 orbits means the magnetic-mirror picture does not apply to this
field: no launched electron turned around before reaching a boundary. The
estimand is therefore a wall-hit versus axial-escape split that is decided
almost entirely by the launch cell. Cells 2 and 3 (z = 9.5 and 15.5 mm)
lose 100 % of launches to the dielectric in both directions (2304/2304);
cell 4 (z = 21.5 mm, beyond the 18 mm end of the wall) escapes 100 %
(1152/1152, through the z = 23 mm exit plane or radially in the divergent
section, which the protocol counts as escape); cell 1 (z = 3.5 mm) is
direction-dependent: the −1 direction is all lost (576/576) while the +1
direction mostly escapes through the injector end at z = 1 mm (494/576, with
82 wall hits). The pooled 0.643 is thus an equal-weight design average of a
bimodal per-cell result, not a physical loss fraction of any real electron
population. About 43 % of orbits (1967/4608) terminate through the
tolerance-close snapping path, i.e. they end within one event tolerance of a
boundary rather than by an interpolated crossing, which is the main path of
this campaign and not an edge case. Magnetic moment varying by more than
10 % in 60 % of orbits is non-adiabatic cusp-transit physics for electrons
whose gyroradius approaches the field scale length; numerical fidelity is
established by the exact energy gate, the rotation bound, the manufactured
orders and CPU/CUDA parity, not by μ.

## 6. Overlay contents

This audit adds only `POSTHOC_AUDIT.md`, `audit_sidecar_eol.py` and
`tests/experiments/cft_orbit_wall_loss_v4/test_posthoc_audit.py`. No file
under `results/`, no frozen preregistration file and no orbit_mc source is
modified by it; the results tree hash is asserted unchanged by the test.
