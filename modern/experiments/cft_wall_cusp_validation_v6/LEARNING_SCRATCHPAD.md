# Learning scratchpad

## 2026-09-02 — v6 preregistration construction

- `COMMITTED`: this scratchpad is experiment-local evidence.
- [user] Preserve v1-v5 and exclude `wcval-v5-s05-p0-r0-neg` plus its exact
  coordinate; keep the six-cusp scientific requirement unchanged.
- [self] A value accepted directly by `RunContext.write_json` must never first
  pass through runtime `normalize`; its reserved envelopes are an encoded
  representation, not domain JSON.
- [self] Tuple and dataclass callback values are now converted to lists and
  named dictionaries by `_plain_domain_json`; boundary nulls and ambiguous
  assessments have explicit serializers.
- [self] Test callback shapes, not only public dataclass types: resolved,
  ambiguous zero-cell/orbit, boundary-nonempty, and full rejection summaries
  all use the same `_write_callback_json` path as assessment.
- [tool] Foundation `b46e263` fixes inventory sorting globally. Same-stem
  file/directory names remain avoidable, so production fields use a distinct
  `preflight/production-fields/` directory.
- [tool] Git line-ending conversion would invalidate canonical field bytes;
  result-local `.gitattributes` therefore declares `* -text` and is an approved
  runtime placeholder.
- [self] A numerical rejection must write its outcome immediately before the
  next case. Deferred aggregate promotion work must not erase atomic evidence.
- [user] After preregistration, execute once from clean detached HEAD and commit
  that exact valid terminal bundle without patching or rerunning.
