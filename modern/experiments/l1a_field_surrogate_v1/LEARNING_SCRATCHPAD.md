# L1a field-surrogate v1 learning scratchpad

## Before execution

- A 41x73 solve requires a 1.6 mm coarse radial source smear to preserve the
  accepted minimum-two-grid-spacing source contract. Signed ampere-turns and
  geometry identity remain paired; the smear discrepancy is intentionally
  learned rather than hidden.
- Direct nested-grid prolongation is a strong baseline, so the experiment must
  demonstrate nontrivial residual learning against development labels.
- Assessment remains valid even if a gate fails; no threshold or model may be
  changed after the protocol commit.
- The configured learning-scratchpad-loop and devlog-loop skills are absent
  from this agent catalog, so both persistent loop artifacts are updated
  directly, following the prior accepted experiment convention.

## After execution

- Pending exactly-once results.
