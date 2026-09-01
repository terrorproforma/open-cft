# L0 surrogate v4 development log

## 2026-09-02 — preregistration

- Inherited the v2/v3 scientific protocol without threshold/model changes.
- Added Git commit binding before real-data access: observed HEAD, commit
  existence, remote containment, exact preregistration subject/path isolation,
  unchanged protocol blobs, and clean v4 working paths.
- Added synthetic repositories covering valid binding, nonexistent/wrong SHA,
  dirty protocol files, and unrelated intervening commits.
