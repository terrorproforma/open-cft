# L0 surrogate v8

V8 is a same-domain prospective validation informed openly by V7's valid failed
result. It uses fresh exact coordinates and does not claim untouched coarse
spatial regions or physical accuracy.

The protocol preserves V7's group-count-ranked conformal target: simultaneous
coverage of all rows in a future exchangeable spatial group. Row coverage has a
lower sanity gate only; values above 0.95 are diagnostic because group-max
intervals intentionally overcover rows. Efficiency is separately gated at
normalized median full width 0.25 and p90 full width 0.40.

Raw and physics-informed ARD Matérn-5/2 feature GPs at 128, 160 and 224 rows are
compared using development/method labels only. The smallest passing budget is
chosen; family ties use method-group OOD worst error. Method freezes before
final calibration, which freezes before exactly-once assessment.
