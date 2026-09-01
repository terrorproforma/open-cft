# L0 surrogate v9 posthoc audit correction

## Corrected status

`NUMERICAL_PASS_SCIENTIFIC_SURROGATE_FAIL`

V9's frozen numerical and reproducibility gates passed exactly as recorded.
Those immutable artifacts are not altered by this overlay. However, V9 does
**not** establish learned-surrogate or accelerated-surrogate evidence.

## Scientific claim correction

The selected model's analytic leading term is the complete accepted L0 target
for both reported outputs:

- `thrust = mass_flow * ionized * [1 + (sqrt(2)-1)*double_share] *
  sqrt(2*e*V/m_Xe) * axial_divergence`;
- `Isp = ionized * [1 + (sqrt(2)-1)*double_share] *
  sqrt(2*e*V/m_Xe) * axial_divergence / g0`.

The residual GP therefore learns only binary64 regrouping/roundoff, explaining
the approximately `1e-16` normalized errors and `1e-15` normalized widths. This
is an algebraic reimplementation check with a negligible GP correction, not
evidence that a useful unknown response was learned from 64 observations.

The only permitted public claim is exactly:

> Algebraic implementation plus negligible GP matched fresh same-domain evaluations.

No L0 surrogate-learning, active-learning, acceleration, independent-physics,
or physical-accuracy claim is permitted.

## Preflight information-barrier correction

The preregistered preflight called the accepted L0 evaluator for 36 identity
points while recording `physics_label_access_count: 0`. Its 32 deterministic
design points were V9 design indices 0–31. Index 24 belongs to the frozen
`final-calibration` role, so preflight accessed one final-calibration L0 label
before method freeze. It accessed no assessment-role index in that set.

This does not change the numerical artifact identity, but it invalidates the
claimed final-calibration information barrier and independently prevents a
clean prospective learned-surrogate interpretation.

## Diagnostic speed probe

A posthoc diagnostic-only probe used 4,096 deterministic five-coordinate rows,
two warmups, and seven trials. Median direct algebra evaluation was
`0.0012726998 s`; median selected residual-GP evaluation was `0.6128954003 s`,
or `481.57x` slower. The probe specification hash is
`b7a55b991d26b67b447bac60e35797d8f6f2c569e0056f85f59bb61b9bb9b9f9`.
These machine-dependent timings support no general performance ratio, but they
rule out a speed claim for this diagnostic implementation.

## Immutable bindings

- preregistration commit:
  `a4fc39dd46e1b93d775eab1456d0d8693ff4f4f9`;
- result commit:
  `71712057f93a73ba8d5cc82f50a111023de5077e`;
- preregistration blob:
  `37b7387ef5e7b4d8fea0d3693e9bbf7c46b91838`;
- result tree:
  `0fd24303cf28ecb78e66e19cfc2577801ec053ce`;
- run-manifest blob/hash:
  `1aff92987bb3a6df7e8919fa3d8e4afc5b11879d` /
  `c8170ed1264dfced07241feb93f9b6b9ae34946899c556bf8998b4f8a66e3944`;
- final-assessment blob/hash:
  `c1f69d6bc2a46390963c876e221191e77ba58fc8` /
  `28da1e151cecace8ab61f4f943d65ae40a1f38fb41bb4b417cc167ba9d1ee6b4`;
- preflight blob/hash:
  `20dfba6ea146a6e23ac8ca8a863f2287f9db48f1` /
  `fffa14e564767d6e6246e518fb19f7ed1c645f0ae9a793e45e85643ab097d7f3`.

## Next work

The next scientifically useful surrogate is an L1 discrepancy surrogate for
field-resolved/higher-fidelity corrections relative to the known L0 equations.
There should be no L0 surrogate v10 intended to relearn this closed-form target.
