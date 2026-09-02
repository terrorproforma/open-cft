"""Acceptance evidence checks independent of the FEM solve."""

from __future__ import annotations

from math import isfinite

from .models import FEMValidationError


def evaluate_phase_matched_domain_expansion(
    studies: tuple[dict[str, object], ...],
    *,
    required_padding_factors: tuple[float, ...] = (0.5, 1.0, 1.5),
    maximum_qoi_relative_change: float = 0.01,
) -> dict[str, object]:
    """Evaluate Robin truncation only when local resolution is phase matched."""

    if not studies:
        raise FEMValidationError("domain-expansion studies are required")
    try:
        padding_values = tuple(float(item["padding_factor"]) for item in studies)
    except (KeyError, TypeError, ValueError, OverflowError) as error:
        raise FEMValidationError("domain-expansion padding is invalid") from error
    if (
        any(not isfinite(value) or value <= 0.0 for value in padding_values)
        or padding_values != required_padding_factors
    ):
        raise FEMValidationError("domain-expansion padding phases are incomplete")
    if (
        not isfinite(maximum_qoi_relative_change)
        or maximum_qoi_relative_change <= 0.0
    ):
        raise FEMValidationError("domain-expansion QoI limit must be positive")
    qoi_keys = set(studies[0]["qois_bz_t"])
    h_keys = set(studies[0]["qoi_h_m"])
    if not qoi_keys or qoi_keys != h_keys:
        raise FEMValidationError("domain-expansion QoI and local-h keys differ")
    reference_h = studies[0]["qoi_h_m"]
    reference_local_h = studies[0]["local_h_m"]
    previous_domain = None
    for study in studies:
        if (
            set(study["qois_bz_t"]) != qoi_keys
            or set(study["qoi_h_m"]) != h_keys
            or set(study["local_h_m"]) != set(reference_local_h)
        ):
            raise FEMValidationError("domain-expansion evidence keys differ")
        domain = study["domain"]
        try:
            r_min = float(domain["r_min_m"])
            r_max = float(domain["r_max_m"])
            z_min = float(domain["z_min_m"])
            z_max = float(domain["z_max_m"])
        except (KeyError, TypeError, ValueError, OverflowError) as error:
            raise FEMValidationError("domain-expansion extent is invalid") from error
        if (
            any(not isfinite(value) for value in (r_min, r_max, z_min, z_max))
            or r_min < 0.0
            or r_max <= r_min
            or z_max <= z_min
        ):
            raise FEMValidationError("domain-expansion extent is invalid")
        if previous_domain is not None and not (
            r_min == previous_domain[0]
            and r_max > previous_domain[1]
            and z_min < previous_domain[2]
            and z_max > previous_domain[3]
        ):
            raise FEMValidationError("domain-expansion extents are not nested")
        previous_domain = (r_min, r_max, z_min, z_max)
        for key in h_keys:
            left = float(reference_h[key])
            right = float(study["qoi_h_m"][key])
            qoi = float(study["qois_bz_t"][key])
            if (
                not isfinite(left)
                or not isfinite(right)
                or left <= 0.0
                or right <= 0.0
                or not isfinite(qoi)
            ):
                raise FEMValidationError("domain-expansion QoI/local h is invalid")
            if abs(left - right) / max(abs(left), abs(right), 1.0e-300) > 1.0e-12:
                raise FEMValidationError(
                    "domain-expansion source/QoI local h is not phase matched"
                )
        for key in reference_local_h:
            left = float(reference_local_h[key])
            right = float(study["local_h_m"][key])
            if (
                not isfinite(left)
                or not isfinite(right)
                or left <= 0.0
                or right <= 0.0
                or abs(left - right) / max(abs(left), abs(right), 1.0e-300)
                > 1.0e-12
            ):
                raise FEMValidationError(
                    "domain-expansion source/QoI local h is not phase matched"
                )
    changes = []
    for left, right in zip(studies, studies[1:]):
        changes.append(
            {
                key: abs(
                    float(right["qois_bz_t"][key])
                    - float(left["qois_bz_t"][key])
                )
                / max(
                    abs(float(right["qois_bz_t"][key])),
                    abs(float(left["qois_bz_t"][key])),
                    1.0e-300,
                )
                for key in sorted(qoi_keys)
            }
        )
    return {
        "phase_matched": True,
        "successive_qoi_relative_changes": changes,
        "maximum_qoi_relative_change": maximum_qoi_relative_change,
        "passed": all(
            value < maximum_qoi_relative_change
            for change in changes
            for value in change.values()
        ),
    }
