"""Deprecated same-z v2 screening proxy.

This module is intentionally separate from the accepted v3 builder.  Its
results are sensitivity-screening diagnostics, not plasma coupling records.
"""

from __future__ import annotations

import warnings
from datetime import datetime

from .models import (
    AcceptedFieldEvidence,
    CouplingRecord,
    TopologyPolicy,
    UncertaintyModel,
)
from .records import _build_screening_proxy_record


def build_screening_proxy(
    evidence: AcceptedFieldEvidence,
    *,
    wall_radius_m: float,
    topology_policy: TopologyPolicy = TopologyPolicy(),
    uncertainty_model: UncertaintyModel = UncertaintyModel(),
    reference_time_utc: datetime | None = None,
) -> CouplingRecord:
    """Return deprecated same-z axis/wall diagnostics with no acceptance authority."""

    warnings.warn(
        "same-z v2 axis/wall comparison is a deprecated screening_proxy and "
        "cannot create an accepted plasma coupling record",
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_screening_proxy_record(
        evidence,
        wall_radius_m=wall_radius_m,
        topology_policy=topology_policy,
        uncertainty_model=uncertainty_model,
        reference_time_utc=reference_time_utc,
    )
