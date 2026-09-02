"""Axisymmetric (r,z) electrostatic PIC-MCC for the CFT discharge channel.

Development/screening fidelity level.  See ``modern/docs/workstreams/pic2d-*.md``
for the formulation, simplification list, and claim boundary.
"""

from .models import (
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PIC2DConvergenceError,
    PIC2DDeviceError,
    PIC2DError,
    PIC2DStabilityError,
    PIC2DValidationError,
    ParticleArrays,
    PoissonConfig2D,
    Species2D,
    StabilityLimits,
    StabilityReport2D,
    electron_species,
    require_stable,
    stability_report,
    xenon_ion_species,
)

__version__ = "0.1.0"

__all__ = [
    "BoundaryPotentials",
    "ChannelGeometry",
    "Grid2D",
    "PIC2DConvergenceError",
    "PIC2DDeviceError",
    "PIC2DError",
    "PIC2DStabilityError",
    "PIC2DValidationError",
    "ParticleArrays",
    "PoissonConfig2D",
    "Species2D",
    "StabilityLimits",
    "StabilityReport2D",
    "__version__",
    "electron_species",
    "require_stable",
    "stability_report",
    "xenon_ion_species",
]
