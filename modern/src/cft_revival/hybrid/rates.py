"""Maxwellian rate coefficients from the PIC's hash-bound xenon cross sections.

The per-cell electron fluid needs ``<sigma v>(T_e)`` for elastic, excitation and ionisation
collisions.  They are computed from the SAME ``XenonCrossSections`` tables the PIC-MCC samples
(LXCat Biagi-v7.1, payload SHA-256 bound), resampled on the PIC's uniform energy grid
(``UniformSigmaTable``: 0.05 eV steps to 2000 eV), and integrated against the Maxwellian energy
distribution ``f(E) = 2 sqrt(E / pi) T^-3/2 exp(-E / T)`` (E, T in eV).  The result is a
tabulated ``RateTable`` on a log-spaced temperature grid with log-log interpolation.

The fluid rates are therefore the Maxwellian average of the PIC's cross sections; a PIC
electron population with a hot tail ionises more per electron than a Maxwellian at its
density-weighted temperature.  That difference is a declared model discrepancy of L2 v2, not
a cross-section difference.
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, pi
from typing import Any

import numpy as np

from ..pic2d.mcc import (
    PROCESS_ORDER,
    UniformSigmaTable,
    XenonCrossSections,
    electron_speed_from_energy,
)
from .models import HybridValidationError

RATE_PROCESSES = PROCESS_ORDER  # ("elastic", "excitation", "ionization")


@dataclass(frozen=True, slots=True)
class RateTable:
    """``k_process(T_e)`` in m^3/s on a log-spaced temperature grid (eV)."""

    temperature_ev: np.ndarray            # (n_t,)
    rates_m3_per_s: np.ndarray            # (3, n_t) in PROCESS_ORDER
    thresholds_ev: tuple[float, float, float]
    cross_section_payload_sha256: str
    uniform_table_sha256: str
    energy_step_ev: float
    energy_max_ev: float

    def __post_init__(self) -> None:
        t = np.asarray(self.temperature_ev, dtype=np.float64)
        k = np.asarray(self.rates_m3_per_s, dtype=np.float64)
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0) or t[0] <= 0.0:
            raise HybridValidationError("rate table temperatures must be positive and strictly increasing")
        if k.shape != (3, t.size) or not np.isfinite(k).all() or np.any(k < 0.0):
            raise HybridValidationError("rate table must be a finite non-negative (3, n_t) array")
        object.__setattr__(self, "temperature_ev", t)
        object.__setattr__(self, "rates_m3_per_s", k)

    def rate(self, process: str, temperature_ev: np.ndarray | float) -> np.ndarray:
        """Log-log interpolation in T (clamped to the table range; zero where the table is zero)."""

        index = RATE_PROCESSES.index(process)
        t = np.clip(np.asarray(temperature_ev, dtype=np.float64), self.temperature_ev[0], self.temperature_ev[-1])
        if not np.isfinite(t).all():
            raise HybridValidationError("temperature must be finite")
        k = self.rates_m3_per_s[index]
        positive = k > 0.0
        if not positive.any():
            return np.zeros_like(t)
        log_k = np.where(positive, np.log(np.where(positive, k, 1.0)), -np.inf)
        # interpolate log k on log T; below the first positive entry the rate is treated as zero
        first = int(np.argmax(positive))
        out = np.interp(np.log(t), np.log(self.temperature_ev[first:]), log_k[first:], left=-np.inf)
        return np.where(np.isfinite(out), np.exp(out), 0.0)

    def sha256(self) -> str:
        digest = sha256()
        digest.update(self.cross_section_payload_sha256.encode())
        digest.update(np.ascontiguousarray(self.temperature_ev).tobytes())
        digest.update(np.ascontiguousarray(self.rates_m3_per_s).tobytes())
        return digest.hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "processes": list(RATE_PROCESSES),
            "thresholds_ev": list(self.thresholds_ev),
            "temperature_grid_ev": [float(self.temperature_ev[0]), float(self.temperature_ev[-1]), int(self.temperature_ev.size)],
            "cross_section_payload_sha256": self.cross_section_payload_sha256,
            "uniform_table_sha256": self.uniform_table_sha256,
            "energy_step_ev": self.energy_step_ev,
            "energy_max_ev": self.energy_max_ev,
            "rate_table_sha256": self.sha256(),
            "distribution": "Maxwellian f(E) = 2 sqrt(E/pi) T^-3/2 exp(-E/T); trapezoid on the PIC uniform sigma grid",
        }


def maxwellian_energy_pdf(energy_ev: np.ndarray, temperature_ev: float) -> np.ndarray:
    e = np.asarray(energy_ev, dtype=np.float64)
    t = float(temperature_ev)
    return 2.0 * np.sqrt(e / pi) * t ** -1.5 * np.exp(-e / t)


def maxwellian_rate_m3_per_s(energy_ev: np.ndarray, sigma_m2: np.ndarray, temperature_ev: float) -> float:
    """``<sigma v>`` by trapezoid integration on the given energy grid (a brute-force oracle for tests)."""

    if not isfinite(temperature_ev) or temperature_ev <= 0.0:
        raise HybridValidationError("temperature must be positive")
    speed = electron_speed_from_energy(energy_ev)
    integrand = np.asarray(sigma_m2, dtype=np.float64) * speed * maxwellian_energy_pdf(energy_ev, temperature_ev)
    return float(np.trapezoid(integrand, energy_ev))


def build_rate_table(
    cross_sections: XenonCrossSections,
    *,
    temperature_min_ev: float = 0.2,
    temperature_max_ev: float = 100.0,
    temperature_points: int = 225,
    energy_step_ev: float = 0.05,
    energy_max_ev: float = 2000.0,
) -> RateTable:
    """Tabulate the Maxwellian rates of every process on a log-spaced temperature grid."""

    if not 0.0 < temperature_min_ev < temperature_max_ev or temperature_points < 2:
        raise HybridValidationError("temperature grid parameters are inconsistent")
    table = UniformSigmaTable.build(cross_sections, energy_step_ev=energy_step_ev, energy_max_ev=energy_max_ev)
    energy = np.arange(table.point_count, dtype=np.float64) * table.energy_step_ev
    temperatures = np.geomspace(temperature_min_ev, temperature_max_ev, temperature_points)
    if np.exp(-energy_max_ev / temperature_max_ev) > 1e-6:
        raise HybridValidationError("the energy grid does not contain the Maxwellian tail at the maximum temperature")
    rates = np.empty((3, temperatures.size), dtype=np.float64)
    for row in range(3):
        sigma = table.table_m2[row]
        for column, t in enumerate(temperatures):
            rates[row, column] = maxwellian_rate_m3_per_s(energy, sigma, float(t))
    return RateTable(
        temperatures, rates, table.thresholds_ev, cross_sections.payload_sha256, table.sha256(),
        float(table.energy_step_ev), float(table.energy_max_ev),
    )


__all__ = ["RATE_PROCESSES", "RateTable", "build_rate_table", "maxwellian_energy_pdf", "maxwellian_rate_m3_per_s"]
