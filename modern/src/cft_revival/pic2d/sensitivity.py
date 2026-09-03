"""v1.4 sensitivity hooks (default OFF): Bohm-type anomalous scattering and the SEE scaffold.

Both exist so that the preregistered campaign proposal can reference implemented, unit-tested
switches (literature review ``pic-mcc-blockers.md``, sections 4.2 and 4.3).  Neither is part of
the v1.4 development run.

**Anomalous (Bohm) scattering** -- ``AnomalousCollisionConfig(alpha)``: every electron suffers an
isotropic, speed-preserving velocity redirection with probability ``1 - exp(-alpha omega_ce dt)``
per step, ``omega_ce = e |B| / m_e`` gathered at the particle position.  ``nu_an = alpha
omega_ce`` is the Bohm-type effective collision frequency used as a transport bracket by Szabo
2001 / Szabo et al. 2014 and Brandt et al. 2016 (who impose ``D = 0.4 kT_e / eB`` via velocity
rotation) and inferred as ``nu_B ~ omega_c / 16`` for the cylindrical Hall thruster by Smirnov,
Raitses and Fisch 2004.  The review recommends the bracket ``alpha in {1/64, 1/16}`` as a
reported, non-binding sensitivity block: an axisymmetric electrostatic code excludes the
azimuthal drift instability by construction, so this term is a bracket, not a model.  The
scattering is a pure velocity rotation: kinetic energy is conserved to round-off and the
energy ledger is untouched; the collision count is tallied (``cumulative["anomalous"]``).

**Secondary electron emission scaffold** -- ``SEEConfig``: the Vaughan (1989) yield curve with
the boron-nitride parameter set the review points at (Dunaevsky, Raitses and Fisch 2003; Tondu,
Belhaj and Inguimbert 2011 measured HET-grade BN/Al2O3 ceramics; the numbers below are the
usual Hall-thruster PIC values for BN and are marked *provisional* until digitised from those
papers), plus the Hobbs-Wesson (1967) space-charge limit ``delta -> 1`` as a fail-closed check.
``enabled=True`` is refused by ``PIC2DConfig`` (emission is not implemented in v1.4); the
scaffold computes the *virtual* yield the wall would have at the observed electron impact
energies, per wall column, so the campaign can report where the cusp sheaths would go
space-charge-limited (Campanell, Khrabrov and Kaganovich 2012) before emission is switched on.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .models import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, PIC2DValidationError

BOHM_ALPHA_BRACKET = (1.0 / 64.0, 1.0 / 16.0)


@dataclass(frozen=True, slots=True)
class AnomalousCollisionConfig:
    """Bohm-type anomalous scattering, ``nu_an = alpha omega_ce`` (0 < alpha <= 1)."""

    alpha: float

    def __post_init__(self) -> None:
        if not isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise PIC2DValidationError("anomalous alpha must be in (0, 1]")

    def to_dict(self) -> dict[str, Any]:
        return {"model": "bohm_isotropic_scattering", "alpha": self.alpha}


def bohm_collision_probability(alpha: float, b_magnitude_t: np.ndarray | float, dt_s: float) -> np.ndarray:
    """``1 - exp(-alpha omega_ce dt)`` at the local field magnitude (exact Poisson probability)."""

    omega_ce = ELEMENTARY_CHARGE_C * np.abs(np.asarray(b_magnitude_t, dtype=np.float64)) / ELECTRON_MASS_KG
    return -np.expm1(-alpha * omega_ce * dt_s)


def isotropic_redirect(speed: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly random direction with the given speed (same map as the MCC isotropic scatter)."""

    cos_t = 1.0 - 2.0 * u1
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    phi = 2.0 * np.pi * u2
    return speed * sin_t * np.cos(phi), speed * sin_t * np.sin(phi), speed * cos_t


def apply_bohm_scattering(
    alpha: float, vr: np.ndarray, vt: np.ndarray, vz: np.ndarray, b_magnitude_t: np.ndarray, dt_s: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """CPU reference: redirect the electrons selected with the local Bohm probability; returns (vr, vt, vz, count)."""

    probability = bohm_collision_probability(alpha, b_magnitude_t, dt_s)
    hit = rng.random(vr.size) < probability
    count = int(hit.sum())
    if count == 0:
        return vr, vt, vz, 0
    speed = np.sqrt(vr[hit] ** 2 + vt[hit] ** 2 + vz[hit] ** 2)
    new_r, new_t, new_z = isotropic_redirect(speed, rng.random(count), rng.random(count))
    vr, vt, vz = vr.copy(), vt.copy(), vz.copy()
    vr[hit], vt[hit], vz[hit] = new_r, new_t, new_z
    return vr, vt, vz, count


# Vaughan (1989) parameters for hexagonal boron nitride as used in Hall-thruster PIC.
# PROVISIONAL: to be digitised from Dunaevsky et al. 2003 (Fig. yields of BN) and Tondu et al.
# 2011 before any SEE-on case; the first crossover implied here is ~35 eV.
BN_VAUGHAN: dict[str, float] = {"delta_max": 2.9, "energy_max_ev": 350.0, "energy_threshold_ev": 12.5}


@dataclass(frozen=True, slots=True)
class SEEConfig:
    """SEE scaffold: Vaughan yield for a declared material; emission itself is not implemented (fail closed when enabled)."""

    enabled: bool = False
    material: str = "BN"
    delta_max: float = BN_VAUGHAN["delta_max"]
    energy_max_ev: float = BN_VAUGHAN["energy_max_ev"]
    energy_threshold_ev: float = BN_VAUGHAN["energy_threshold_ev"]
    emission_temperature_ev: float = 2.0
    space_charge_limit_yield: float = 1.0      # Hobbs-Wesson 1967: the classical sheath ceases to exist at delta -> 1

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise PIC2DValidationError("see.enabled must be a bool")
        for name in ("delta_max", "energy_max_ev", "energy_threshold_ev", "emission_temperature_ev", "space_charge_limit_yield"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise PIC2DValidationError(f"see.{name} must be positive")
        if self.energy_threshold_ev >= self.energy_max_ev:
            raise PIC2DValidationError("see.energy_threshold_ev must be below energy_max_ev")

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled, "material": self.material, "yield_model": "vaughan_1989",
            "delta_max": self.delta_max, "energy_max_ev": self.energy_max_ev, "energy_threshold_ev": self.energy_threshold_ev,
            "emission_temperature_ev": self.emission_temperature_ev, "space_charge_limit_yield": self.space_charge_limit_yield,
            "parameter_status": "provisional (Hall-thruster PIC values for BN; digitise Dunaevsky 2003 / Tondu 2011 before use)",
        }

    def yield_at(self, energy_ev: np.ndarray | float, incidence_angle_rad: np.ndarray | float = 0.0) -> np.ndarray:
        return vaughan_yield(energy_ev, incidence_angle_rad, delta_max=self.delta_max, energy_max_ev=self.energy_max_ev,
                             energy_threshold_ev=self.energy_threshold_ev)


def vaughan_yield(
    energy_ev: np.ndarray | float, incidence_angle_rad: np.ndarray | float = 0.0, *,
    delta_max: float, energy_max_ev: float, energy_threshold_ev: float, smoothness: float = 1.0,
) -> np.ndarray:
    """Vaughan 1989 secondary-electron yield ``delta(E, theta)``.

    ``v = (E - E_0) / (E_max(theta) - E_0)``; ``delta = delta_max(theta) (v e^(1-v))^k`` with
    ``k = 0.56`` for ``v < 1`` and ``k = 0.25`` for ``1 <= v <= 3.6``; beyond ``v = 3.6`` the
    ``1.125 / v^0.35`` tail; zero below the threshold ``E_0``.  Angular dependence:
    ``E_max(theta) = E_max (1 + k_s theta^2 / (2 pi))``, ``delta_max(theta) = delta_max
    (1 + k_s theta^2 / pi)`` with the surface-smoothness factor ``k_s`` (1 = typical).
    """

    energy = np.asarray(energy_ev, dtype=np.float64)
    theta = np.asarray(incidence_angle_rad, dtype=np.float64)
    e_max = energy_max_ev * (1.0 + smoothness * theta**2 / (2.0 * np.pi))
    d_max = delta_max * (1.0 + smoothness * theta**2 / np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (energy - energy_threshold_ev) / (e_max - energy_threshold_ev)
        k = np.where(v < 1.0, 0.56, 0.25)
        core = d_max * np.power(np.maximum(v, 0.0) * np.exp(1.0 - np.maximum(v, 0.0)), k)
        tail = d_max * 1.125 / np.power(np.maximum(v, 1e-300), 0.35)
        out = np.where(v <= 3.6, core, tail)
    return np.where(energy > energy_threshold_ev, out, 0.0)


def first_crossover_ev(config: SEEConfig) -> float:
    """Energy where ``delta = 1`` on the rising branch (bisection on the Vaughan curve)."""

    if config.delta_max <= 1.0:
        return float("inf")
    lo, hi = config.energy_threshold_ev, config.energy_max_ev
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if float(config.yield_at(mid)) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def virtual_wall_yield(
    config: SEEConfig, wall_electron_flux: np.ndarray, wall_electron_energy_flux_j: np.ndarray,
) -> dict[str, Any]:
    """Diagnostic for the scaffold: Vaughan yield at the mean electron impact energy per wall column.

    ``wall_electron_flux`` and ``wall_electron_energy_flux_j`` are the per-column (axial) wall
    electron particle and energy fluxes from the window maps (any common normalisation).  Reports
    the per-column yield, its flux-weighted mean, and the columns above the Hobbs-Wesson limit.
    """

    flux = np.asarray(wall_electron_flux, dtype=np.float64)
    energy_flux = np.asarray(wall_electron_energy_flux_j, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_energy_ev = np.where(flux > 0.0, energy_flux / np.maximum(flux, 1e-300) / ELEMENTARY_CHARGE_C, 0.0)
    delta = np.where(flux > 0.0, config.yield_at(mean_energy_ev), 0.0)
    total = float(flux.sum())
    return {
        "material": config.material,
        "mean_impact_energy_ev_per_column": mean_energy_ev.tolist(),
        "yield_per_column": delta.tolist(),
        "flux_weighted_yield": float(np.sum(delta * flux) / total) if total > 0.0 else None,
        "columns_above_space_charge_limit": int(np.sum((delta >= config.space_charge_limit_yield) & (flux > 0.0))),
        "first_crossover_ev": first_crossover_ev(config),
        "note": "virtual yield: emission is OFF (scaffold); columns at or above the limit would need the Hobbs-Wesson cap",
    }


__all__ = [
    "BN_VAUGHAN",
    "BOHM_ALPHA_BRACKET",
    "AnomalousCollisionConfig",
    "SEEConfig",
    "apply_bohm_scattering",
    "bohm_collision_probability",
    "first_crossover_ev",
    "isotropic_redirect",
    "vaughan_yield",
    "virtual_wall_yield",
]
