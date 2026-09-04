"""v1.4 sensitivity hooks (default OFF): Bohm-type anomalous scattering and the SEE scaffold.

Both exist so that the preregistered campaign proposal can reference implemented, unit-tested
switches (literature review ``pic-mcc-blockers.md``, sections 4.2 and 4.3).  Neither is part of
the v1.4 development run.

**Anomalous (Bohm) scattering** -- ``AnomalousCollisionConfig(alpha, model)``: every electron
suffers a speed-preserving velocity change with probability ``1 - exp(-alpha omega_ce dt)`` per
step, ``omega_ce = e |B| / m_e`` gathered at the particle position (an exact Poisson process
with the effective collision frequency ``nu_an = alpha omega_ce``, applied wherever the electron
is - the probability follows the local field, so it vanishes at the nulls and is largest at the
cusps).  ``nu_an = alpha omega_ce`` is the Bohm-type effective collision frequency used as a
transport bracket by Szabo 2001 / Szabo et al. 2014 and Brandt et al. 2016 and inferred as
``nu_B ~ omega_c / 16`` for the cylindrical Hall thruster by Smirnov, Raitses and Fisch 2004.
Two event models (``ANOMALOUS_MODELS``):

* ``bohm_isotropic_scattering`` (v1.4): the velocity is redirected uniformly on the sphere
  (the MCC isotropic map).  The parallel speed is randomised too, so beside the cross-field
  step the event also scatters the pitch angle (mirror trapping / loss cone) - a bracket of
  the literature model, not the model.
* ``bohm_perpendicular_rotation`` (v2.1.0): the velocity is rotated about the local ``B``
  direction by a uniformly random angle (Rodrigues), so ``|v|`` AND ``v_parallel`` are
  unchanged to round-off and only the gyro-phase is reset - the guiding centre jumps by
  ``2 r_L sin(phi / 2)`` in the plane perpendicular to ``B``.  This is the model Brandt et
  al. 2016 (doi:10.2322/tastj.14.Pb_235, p. Pb_237) describe: "only the component of the
  velocity vector perpendicular to the local magnetic field direction is rotated, to ensure
  that the speed of the electrons along the magnetic field lines does not change", electrons
  selected at random at a rate set by the local flux density.

Both models give the same cross-field diffusion coefficient for a Maxwellian: the velocity
autocorrelation of a gyrating electron whose gyro-phase is reset at Poisson rate ``nu`` is
``<v_x^2> exp(-nu t) cos(omega_ce t)``, so ``D_perp = (kT_e / m_e) nu / (nu^2 + omega_ce^2) =
(kT_e / eB) alpha / (1 + alpha^2)`` (Green-Kubo; the random-walk picture ``D = <step^2> nu / 4
= r_L^2 nu / 2`` is its small-alpha limit).  ``alpha = 1/16`` is the classical Bohm value;
Brandt's ``D_perp = 0.4 kT_e / eB`` read as ``nu = 0.4 omega_ce`` gives the exact factor 0.345.
The event is elastic: kinetic energy is conserved to round-off, the energy ledger carries no
term for it (correct for an elastic process), the axial momentum it hands to the "turbulent
field" is tallied in ``pz_collisions`` like the MCC elastic events, and the event count is
tallied (``cumulative["anomalous"]``).  It is NOT part of the MCC null-collision budget: it is
a separate exact-Poisson process drawn from its own seed stream after the push and before the
MCC, which is statistically equivalent to a joint budget to ``O(nu_an dt x nu_mcc dt)``
(``nu_an dt <= 0.03`` at 0.3 T / 1.4 ps / alpha 0.345; ``nu_mcc dt ~ 1e-5``) and keeps the
``alpha = 0`` path bitwise identical to the model without the hook.

**Secondary electron emission** -- ``SEEConfig`` (model v2.2.0 ``see_dielectric_v1``, implemented
in ``see.py`` / ``warp_see.py`` and re-exported here): the v1.4 scaffold refused ``enabled=True``;
since v2.2.0 emission is implemented on both backends (Vaughan yield with the Villemant 2019 BN
fit, Sydorenko 2006 component split, cosine emission, integer yield sampling, surface-charge
update).  ``virtual_wall_yield`` below stays as the map-based diagnostic of the yield the wall
would have at the observed impact energies, per wall column, so a run without emission can still
report where the cusp sheaths would go space-charge-limited (Campanell, Khrabrov and Kaganovich
2012; Hobbs and Wesson 1967).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any

import numpy as np

from .models import ELECTRON_MASS_KG, ELEMENTARY_CHARGE_C, PIC2DValidationError
from .see import SEEConfig, first_crossover_ev, vaughan_yield

BOHM_ALPHA_BRACKET = (1.0 / 64.0, 1.0 / 16.0)
# v2.1.0 alpha-series (physics audit section 4.c / roadmap R1): classical Bohm 1/16, a quarter of it, and Brandt 2016's
# D_perp = 0.4 kT_e / eB expressed as the exact Green-Kubo factor alpha / (1 + alpha^2) = 0.345 of nu = 0.4 omega_ce
BOHM_ALPHA_SERIES = (1.0 / 64.0, 1.0 / 16.0, 0.345)
ANOMALOUS_MODEL_ISOTROPIC = "bohm_isotropic_scattering"
ANOMALOUS_MODEL_ROTATION = "bohm_perpendicular_rotation"
ANOMALOUS_MODELS = (ANOMALOUS_MODEL_ISOTROPIC, ANOMALOUS_MODEL_ROTATION)


@dataclass(frozen=True, slots=True)
class AnomalousCollisionConfig:
    """Bohm-type anomalous scattering, ``nu_an = alpha omega_ce`` (0 < alpha <= 1), event model in ``ANOMALOUS_MODELS``.

    ``model`` defaults to the v1.4 isotropic redirect so every recorded identity (``to_dict``) is unchanged; the
    v2.1.0 perpendicular-rotation model (Brandt et al. 2016) is selected explicitly and enters ``config_sha256``.
    """

    alpha: float
    model: str = ANOMALOUS_MODEL_ISOTROPIC

    def __post_init__(self) -> None:
        if not isfinite(self.alpha) or not 0.0 < self.alpha <= 1.0:
            raise PIC2DValidationError("anomalous alpha must be in (0, 1]")
        if self.model not in ANOMALOUS_MODELS:
            raise PIC2DValidationError(f"anomalous model must be one of {ANOMALOUS_MODELS}, got {self.model!r}")

    @property
    def rotation(self) -> bool:
        return self.model == ANOMALOUS_MODEL_ROTATION

    @property
    def diffusion_factor(self) -> float:
        """``D_perp / (kT_e / eB) = alpha / (1 + alpha^2)`` (Green-Kubo, Maxwellian; both event models)."""

        return self.alpha / (1.0 + self.alpha**2)

    def to_dict(self) -> dict[str, Any]:
        return {"model": self.model, "alpha": self.alpha}


def bohm_collision_probability(alpha: float, b_magnitude_t: np.ndarray | float, dt_s: float) -> np.ndarray:
    """``1 - exp(-alpha omega_ce dt)`` at the local field magnitude (exact Poisson probability)."""

    omega_ce = ELEMENTARY_CHARGE_C * np.abs(np.asarray(b_magnitude_t, dtype=np.float64)) / ELECTRON_MASS_KG
    return -np.expm1(-alpha * omega_ce * dt_s)


def bohm_diffusion_coefficient_m2_per_s(alpha: float, t_e_ev: float, b_t: float) -> float:
    """``D_perp = (kT_e / eB) alpha / (1 + alpha^2)``: the cross-field diffusion coefficient both event models produce."""

    return t_e_ev / abs(b_t) * alpha / (1.0 + alpha**2)      # kT_e / e in eV -> V; V / T = m^2 / s


def isotropic_redirect(speed: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly random direction with the given speed (same map as the MCC isotropic scatter)."""

    cos_t = 1.0 - 2.0 * u1
    sin_t = np.sqrt(np.maximum(1.0 - cos_t**2, 0.0))
    phi = 2.0 * np.pi * u2
    return speed * sin_t * np.cos(phi), speed * sin_t * np.sin(phi), speed * cos_t


def rotate_about_field(
    vr: np.ndarray, vt: np.ndarray, vz: np.ndarray, b_r: np.ndarray, b_z: np.ndarray, phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rodrigues rotation of ``v = (v_r, v_theta, v_z)`` about the local unit field ``b = (b_r, 0, b_z) / |B|`` by ``phi``.

    ``v' = v cos(phi) + (b x v) sin(phi) + b (b . v) (1 - cos(phi))``: the parallel component ``b . v`` is invariant
    and the perpendicular component turns by ``phi`` (a gyro-phase reset); ``|v|`` is preserved to round-off.  Where
    ``|B| = 0`` the direction is undefined and the velocity is returned unchanged (the selection probability is 0
    there anyway).  The same formula, in the same operation order, is the Warp ``rotate_about_field`` device function.
    """

    b_mag = np.sqrt(b_r * b_r + b_z * b_z)
    safe = b_mag > 0.0
    denominator = np.where(safe, b_mag, 1.0)
    nr = b_r / denominator
    nz = b_z / denominator
    c = np.cos(phi)
    s = np.sin(phi)
    dot = nr * vr + nz * vz
    k = dot * (1.0 - c)
    # b x v with b = (nr, 0, nz) in the right-handed (r, theta, z) triad
    cr = -nz * vt
    ct = nz * vr - nr * vz
    cz = nr * vt
    new_r = vr * c + cr * s + nr * k
    new_t = vt * c + ct * s
    new_z = vz * c + cz * s + nz * k
    return np.where(safe, new_r, vr), np.where(safe, new_t, vt), np.where(safe, new_z, vz)


def apply_bohm_scattering(
    alpha: float, vr: np.ndarray, vt: np.ndarray, vz: np.ndarray, b_magnitude_t: np.ndarray, dt_s: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """CPU reference (isotropic model): redirect the electrons selected with the local Bohm probability; returns (vr, vt, vz, count)."""

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


def apply_bohm_rotation(
    alpha: float, vr: np.ndarray, vt: np.ndarray, vz: np.ndarray, b_r: np.ndarray, b_z: np.ndarray, dt_s: float, rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """CPU reference (perpendicular-rotation model): gyro-phase reset of the electrons selected with the local Bohm probability."""

    probability = bohm_collision_probability(alpha, np.hypot(b_r, b_z), dt_s)
    hit = rng.random(vr.size) < probability
    count = int(hit.sum())
    if count == 0:
        return vr, vt, vz, 0
    phi = 2.0 * np.pi * rng.random(count)
    new_r, new_t, new_z = rotate_about_field(vr[hit], vt[hit], vz[hit], b_r[hit], b_z[hit], phi)
    vr, vt, vz = vr.copy(), vt.copy(), vz.copy()
    vr[hit], vt[hit], vz[hit] = new_r, new_t, new_z
    return vr, vt, vz, count


def apply_anomalous_scattering(
    config: AnomalousCollisionConfig, vr: np.ndarray, vt: np.ndarray, vz: np.ndarray, b_r: np.ndarray, b_z: np.ndarray, dt_s: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Dispatch on the declared event model (the isotropic path draws exactly as v1.4 did: recorded runs replay bitwise)."""

    if config.rotation:
        return apply_bohm_rotation(config.alpha, vr, vt, vz, b_r, b_z, dt_s, rng)
    return apply_bohm_scattering(config.alpha, vr, vt, vz, np.hypot(b_r, b_z), dt_s, rng)


# v1.4 scaffold constants kept for reference: the Hall-thruster PIC "usual" BN Vaughan set (Dawson-like maximum,
# Vaughan default threshold).  Model v2.2.0 (see.py) replaces them with the Villemant 2019 fit; the scaffold's
# SEEConfig / vaughan_yield / first_crossover_ev now live in see.py and are re-exported here.
BN_VAUGHAN: dict[str, float] = {"delta_max": 2.9, "energy_max_ev": 350.0, "energy_threshold_ev": 12.5}


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
        "note": ("virtual yield at the column-mean impact energy (map diagnostic); columns at or above the Hobbs-Wesson "
                 "limit are where an emitting wall goes space-charge-limited (v2.2.0 lets the PIC form the virtual cathode itself)"),
    }


__all__ = [
    "ANOMALOUS_MODELS",
    "ANOMALOUS_MODEL_ISOTROPIC",
    "ANOMALOUS_MODEL_ROTATION",
    "BN_VAUGHAN",
    "BOHM_ALPHA_BRACKET",
    "BOHM_ALPHA_SERIES",
    "AnomalousCollisionConfig",
    "SEEConfig",
    "apply_anomalous_scattering",
    "apply_bohm_rotation",
    "apply_bohm_scattering",
    "bohm_collision_probability",
    "bohm_diffusion_coefficient_m2_per_s",
    "first_crossover_ev",
    "isotropic_redirect",
    "rotate_about_field",
    "vaughan_yield",
    "virtual_wall_yield",
]
