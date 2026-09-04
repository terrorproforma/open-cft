"""Coulomb collisions (model v2.4.0 ``coulomb_v1``): binary pairing per cell, Nanbu cumulative angle - CPU reference.

Physics audit gap (d) / roadmap R4.  Electron-electron and electron-ion (optionally ion-ion) Coulomb collisions
as the binary-collision Monte Carlo of Takizuka and Abe (1977) with the cumulative small-angle scattering
statistics of Nanbu (1997):

* every ``cycle_steps`` steps (``Delta t_c = cycle_steps x dt``) the alive particles of each species are grouped
  by mesh cell; like particles are paired at random inside their cell (a uniform random permutation of the
  cell's members; an odd count uses the Takizuka-Abe triplet: the first three members collide pairwise
  ``(0,1), (0,2), (1,2)`` with ``Delta t_c / 2`` each, so every particle experiences exactly one full time step
  of scattering); electron-ion pairs put EVERY electron of a cell against an ion of the same cell (electron
  ``l`` against ion ``(l + shift) mod N_i``, a random ``shift`` per cell), which gives the electrons exactly
  one collision each at the field density ``n_i`` and the ions ``N_e / N_i`` collisions on average - both
  rates are the physical ones (the Takizuka-Abe "smaller group collides repeatedly" rule);
* the deflection of a pair with relative speed ``g = |v_a - v_b|`` over ``Delta t`` is drawn from Nanbu's
  cumulative distribution ``f(chi) ~ exp(A cos chi)`` with ``coth A - 1/A = exp(-s)``,
  ``s = (ln Lambda / 4 pi) (q_a q_b / (epsilon_0 m_ab))^2 n_field Delta t / g^3`` (``m_ab`` the reduced mass):
  ``<1 - cos chi> = 1 - exp(-s)``, i.e. ``<chi^2> = 2 s`` for small ``s``, the Takizuka-Abe variance
  ``<delta^2> = s / 2`` of ``delta = tan(chi / 2)`` - valid for ANY ``s`` (isotropic for ``s -> infinity``), so
  the operator does not need ``s << 1`` per cycle; ``A(s)`` uses ``A = 1 / (1 - exp(-s))`` (exact mean) for
  ``s < 0.2``, the Perez et al. (2012) polynomial for ``0.2 <= s < 3``, ``A = 3 exp(-s)`` for ``3 <= s < 6`` and an
  isotropic angle beyond; the azimuth is uniform;
* the post-collision velocities are the exact centre-of-mass rotation: ``v_a' = v_a + (m_b / M) Delta u``,
  ``v_b' = v_b - (m_a / M) Delta u`` with ``|u + Delta u| = |u|`` - momentum and (classical) kinetic energy of
  the pair are conserved to round-off; equal macro weights (all species share ``macro_weight``) make the pair
  update exact without the Nanbu-Yonemura weighted-particle correction;
* the Coulomb logarithm is the NRL Formulary expression from the cell's own density and temperature
  (``n = N W / V_cell``, ``T = m (<v^2> - |<v>|^2) / 3 e``), with the temperature floor ``min_temperature_ev``
  and the floor ``coulomb_log_floor`` (a fixed value is available for tests).

The velocity components ``(v_r, v_theta, v_z)`` of a pair are treated as Cartesian components of one local
frame (the two particles sit in the same cell of an axisymmetric domain; the same convention as the MCC
isotropic scatter), so ``v_z`` momentum is conserved exactly and the ledger term ``pz_coulomb`` is zero by
construction.  The energy ledger books ``ke_coulomb_j``: the change of the RELATIVISTIC kinetic energy of the
pairs, ``O(v^2 / c^2)`` of the redistributed energy (~1e-5 relative at 10 eV), so the ledger identity closes to
round-off without a physical energy source (the operator is elastic).

Frequencies: ``s`` is the per-pair deflection parameter ``nu_pair Delta t``; the reported ``nu_ee`` is
``2 sum s / (sum_cycles N_e Delta t_c)`` (each pair gives both electrons ``s``), ``nu_ei = sum s_ei / (sum_cycles
N_e Delta t_c)``.  The audit's estimates (NRL ``nu_e = 2.91e-6 n lnL T^-3/2``) are the reference for the
comparison with the electron-neutral rate.

The GPU stage (``warp_coulomb.py``) implements the same pairing on a per-step cell-sorted PERMUTATION of the
particle slots (particles are never reordered, so the MCC / injection / anomalous / ion-MCC / SEE random streams
keyed on the slot index are untouched and every configuration without Coulomb replays bitwise); its random
numbers come from the dedicated seed-table stream (id 6), so parity with this reference is distributional.

References: Takizuka & Abe, J. Comput. Phys. 25, 205 (1977) doi:10.1016/0021-9991(77)90099-7; Nanbu, Phys. Rev.
E 55, 4642 (1997) doi:10.1103/PhysRevE.55.4642; Nanbu & Yonemura, J. Comput. Phys. 145, 639 (1998)
doi:10.1006/jcph.1998.6049; Perez et al., Phys. Plasmas 19, 083104 (2012) doi:10.1063/1.4742167; NRL Plasma
Formulary (2019) for the Coulomb logarithms and the relaxation rates; Trubnikov, Rev. Plasma Phys. 1, 105 (1965).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import atan, isfinite, pi, sqrt
from typing import Any

import numpy as np

from .mesh import MeshMasks, cell_index
from .models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EPSILON_0_F_PER_M,
    EV_J,
    LIGHT_SPEED_M_PER_S,
    Grid2D,
    ParticleArrays,
    PIC2DValidationError,
)

COULOMB_MODEL = "coulomb_v1"
# CPU RNG stream (np.random.default_rng([seed, step, id])) and the Warp seed-table stream id of the Coulomb stage:
# 1 MCC, 2 injection, 3 anomalous scattering (0 = seed plasma), 4 ion-neutral MCC, 5 SEE, 6 Coulomb (v2.4.0)
COULOMB_RNG_STREAM = 6
# Nanbu branch limits
S_SMALL = 0.2
S_LARGE = 6.0
# Coulomb constant e^2 / (4 pi epsilon_0) in J m
COULOMB_CONSTANT_J_M = ELEMENTARY_CHARGE_C**2 / (4.0 * pi * EPSILON_0_F_PER_M)

# Extra ledger keys (present only when the operator is on; counts are pair collisions, sums are over pairs):
#   coulomb_ee_pairs / coulomb_ei_pairs / coulomb_ii_pairs  binary collisions performed (triplet sub-pairs count each)
#   coulomb_ee_s_sum / ...                                  sum of the pair deflection parameter s (= nu_pair Delta t)
#   coulomb_ee_large_s / ...                                pairs with s > 1 (the once-per-cycle pairing is coarse there)
#   coulomb_ee_lnl_sum / ...                                sum of the Coulomb logarithm over pairs
#   coulomb_electron_cycles / coulomb_ion_cycles            sum over cycles of the alive electron / ion count
#   coulomb_cycles                                          number of collision cycles
#   pz_coulomb                                              W sum (m_a dv_a,z + m_b dv_b,z) over pairs (0 by construction)
#   ke_coulomb_j                                            W sum of the relativistic pair kinetic-energy change (O(v^2/c^2))
COULOMB_SPECIES_PAIRS = ("ee", "ei", "ii")
COULOMB_KEYS = tuple(
    f"coulomb_{pair}_{name}" for pair in COULOMB_SPECIES_PAIRS for name in ("pairs", "s_sum", "large_s", "lnl_sum")
) + ("coulomb_electron_cycles", "coulomb_ion_cycles", "coulomb_cycles", "pz_coulomb", "ke_coulomb_j")


@dataclass(frozen=True, slots=True)
class CoulombConfig:
    """Coulomb collision operator (model v2.4.0).  Every field enters ``config_sha256`` through ``to_dict``.

    ``cycle_steps`` k: the operator is applied every k steps with ``Delta t_c = k dt`` (the "sub-cycling" of the
    audit: at 1.4 ps and n_e 1e19 / 5 eV, ``nu_ee dt ~ 4e-5``, so k = 10 keeps the per-cycle deflection parameter
    ``s ~ 4e-4`` at the peak - far inside the small-angle regime the pairing-once-per-cycle statistics assume - and
    ``k dt = 14 ps`` is under half the 33 um cell crossing time of a 5 eV electron (33 ps), so a particle's collision
    partners are still its cell neighbours).  ``electron_ion`` collides every electron against a cell ion;
    ``ion_ion`` is off by default: with the local ion "temperature" set by the birth-potential spread (10-50 eV)
    ``nu_ii / nu_ee = (m_e / M)^1/2 (T_e / T_i)^3/2 ~ 1e-3`` and ``nu_ii x transit << 1``; in the cold anode-side
    population ``nu_ii tau`` can reach 0.1-1 but those ions are unaccelerated and their thermalisation at 0.03 eV
    changes nothing the model claims.  ``coulomb_log_fixed`` replaces the NRL local value (tests; declared).
    """

    enabled: bool = True
    electron_electron: bool = True
    electron_ion: bool = True
    ion_ion: bool = False
    cycle_steps: int = 10
    coulomb_log_floor: float = 2.0
    coulomb_log_fixed: float | None = None
    min_temperature_ev: float = 0.01

    def __post_init__(self) -> None:
        for name in ("enabled", "electron_electron", "electron_ion", "ion_ion"):
            if not isinstance(getattr(self, name), bool):
                raise PIC2DValidationError(f"coulomb.{name} must be a bool")
        if isinstance(self.cycle_steps, bool) or not isinstance(self.cycle_steps, int) or self.cycle_steps < 1:
            raise PIC2DValidationError("coulomb.cycle_steps must be a positive integer")
        for name in ("coulomb_log_floor", "min_temperature_ev"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
                raise PIC2DValidationError(f"coulomb.{name} must be positive")
            object.__setattr__(self, name, float(value))
        if self.coulomb_log_fixed is not None:
            value = self.coulomb_log_fixed
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not isfinite(value) or value <= 0.0:
                raise PIC2DValidationError("coulomb.coulomb_log_fixed must be positive when set")
            object.__setattr__(self, "coulomb_log_fixed", float(value))
        if self.enabled and not (self.electron_electron or self.electron_ion or self.ion_ion):
            raise PIC2DValidationError("an enabled Coulomb operator needs at least one species pair")

    @property
    def active(self) -> bool:
        return self.enabled

    def to_dict(self) -> dict[str, Any]:
        return {
            "model": COULOMB_MODEL, "enabled": self.enabled,
            "electron_electron": self.electron_electron, "electron_ion": self.electron_ion, "ion_ion": self.ion_ion,
            "cycle_steps": self.cycle_steps,
            "coulomb_log": ("fixed" if self.coulomb_log_fixed is not None else "nrl_local_cell"),
            "coulomb_log_fixed": self.coulomb_log_fixed, "coulomb_log_floor": self.coulomb_log_floor,
            "min_temperature_ev": self.min_temperature_ev,
            "method": {
                "pairing": "Takizuka-Abe 1977 binary pairs inside each mesh cell (random permutation; odd count: triplet with dt/2); "
                           "electron-ion: every electron against ion (l + shift) mod N_i of its cell at the field density n_i",
                "angle": "Nanbu 1997 cumulative scattering angle, coth A - 1/A = exp(-s), s = (lnL / 4 pi) (q_a q_b / eps0 m_ab)^2 n Delta t / g^3; "
                         "A = 1/(1 - exp(-s)) below 0.2, Perez 2012 fit to 3, 3 exp(-s) to 6, isotropic beyond",
                "kinematics": "exact centre-of-mass rotation, equal macro weights: pair momentum and classical energy conserved to round-off",
                "rng_stream": COULOMB_RNG_STREAM,
                "gpu": "per-step cell-sorted slot permutation (counting sort with a deterministic within-cell rank), particles never reordered",
            },
        }


# -- Coulomb logarithms (NRL Plasma Formulary 2019; n in m^-3 here, converted to cm^-3 inside) ----------------------------

def coulomb_log_ee(n_e_per_m3: np.ndarray | float, t_e_ev: np.ndarray | float, floor: float = 2.0) -> np.ndarray:
    """Electron-electron: ``23.5 - ln(n^1/2 T^-5/4) - [1e-5 + (ln T - 2)^2 / 16]^1/2`` (n cm^-3, T eV), floored."""

    n = np.maximum(np.asarray(n_e_per_m3, dtype=np.float64) * 1.0e-6, 1.0e-300)
    t = np.maximum(np.asarray(t_e_ev, dtype=np.float64), 1.0e-300)
    value = 23.5 - np.log(np.sqrt(n) * t ** (-1.25)) - np.sqrt(1.0e-5 + (np.log(t) - 2.0) ** 2 / 16.0)
    return np.maximum(value, floor)


def coulomb_log_ei(n_e_per_m3: np.ndarray | float, t_e_ev: np.ndarray | float, floor: float = 2.0, z: float = 1.0) -> np.ndarray:
    """Electron-ion (T_i m_e / m_i < T_e): ``23 - ln(n^1/2 Z T^-3/2)`` below ``10 Z^2`` eV, ``24 - ln(n^1/2 T^-1)`` above."""

    n = np.maximum(np.asarray(n_e_per_m3, dtype=np.float64) * 1.0e-6, 1.0e-300)
    t = np.maximum(np.asarray(t_e_ev, dtype=np.float64), 1.0e-300)
    low = 23.0 - np.log(np.sqrt(n) * z * t ** (-1.5))
    high = 24.0 - np.log(np.sqrt(n) / t)
    return np.maximum(np.where(t < 10.0 * z * z, low, high), floor)


def coulomb_log_ii(n_i_per_m3: np.ndarray | float, t_i_ev: np.ndarray | float, floor: float = 2.0) -> np.ndarray:
    """Like singly-charged ions: ``23 - ln[(1 / T_i) (2 n_i / T_i)^1/2]`` (n cm^-3, T eV), floored."""

    n = np.maximum(np.asarray(n_i_per_m3, dtype=np.float64) * 1.0e-6, 1.0e-300)
    t = np.maximum(np.asarray(t_i_ev, dtype=np.float64), 1.0e-300)
    return np.maximum(23.0 - np.log(np.sqrt(2.0 * n / t) / t), floor)


# -- Nanbu cumulative scattering angle ---------------------------------------------------------------------------------

def nanbu_inverse_a(s: np.ndarray) -> np.ndarray:
    """``1 / A(s)`` of ``coth A - 1/A = exp(-s)``: exact-mean form below 0.2, Perez 2012 polynomial to 3, ``exp(s) / 3`` to 6."""

    s = np.asarray(s, dtype=np.float64)
    small = -np.expm1(-s)                                   # 1/A = 1 - exp(-s): <cos chi> = exp(-s) exactly (e^-2A negligible)
    poly = 0.0056958 + s * (0.9560202 + s * (-0.508139 + s * (0.47913906 + s * (-0.12788975 + s * 0.02389567))))
    large = np.exp(np.minimum(s, S_LARGE)) / 3.0
    return np.where(s < S_SMALL, small, np.where(s < 3.0, poly, large))


def nanbu_cos_chi(s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Sample ``cos chi`` of the cumulative scattering angle for deflection parameter ``s`` and uniform ``u`` in (0, 1)."""

    s = np.asarray(s, dtype=np.float64)
    u = np.clip(np.asarray(u, dtype=np.float64), 1.0e-30, 1.0)
    inv_a = nanbu_inverse_a(s)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        a = 1.0 / inv_a
        small = 1.0 + inv_a * np.log(u)                                         # A large: ln(2 u sinh A + e^-A) / A -> 1 + ln(u) / A
        mid = inv_a * np.log(np.exp(-a) + 2.0 * u * np.sinh(a))
    isotropic = 2.0 * u - 1.0
    value = np.where(s < S_SMALL, small, np.where(s < S_LARGE, mid, isotropic))
    return np.clip(value, -1.0, 1.0)


def deflection_parameter(g: np.ndarray, *, charge_a: float, charge_b: float, mass_a: float, mass_b: float,
                         n_field_per_m3: np.ndarray | float, coulomb_log: np.ndarray | float, dt_s: float) -> np.ndarray:
    """``s = (ln Lambda / 4 pi) (q_a q_b / (epsilon_0 m_ab))^2 n_field dt / g^3`` (Nanbu 1997 Eq. 3; = 2 x TA variance)."""

    reduced = mass_a * mass_b / (mass_a + mass_b)
    factor = (charge_a * charge_b / (EPSILON_0_F_PER_M * reduced)) ** 2 / (4.0 * pi)
    g3 = np.maximum(np.asarray(g, dtype=np.float64), 1.0e-300) ** 3
    return factor * np.asarray(coulomb_log, dtype=np.float64) * np.asarray(n_field_per_m3, dtype=np.float64) * dt_s / g3


def scatter_pairs(
    va: np.ndarray, vb: np.ndarray, mass_a: float, mass_b: float, cos_chi: np.ndarray, phi: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Exact centre-of-mass rotation of the relative velocity by (chi, phi) (Takizuka-Abe 1977 Eqs. 4-6).

    ``va``, ``vb`` are ``(n, 3)`` arrays; returns the post-collision velocities.  ``|u'| = |u|`` and
    ``m_a v_a + m_b v_b`` are preserved to round-off; pairs with ``g = 0`` are returned unchanged.
    """

    u = va - vb
    ux, uy, uz = u[:, 0], u[:, 1], u[:, 2]
    g = np.sqrt(ux * ux + uy * uy + uz * uz)
    u_perp = np.sqrt(ux * ux + uy * uy)
    sin_chi = np.sqrt(np.maximum(1.0 - cos_chi * cos_chi, 0.0))
    one_minus = 1.0 - cos_chi
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    safe = u_perp > 0.0
    inv_perp = np.where(safe, 1.0 / np.where(safe, u_perp, 1.0), 0.0)
    dux = np.where(
        safe,
        (ux * inv_perp) * uz * sin_chi * cos_phi - (uy * inv_perp) * g * sin_chi * sin_phi - ux * one_minus,
        g * sin_chi * cos_phi,
    )
    duy = np.where(
        safe,
        (uy * inv_perp) * uz * sin_chi * cos_phi + (ux * inv_perp) * g * sin_chi * sin_phi - uy * one_minus,
        g * sin_chi * sin_phi,
    )
    duz = np.where(safe, -u_perp * sin_chi * cos_phi - uz * one_minus, -uz * one_minus)
    du = np.stack((dux, duy, duz), axis=1)
    du[g == 0.0] = 0.0
    total = mass_a + mass_b
    return va + (mass_b / total) * du, vb - (mass_a / total) * du


def relativistic_kinetic_energy_j(v: np.ndarray, mass_kg: float) -> np.ndarray:
    """Per-particle ``(gamma - 1) m c^2`` for an ``(n, 3)`` velocity array (the ledger's kinetic energy)."""

    c2 = LIGHT_SPEED_M_PER_S**2
    speed2 = np.sum(v * v, axis=1)
    return speed2 / c2 / (1.0 + np.sqrt(1.0 - speed2 / c2)) * mass_kg * c2


# -- reference relaxation rates (tests / documentation) -----------------------------------------------------------------

def trubnikov_isotropization_rate(n_per_m3: float, t_par_ev: float, t_perp_ev: float, mass_kg: float, coulomb_log: float,
                                  charge_c: float = ELEMENTARY_CHARGE_C) -> float:
    """``nu_T`` of ``dT_perp/dt = -1/2 dT_par/dt = -nu_T (T_perp - T_par)`` for a bi-Maxwellian (Trubnikov 1965; NRL).

    ``nu_T = 2 sqrt(pi) (q^2 / 4 pi eps0)^2 n lnL / (m^1/2 (k T_par)^3/2) A^-2 [-3 + (A + 3) arctan(A^1/2) / A^1/2]``,
    ``A = T_perp / T_par - 1`` (the ``atanh`` form for ``A < 0``); ``A -> 0`` gives ``(4/15)`` for the bracket, i.e.
    ``8.2e-7 n[cm^-3] lnL T^-3/2`` for electrons.
    """

    a = t_perp_ev / t_par_ev - 1.0
    if abs(a) < 1.0e-6:
        bracket = 4.0 / 15.0
    elif a > 0.0:
        root = sqrt(a)
        bracket = (-3.0 + (a + 3.0) * atan(root) / root) / (a * a)
    else:
        root = sqrt(-a)
        bracket = (-3.0 + (a + 3.0) * np.arctanh(root) / root) / (a * a)
    prefactor = 2.0 * sqrt(pi) * (charge_c**2 / (4.0 * pi * EPSILON_0_F_PER_M)) ** 2 * n_per_m3 * coulomb_log
    return float(prefactor / (sqrt(mass_kg) * (t_par_ev * EV_J) ** 1.5) * bracket)


def spitzer_electron_ion_momentum_rate(n_i_per_m3: float, t_e_ev: float, coulomb_log: float, z: float = 1.0) -> float:
    """Drift decay rate of a Maxwellian electron population on cold ions: ``nu_ei = 4 sqrt(2 pi) / 3 (e^2/4 pi eps0)^2 Z^2 n_i lnL /
    (m_e^1/2 (k T_e)^3/2)`` = NRL ``2.91e-6 n[cm^-3] lnL T^-3/2`` s^-1 (the "electron collision rate" of the audit)."""

    prefactor = 4.0 * sqrt(2.0 * pi) / 3.0 * COULOMB_CONSTANT_J_M**2 * z * z * n_i_per_m3 * coulomb_log
    return float(prefactor / (sqrt(ELECTRON_MASS_KG) * (t_e_ev * EV_J) ** 1.5))


def temperature_equilibration_rate(n_b_per_m3: float, mass_a: float, mass_b: float, t_a_ev: float, t_b_ev: float, coulomb_log: float,
                                   z_a: float = 1.0, z_b: float = 1.0) -> float:
    """``nu_eps^{a\\b}`` of ``dT_a/dt = nu (T_b - T_a)`` between two Maxwellians (NRL "thermal equilibration"):
    ``8 sqrt(2 pi) / 3 (e^2/4 pi eps0)^2 Z_a^2 Z_b^2 n_b lnL (m_a m_b)^1/2 / (m_a k T_b + m_b k T_a)^3/2``
    (= ``1.8e-19 (m_a m_b)^1/2 Z^2 Z^2 n_b lnL / (m_a T_b + m_b T_a)^3/2`` in g, cm^-3, eV)."""

    prefactor = 8.0 * sqrt(2.0 * pi) / 3.0 * COULOMB_CONSTANT_J_M**2 * z_a**2 * z_b**2 * n_b_per_m3 * coulomb_log
    return float(prefactor * sqrt(mass_a * mass_b) / (mass_a * t_b_ev * EV_J + mass_b * t_a_ev * EV_J) ** 1.5)


# -- CPU reference operator ------------------------------------------------------------------------------------------------

@dataclass(slots=True)
class CoulombTally:
    ee_pairs: float = 0.0
    ee_s_sum: float = 0.0
    ee_large_s: float = 0.0
    ee_lnl_sum: float = 0.0
    ei_pairs: float = 0.0
    ei_s_sum: float = 0.0
    ei_large_s: float = 0.0
    ei_lnl_sum: float = 0.0
    ii_pairs: float = 0.0
    ii_s_sum: float = 0.0
    ii_large_s: float = 0.0
    ii_lnl_sum: float = 0.0
    electron_cycles: float = 0.0
    ion_cycles: float = 0.0
    cycles: float = 0.0
    pz_coulomb: float = 0.0
    ke_coulomb_j: float = 0.0
    ee_s_max: float = 0.0
    ei_s_max: float = 0.0

    def to_cumulative(self) -> dict[str, float]:
        out: dict[str, float] = {}
        for pair in COULOMB_SPECIES_PAIRS:
            for name in ("pairs", "s_sum", "large_s", "lnl_sum"):
                out[f"coulomb_{pair}_{name}"] = float(getattr(self, f"{pair}_{name}"))
        out["coulomb_electron_cycles"] = self.electron_cycles
        out["coulomb_ion_cycles"] = self.ion_cycles
        out["coulomb_cycles"] = self.cycles
        out["pz_coulomb"] = self.pz_coulomb
        out["ke_coulomb_j"] = self.ke_coulomb_j
        return out


@dataclass(slots=True)
class CoulombResult:
    electrons: ParticleArrays
    ions: ParticleArrays
    tally: CoulombTally
    # per-cell window contributions of this cycle (cell-shaped (nr, nz)): sum s and pair count per species pair, and the
    # electron-seconds N_e x dt_c (so nu_ee = 2 sum s / electron_seconds needs no cycle length)
    cell_ee_s: np.ndarray
    cell_ee_pairs: np.ndarray
    cell_ei_s: np.ndarray
    cell_ei_pairs: np.ndarray
    cell_electron_seconds: np.ndarray


def cell_volumes_m3(grid: Grid2D) -> np.ndarray:
    """Ring volume ``pi (r_{i+1}^2 - r_i^2) dz`` per radial cell index (the same for every axial index)."""

    r = grid.r_m
    return pi * (r[1:] ** 2 - r[:-1] ** 2) * grid.dz_m


def cell_moments(cell: np.ndarray, v: np.ndarray, n_cells: int, mass_kg: float, *, min_temperature_ev: float) -> tuple[np.ndarray, np.ndarray]:
    """Per-cell (count, temperature eV) from ``(n, 3)`` velocities; ``T = m (<v^2> - |<v>|^2) / 3 e`` floored."""

    count = np.bincount(cell, minlength=n_cells).astype(np.float64)
    sums = np.stack([np.bincount(cell, weights=v[:, k], minlength=n_cells) for k in range(3)], axis=1)
    sum2 = np.bincount(cell, weights=np.sum(v * v, axis=1), minlength=n_cells)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = sums / np.maximum(count, 1.0)[:, None]
        mean2 = sum2 / np.maximum(count, 1.0)
    variance = np.maximum(mean2 - np.sum(mean * mean, axis=1), 0.0)
    temperature = np.where(count > 0.0, mass_kg * variance / (3.0 * EV_J), 0.0)
    return count, np.maximum(temperature, min_temperature_ev)


class CoulombOperator:
    """Numpy reference of one Coulomb cycle on the full electron and ion populations (``apply``)."""

    def __init__(self, config: CoulombConfig, grid: Grid2D, masks: MeshMasks, macro_weight: float, *,
                 electron_mass_kg: float = ELECTRON_MASS_KG, ion_mass_kg: float, ion_charge_c: float = ELEMENTARY_CHARGE_C) -> None:
        self.config = config
        self.grid = grid
        self.masks = masks
        self.macro_weight = float(macro_weight)
        self.m_e = float(electron_mass_kg)
        self.m_i = float(ion_mass_kg)
        self.q_i = float(ion_charge_c)
        self.nr, self.nz = grid.cell_shape
        self.n_cells = self.nr * self.nz
        self.cell_volume = cell_volumes_m3(grid)         # per radial index

    def _cells(self, particles: ParticleArrays) -> np.ndarray:
        i, j, _, _ = cell_index(self.grid, particles.r_m, particles.z_m)
        return (i * self.nz + j).astype(np.int64)

    def _coulomb_log(self, kind: str, n: np.ndarray, t: np.ndarray) -> np.ndarray:
        if self.config.coulomb_log_fixed is not None:
            return np.full(np.shape(n), self.config.coulomb_log_fixed, dtype=np.float64)
        floor = self.config.coulomb_log_floor
        if kind == "ee":
            return coulomb_log_ee(n, t, floor)
        if kind == "ei":
            return coulomb_log_ei(n, t, floor)
        return coulomb_log_ii(n, t, floor)

    def _like_pairs(self, cell: np.ndarray, order: np.ndarray, count: np.ndarray, rng: np.random.Generator,
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Takizuka-Abe pairing of ``order`` (particles sorted by cell, cells shuffled internally).

        Returns (a, b, dt_fraction) index arrays: for even cells consecutive pairs with 1.0; odd cells the triplet
        ``(0,1), (0,2), (1,2)`` with 0.5 each then consecutive pairs from member 3.  Members of the three triplet
        sub-pairs share particles, so the caller applies them in three sequential passes (pass id = position).
        """

        starts = np.concatenate(([0], np.cumsum(count))).astype(np.int64)
        a_list: list[np.ndarray] = []
        b_list: list[np.ndarray] = []
        f_list: list[np.ndarray] = []
        pass_list: list[np.ndarray] = []
        occupied = np.flatnonzero(count >= 2)
        if occupied.size == 0:
            empty = np.zeros(0, dtype=np.int64)
            return empty, empty, np.zeros(0, dtype=np.float64), empty
        # shuffle inside each occupied cell (a random permutation per cell)
        for c in occupied:
            base, n = starts[c], int(count[c])
            segment = order[base:base + n]
            order[base:base + n] = segment[rng.permutation(n)]
        n_occ = count[occupied]
        starts_occ = starts[occupied]
        odd = (n_occ % 2) == 1
        # regular pairs: even cells from member 0, odd cells from member 3
        first = np.where(odd, 3, 0)
        n_pairs = ((n_occ - first) // 2).astype(np.int64)
        n_pairs = np.maximum(n_pairs, 0)
        total_pairs = int(n_pairs.sum())
        if total_pairs:
            cell_rep = np.repeat(np.arange(occupied.size), n_pairs)
            within = np.arange(total_pairs) - np.repeat(np.cumsum(n_pairs) - n_pairs, n_pairs)
            pos = starts_occ[cell_rep] + first[cell_rep] + 2 * within
            a_list.append(order[pos])
            b_list.append(order[pos + 1])
            f_list.append(np.ones(total_pairs))
            pass_list.append(np.zeros(total_pairs, dtype=np.int64))
        tri = np.flatnonzero(odd & (n_occ >= 3))
        if tri.size:
            base = starts_occ[tri]
            for pass_id, (x, y) in enumerate(((0, 1), (0, 2), (1, 2))):
                a_list.append(order[base + x])
                b_list.append(order[base + y])
                f_list.append(np.full(tri.size, 0.5))
                pass_list.append(np.full(tri.size, pass_id, dtype=np.int64))
        a = np.concatenate(a_list)
        b = np.concatenate(b_list)
        fraction = np.concatenate(f_list)
        passes = np.concatenate(pass_list)
        return a, b, fraction, passes

    def _collide(self, va: np.ndarray, vb: np.ndarray, mass_a: float, mass_b: float, charge_a: float, charge_b: float,
                 n_field: np.ndarray, lnl: np.ndarray, dt: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        u = va - vb
        g = np.sqrt(np.sum(u * u, axis=1))
        s = deflection_parameter(g, charge_a=charge_a, charge_b=charge_b, mass_a=mass_a, mass_b=mass_b,
                                 n_field_per_m3=n_field, coulomb_log=lnl, dt_s=1.0) * dt
        s = np.where(g > 0.0, s, 0.0)
        u1 = rng.random(g.size)
        u2 = rng.random(g.size)
        cos_chi = nanbu_cos_chi(s, u1)
        va_new, vb_new = scatter_pairs(va, vb, mass_a, mass_b, cos_chi, 2.0 * pi * u2)
        return va_new, vb_new, s

    def apply(self, electrons: ParticleArrays, ions: ParticleArrays, dt_c: float, rng: np.random.Generator) -> CoulombResult:
        config = self.config
        w = self.macro_weight
        tally = CoulombTally()
        tally.cycles = 1.0
        tally.electron_cycles = float(electrons.count)
        tally.ion_cycles = float(ions.count)
        shape = (self.nr, self.nz)
        cell_ee_s = np.zeros(shape)
        cell_ee_pairs = np.zeros(shape)
        cell_ei_s = np.zeros(shape)
        cell_ei_pairs = np.zeros(shape)
        cell_electron_seconds = np.zeros(shape)
        ve = np.stack((electrons.vr_m_per_s, electrons.vt_m_per_s, electrons.vz_m_per_s), axis=1).copy()
        vi = np.stack((ions.vr_m_per_s, ions.vt_m_per_s, ions.vz_m_per_s), axis=1).copy()
        ke_before = float(np.sum(relativistic_kinetic_energy_j(ve, self.m_e))) + float(np.sum(relativistic_kinetic_energy_j(vi, self.m_i)))
        pz_before = self.m_e * float(np.sum(ve[:, 2])) + self.m_i * float(np.sum(vi[:, 2]))
        cell_e = self._cells(electrons) if electrons.count else np.zeros(0, dtype=np.int64)
        cell_i = self._cells(ions) if ions.count else np.zeros(0, dtype=np.int64)
        count_e, t_e = cell_moments(cell_e, ve, self.n_cells, self.m_e, min_temperature_ev=config.min_temperature_ev)
        count_i, t_i = cell_moments(cell_i, vi, self.n_cells, self.m_i, min_temperature_ev=config.min_temperature_ev)
        volume = np.repeat(self.cell_volume, self.nz)                 # cell c = i * nz + j
        n_e = count_e * w / volume
        n_i = count_i * w / volume
        cell_electron_seconds[:] = count_e.reshape(shape) * dt_c
        order_e = np.argsort(cell_e, kind="stable")
        order_i = np.argsort(cell_i, kind="stable")

        def like(v: np.ndarray, cell: np.ndarray, order: np.ndarray, count: np.ndarray, density: np.ndarray, kind: str, mass: float,
                 charge: float, cell_s: np.ndarray | None, cell_pairs: np.ndarray | None) -> tuple[float, float, float, float, float]:
            temperature = t_e if kind == "ee" else t_i
            lnl_cell = self._coulomb_log(kind, density, temperature)
            a, b, fraction, passes = self._like_pairs(cell, order, count, rng)
            pairs = s_sum = large = lnl_sum = s_max = 0.0
            for pass_id in range(3):
                sel = passes == pass_id
                if not np.any(sel):
                    continue
                ia, ib = a[sel], b[sel]
                c = cell[ia]
                va_new, vb_new, s = self._collide(v[ia], v[ib], mass, mass, charge, charge, density[c], lnl_cell[c], dt_c * fraction[sel], rng)
                v[ia] = va_new
                v[ib] = vb_new
                pairs += float(ia.size)
                s_sum += float(s.sum())
                large += float(np.count_nonzero(s > 1.0))
                lnl_sum += float(lnl_cell[c].sum())
                s_max = max(s_max, float(s.max()))
                if cell_s is not None and cell_pairs is not None:
                    np.add.at(cell_s.reshape(-1), c, s)
                    np.add.at(cell_pairs.reshape(-1), c, 1.0)
            return pairs, s_sum, large, lnl_sum, s_max

        if config.electron_electron and electrons.count >= 2:
            tally.ee_pairs, tally.ee_s_sum, tally.ee_large_s, tally.ee_lnl_sum, tally.ee_s_max = like(
                ve, cell_e, order_e, count_e, n_e, "ee", self.m_e, -ELEMENTARY_CHARGE_C, cell_ee_s, cell_ee_pairs)
        if config.ion_ion and ions.count >= 2:
            tally.ii_pairs, tally.ii_s_sum, tally.ii_large_s, tally.ii_lnl_sum, _ = like(
                vi, cell_i, order_i, count_i, n_i, "ii", self.m_i, self.q_i, None, None)
        if config.electron_ion and electrons.count and ions.count:
            # electron l of cell c meets ion (l + shift_c) mod N_i; rounds m = l // N_i keep every ion in one pair per round
            starts_e = np.concatenate(([0], np.cumsum(count_e))).astype(np.int64)
            starts_i = np.concatenate(([0], np.cumsum(count_i))).astype(np.int64)
            shift = rng.integers(0, 2**31 - 1, size=self.n_cells)
            lnl_cell = self._coulomb_log("ei", n_e, t_e)
            # per electron (sorted order): local index, cell, ion count
            cell_sorted = cell_e[order_e]
            local = np.arange(electrons.count) - starts_e[cell_sorted]
            n_i_cell = count_i[cell_sorted].astype(np.int64)
            has_ion = n_i_cell > 0
            rounds = np.where(has_ion, local // np.maximum(n_i_cell, 1), -1)
            for m in range(int(rounds.max()) + 1 if has_ion.any() else 0):
                sel = np.flatnonzero(rounds == m)
                if sel.size == 0:
                    continue
                c = cell_sorted[sel]
                ie = order_e[sel]
                ion_local = (local[sel] + shift[c]) % n_i_cell[sel]
                ii = order_i[starts_i[c] + ion_local]
                va_new, vb_new, s = self._collide(ve[ie], vi[ii], self.m_e, self.m_i, -ELEMENTARY_CHARGE_C, self.q_i,
                                                  n_i[c], lnl_cell[c], np.full(sel.size, dt_c), rng)
                ve[ie] = va_new
                vi[ii] = vb_new
                tally.ei_pairs += float(sel.size)
                tally.ei_s_sum += float(s.sum())
                tally.ei_large_s += float(np.count_nonzero(s > 1.0))
                tally.ei_lnl_sum += float(lnl_cell[c].sum())
                tally.ei_s_max = max(tally.ei_s_max, float(s.max()))
                np.add.at(cell_ei_s.reshape(-1), c, s)
                np.add.at(cell_ei_pairs.reshape(-1), c, 1.0)
        ke_after = float(np.sum(relativistic_kinetic_energy_j(ve, self.m_e))) + float(np.sum(relativistic_kinetic_energy_j(vi, self.m_i)))
        pz_after = self.m_e * float(np.sum(ve[:, 2])) + self.m_i * float(np.sum(vi[:, 2]))
        tally.ke_coulomb_j = (ke_after - ke_before) * w
        tally.pz_coulomb = (pz_after - pz_before) * w
        new_e = ParticleArrays(electrons.r_m, electrons.z_m, ve[:, 0], ve[:, 1], ve[:, 2])
        new_i = ParticleArrays(ions.r_m, ions.z_m, vi[:, 0], vi[:, 1], vi[:, 2])
        return CoulombResult(new_e, new_i, tally, cell_ee_s, cell_ee_pairs, cell_ei_s, cell_ei_pairs, cell_electron_seconds)

    def to_dict(self) -> dict[str, Any]:
        return {"config": self.config.to_dict(), "cells": self.n_cells, "macro_weight": self.macro_weight,
                "ledger_keys": list(COULOMB_KEYS), "ion_mass_kg": self.m_i}


def coulomb_frequencies(delta: dict[str, float], dt_c: float) -> dict[str, float]:
    """Interval mean frequencies from ledger differences: ``nu_ee = 2 sum s_ee / (sum_cycles N_e dt_c)``, ``nu_ei = sum s_ei / (...)``,
    ``nu_ii = 2 sum s_ii / (sum_cycles N_i dt_c)``; mean s and lnL per pair; the fraction of pairs beyond s = 1."""

    e_cycles = float(delta.get("coulomb_electron_cycles", 0.0))
    i_cycles = float(delta.get("coulomb_ion_cycles", 0.0))
    out: dict[str, float] = {}
    for pair, cycles, factor in (("ee", e_cycles, 2.0), ("ei", e_cycles, 1.0), ("ii", i_cycles, 2.0)):
        pairs = float(delta.get(f"coulomb_{pair}_pairs", 0.0))
        s_sum = float(delta.get(f"coulomb_{pair}_s_sum", 0.0))
        out[f"nu_{pair}_mean_per_s"] = factor * s_sum / (cycles * dt_c) if cycles > 0.0 and dt_c > 0.0 else 0.0
        out[f"mean_s_{pair}"] = s_sum / pairs if pairs > 0.0 else 0.0
        out[f"fraction_large_s_{pair}"] = float(delta.get(f"coulomb_{pair}_large_s", 0.0)) / pairs if pairs > 0.0 else 0.0
        out[f"mean_coulomb_log_{pair}"] = float(delta.get(f"coulomb_{pair}_lnl_sum", 0.0)) / pairs if pairs > 0.0 else 0.0
        out[f"interval_{pair}_pairs"] = pairs
    return out


def cell_maps_to_nodes(cell_map: np.ndarray, node_shape: tuple[int, int]) -> np.ndarray:
    """Store a cell-shaped ``(nr, nz)`` map in the node-shaped ``(nr+1, nz+1)`` layout of the window sums (cell (i, j) at
    node index (i, j); last row / column zero) so the frame recorder and maps.npz carry it unchanged."""

    out = np.zeros(node_shape, dtype=np.float64)
    out[: cell_map.shape[0], : cell_map.shape[1]] = cell_map
    return out


def column_frequency_profile(nu_map: np.ndarray, weight_map: np.ndarray, grid: Grid2D, z_planes_m: tuple[float, ...] | list[float]) -> dict[str, float]:
    """Electron-weighted mean of a cell frequency map over the cell column nearest each ``z`` plane (per-cusp reading)."""

    out: dict[str, float] = {}
    for z in z_planes_m:
        j = int(np.clip(np.floor((z - grid.geometry.z_min_m) / grid.dz_m), 0, grid.axial_cells - 1))
        weights = weight_map[: grid.radial_cells, j]
        values = nu_map[: grid.radial_cells, j]
        total = float(weights.sum())
        out[f"{z * 1e3:.3f}mm"] = float(np.sum(values * weights) / total) if total > 0.0 else 0.0
    return out


__all__ = [
    "COULOMB_KEYS",
    "COULOMB_MODEL",
    "COULOMB_RNG_STREAM",
    "COULOMB_SPECIES_PAIRS",
    "CoulombConfig",
    "CoulombOperator",
    "CoulombResult",
    "CoulombTally",
    "cell_maps_to_nodes",
    "cell_moments",
    "cell_volumes_m3",
    "column_frequency_profile",
    "coulomb_frequencies",
    "coulomb_log_ee",
    "coulomb_log_ei",
    "coulomb_log_ii",
    "deflection_parameter",
    "nanbu_cos_chi",
    "nanbu_inverse_a",
    "relativistic_kinetic_energy_j",
    "scatter_pairs",
    "spitzer_electron_ion_momentum_rate",
    "temperature_equilibration_rate",
    "trubnikov_isotropization_rate",
]
