"""Quasi-steady 0-D neutral inventory (model v1.3).

One scalar neutral density ``n_g(t)`` (uniform in space, as the static background it
replaces) obeys the atom balance of the channel volume ``V``::

    V dn_g/dt = Q_in - S(t) - c n_g - (V / tau_g) (n_g - n_g*)

* ``Q_in`` is the prescribed feed [atoms/s];
* ``S`` is the ionisation rate measured from the MCC tallies over the update interval;
* ``c n_g`` is thermal effusion through the exit plane, ``c = v_bar A_exit / 4`` with the
  mean thermal speed ``v_bar = sqrt(8 k T / (pi m))`` at the documented neutral
  temperature;
* the last term is an ARTIFICIAL relaxation toward the quasi-steady fixed point
  ``n_g* = (Q_in - S) / c`` with time constant ``tau_g``.  Without it the inventory
  relaxes on the physical effusion time ``V / c`` (~0.2 ms for the CFT channel),
  ~100x longer than a feasible run.  The transient is therefore not physical; only
  the fixed point (where the artificial term vanishes and ``Q_in = S + c n_g``) is.

The linear ODE is integrated exactly over each update interval (``S`` held at its
measured interval mean), and the four atom ledgers (fed, ionised, effused,
artificial) are the exact time integrals of the corresponding terms, so
``fed - ionised - effused - artificial = V (n_g,1 - n_g,0)`` to round-off.

The MCC null-collision ceiling stays at the initial density ``n_g0``; the real
collision frequency is scaled by ``n_g / n_g0``.  ``n_g > n_g0`` (which would break
the null-collision bound) and ``n_g < 0`` (inventory exhausted) fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, expm1, isfinite, pi, sqrt
from typing import Any

import numpy as np

from .models import PIC2DStabilityError, PIC2DValidationError

BOLTZMANN_J_PER_K = 1.380649e-23
XENON_MASS_KG = 2.1801714e-25
NEUTRAL_LEDGER_KEYS = ("fed", "ionized", "effused", "artificial")


@dataclass(frozen=True, slots=True)
class NeutralInventoryConfig:
    """Prescribed feed and artificial relaxation time of the quasi-steady inventory."""

    feed_atoms_per_s: float
    relaxation_time_s: float

    def __post_init__(self) -> None:
        if not isfinite(self.feed_atoms_per_s) or self.feed_atoms_per_s <= 0.0:
            raise PIC2DValidationError("neutral feed must be positive")
        if not isfinite(self.relaxation_time_s) or self.relaxation_time_s <= 0.0:
            raise PIC2DValidationError("neutral relaxation time must be positive")

    def to_dict(self) -> dict[str, float]:
        return {"feed_atoms_per_s": self.feed_atoms_per_s, "relaxation_time_s": self.relaxation_time_s}


@dataclass(slots=True)
class NeutralState:
    density_per_m3: float
    ledger: dict[str, float]

    @classmethod
    def initial(cls, density_per_m3: float) -> "NeutralState":
        return cls(float(density_per_m3), {key: 0.0 for key in NEUTRAL_LEDGER_KEYS})

    def copy(self) -> "NeutralState":
        return NeutralState(self.density_per_m3, dict(self.ledger))

    def to_array(self) -> np.ndarray:
        return np.array([self.density_per_m3, *(self.ledger[key] for key in NEUTRAL_LEDGER_KEYS)], dtype=np.float64)

    @classmethod
    def from_array(cls, values: np.ndarray) -> "NeutralState":
        values = np.asarray(values, dtype=np.float64)
        if values.shape != (1 + len(NEUTRAL_LEDGER_KEYS),) or not np.isfinite(values).all():
            raise PIC2DValidationError("neutral state array has the wrong shape or is nonfinite")
        return cls(float(values[0]), {key: float(v) for key, v in zip(NEUTRAL_LEDGER_KEYS, values[1:], strict=True)})

    def to_dict(self) -> dict[str, Any]:
        return {"density_per_m3": self.density_per_m3, "ledger": dict(self.ledger)}


def mean_thermal_speed_m_per_s(temperature_k: float, mass_kg: float = XENON_MASS_KG) -> float:
    return sqrt(8.0 * BOLTZMANN_J_PER_K * temperature_k / (pi * mass_kg))


def effusion_coefficient_m3_per_s(exit_area_m2: float, temperature_k: float, mass_kg: float = XENON_MASS_KG) -> float:
    """``c = v_bar A / 4``: effusion rate is ``c n_g`` [atoms/s]."""

    return 0.25 * mean_thermal_speed_m_per_s(temperature_k, mass_kg) * exit_area_m2


def feed_for_density(density_per_m3: float, exit_area_m2: float, temperature_k: float, mass_kg: float = XENON_MASS_KG) -> float:
    """Feed that balances effusion at ``density_per_m3`` with zero ionisation."""

    return density_per_m3 * effusion_coefficient_m3_per_s(exit_area_m2, temperature_k, mass_kg)


def mass_flow_mg_per_s(atoms_per_s: float, mass_kg: float = XENON_MASS_KG) -> float:
    return atoms_per_s * mass_kg * 1.0e6


@dataclass(frozen=True, slots=True)
class NeutralAdvance:
    """One update: the new state plus the interval terms (all in atoms or atoms/s)."""

    state: NeutralState
    fixed_point_per_m3: float
    ionization_rate_per_s: float
    effusion_rate_per_s: float
    artificial_rate_per_s: float
    ledger_residual_atoms: float


class NeutralInventory:
    """Exact integrator of the inventory ODE with the null-collision ceiling as a hard bound."""

    def __init__(
        self,
        config: NeutralInventoryConfig,
        *,
        ceiling_density_per_m3: float,
        exit_area_m2: float,
        temperature_k: float,
        volume_m3: float,
        mass_kg: float = XENON_MASS_KG,
    ) -> None:
        if not isfinite(ceiling_density_per_m3) or ceiling_density_per_m3 <= 0.0:
            raise PIC2DValidationError("neutral ceiling density must be positive")
        if not isfinite(exit_area_m2) or exit_area_m2 <= 0.0 or not isfinite(volume_m3) or volume_m3 <= 0.0:
            raise PIC2DValidationError("exit area and channel volume must be positive")
        self.config = config
        self.ceiling = float(ceiling_density_per_m3)
        self.exit_area_m2 = float(exit_area_m2)
        self.temperature_k = float(temperature_k)
        self.volume_m3 = float(volume_m3)
        self.mass_kg = float(mass_kg)
        self.effusion_coefficient = effusion_coefficient_m3_per_s(exit_area_m2, temperature_k, mass_kg)
        self.zero_ionization_density = config.feed_atoms_per_s / self.effusion_coefficient
        if self.zero_ionization_density > self.ceiling * (1.0 + 1e-12):
            raise PIC2DValidationError(
                "neutral feed exceeds the null-collision ceiling: zero-ionisation density "
                f"{self.zero_ionization_density:.4g} > n_g0 {self.ceiling:.4g} m^-3"
            )
        self.rate = self.effusion_coefficient / self.volume_m3 + 1.0 / config.relaxation_time_s

    @property
    def physical_time_constant_s(self) -> float:
        return self.volume_m3 / self.effusion_coefficient

    def fixed_point(self, ionization_rate_per_s: float) -> float:
        """Quasi-steady density where feed = ionisation + effusion (may be negative if S > Q_in)."""

        return (self.config.feed_atoms_per_s - ionization_rate_per_s) / self.effusion_coefficient

    def scale(self, state: NeutralState) -> float:
        return state.density_per_m3 / self.ceiling

    def advance(self, state: NeutralState, ionization_rate_per_s: float, interval_s: float) -> NeutralAdvance:
        if not isfinite(ionization_rate_per_s) or ionization_rate_per_s < 0.0 or not isfinite(interval_s) or interval_s <= 0.0:
            raise PIC2DValidationError("ionisation rate must be finite and non-negative, interval positive")
        q, s = self.config.feed_atoms_per_s, ionization_rate_per_s
        c, v, tau, r = self.effusion_coefficient, self.volume_m3, self.config.relaxation_time_s, self.rate
        n0 = state.density_per_m3
        n_star = self.fixed_point(s)
        # dn/dt = a - r n with a = (Q - S)/V + n*/tau  (= r n* when n* is the unclamped fixed point)
        a = (q - s) / v + n_star / tau
        n_inf = a / r
        growth = -expm1(-r * interval_s)                # 1 - exp(-r dt), accurate for small r dt
        n1 = n_inf + (n0 - n_inf) * (1.0 - growth)
        integral_n = n_inf * interval_s + (n0 - n_inf) * growth / r   # int n dt over the interval
        fed = q * interval_s
        ionized = s * interval_s
        effused = c * integral_n
        artificial = (v / tau) * (integral_n - n_star * interval_s)
        if n1 < 0.0:
            raise PIC2DStabilityError(
                f"neutral inventory exhausted: n_g would reach {n1:.3g} m^-3 (S = {s:.3g} /s > feed {q:.3g} /s)"
            )
        if n1 > self.ceiling * (1.0 + 1e-9):
            raise PIC2DStabilityError(f"neutral density {n1:.4g} exceeds the null-collision ceiling {self.ceiling:.4g} m^-3")
        ledger = {
            "fed": state.ledger["fed"] + fed,
            "ionized": state.ledger["ionized"] + ionized,
            "effused": state.ledger["effused"] + effused,
            "artificial": state.ledger["artificial"] + artificial,
        }
        residual = (fed - ionized - effused - artificial) - v * (n1 - n0)
        return NeutralAdvance(
            NeutralState(n1, ledger), n_star, s, c * n1, (v / tau) * (n1 - n_star), residual,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.config.to_dict(),
            "ceiling_density_per_m3": self.ceiling,
            "exit_area_m2": self.exit_area_m2,
            "neutral_temperature_k": self.temperature_k,
            "channel_volume_m3": self.volume_m3,
            "mean_thermal_speed_m_per_s": mean_thermal_speed_m_per_s(self.temperature_k, self.mass_kg),
            "effusion_coefficient_m3_per_s": self.effusion_coefficient,
            "zero_ionization_density_per_m3": self.zero_ionization_density,
            "physical_time_constant_s": self.physical_time_constant_s,
            "mass_flow_mg_per_s": mass_flow_mg_per_s(self.config.feed_atoms_per_s, self.mass_kg),
            "update": "exact exponential relaxation per series interval toward n_g* = (Q_in - S)/c; ledgers are exact interval integrals",
            "transient_is_artificial": True,
        }


__all__ = [
    "NEUTRAL_LEDGER_KEYS",
    "NeutralAdvance",
    "NeutralInventory",
    "NeutralInventoryConfig",
    "NeutralState",
    "effusion_coefficient_m3_per_s",
    "feed_for_density",
    "mass_flow_mg_per_s",
    "mean_thermal_speed_m_per_s",
]
