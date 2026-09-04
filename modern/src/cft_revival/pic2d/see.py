"""Model v2.2.0 ``see_dielectric_v1``: secondary electron emission (SEE) from the dielectric channel wall.

Physics
-------
An electron that reaches the dielectric wall (boundary code 3) with kinetic energy ``E`` and angle of
incidence ``theta`` (to the local wall normal) is absorbed - its charge joins the accumulated surface
charge of the wall node stencil and its kinetic energy is booked as ``ke_absorbed_wall_j`` exactly as
before - and it emits an integer number ``n`` of macro-electrons of the SAME macro weight, sampled
without bias from the total yield ``delta(E, theta)``:

    n = floor(delta) + Bernoulli(delta - floor(delta))                 (E[n] = delta)

Total yield (``yield_model = "vaughan_components"``): the Vaughan (1989, 1993) universal curve

    delta_V(E, theta) = delta_max(theta) * (v e^(1 - v))^k,  v = (E - E_0) / (E_max(theta) - E_0),
    k = k_rise (v < 1), 0.25 (1 <= v <= 3.6), tail 1.125 / v^0.35 (v > 3.6),
    E_max(theta) = E_max (1 + k_s theta^2 / 2 pi),  delta_max(theta) = delta_max (1 + k_s theta^2 / pi),

plus the optional low-energy elastic-reflection bump of Sydorenko (2006, thesis eq. 3.16; PoP 13
014501) ``gamma_e,max * b(E)`` with ``b`` rising as ``v_1 e^(1 - v_1)`` to its peak and decaying as
``(1 + v_2) e^(-v_2)`` beyond it.  Following the Sydorenko / EDIPIC three-component split (elastically
reflected, inelastically backscattered, true secondaries; the fractions r_e = 0.03 and r_i = 0.07 of the
Vaughan yield are Sydorenko's values for BN, the bump is all elastic), every emitted electron is drawn
as one component with probability proportional to its partial yield:

* elastically reflected: keeps the impact speed;
* inelastically backscattered: energy uniform in (0, E) (Sydorenko 2006);
* true secondary: flux-weighted half-Maxwellian at ``T_see`` (normal component Rayleigh, tangential
  Gaussian) - the same sampler as the cathode injection, so the energy distribution is
  ``E exp(-E/T_see) / T_see^2`` (mean 2 T_see) and the angular distribution is the cosine law.

Backscattered components are emitted with the cosine law about the inward wall normal (Sydorenko 2006:
"proportional to the cosine law over the polar angle ... independent of the primary angle of incidence").
The emitted macro-electrons are placed at the point where the impact segment crossed the wall face of
the last plasma cell (linear interpolation of the (r, z) path, nudged 1e-6 of a cell into the plasma),
i.e. at the wall potential, and pushed from the next step on.  The wall's surface charge changes by
``(-1 + n) e W`` per electron impact (absorbed minus emitted) and by ``(+1 + n_i) e W`` per ion impact
(ion-induced yield ``gamma_i``, constant, default 0), deposited with the same renormalised bilinear
weights as the absorbed charge.  No Hobbs-Wesson cap is imposed: when the local yield exceeds the
space-charge limit the PIC forms the virtual cathode itself (the emitted electrons are returned by the
non-monotonic sheath and re-absorbed); the effective yield and the wall potential are recorded so the
regime is diagnosable.

Material constants (``MATERIALS``)
----------------------------------
* ``BN``: Vaughan fit of the total electron emission yield of boron nitride measured by Villemant et al.
  2019 (EPL 127, 23001; 10-1000 eV) as tabulated by the PICLas code (SEE model 13: a = 2.016, b = 299 eV,
  c = 0.563, W = 0): ``delta_max 2.016 at 299 eV, threshold 0, k_rise 0.563``.  Consistency checks
  against the independent low-energy measurement of Dunaevsky, Raitses and Fisch 2003 (PoP 10, 2574;
  BN grade HP, Table 1): first crossover 35 eV (power fit E_1 = 35 eV, alpha = 0.5) and
  ``delta(10 eV) = 0.51`` (linear fit sigma_0 = 0.54) - both reproduced by the fit (pinned in the tests),
  so no extra low-energy bump is added for BN.  Maxwellian-flux-averaged yield 0.48 / 0.58 / 0.69 / 0.98 at
  T_e = 5 / 7 / 10 / 20 eV; critical (space-charge-limit) temperature 20.3 eV against Dunaevsky's linear-fit
  value 19.3 eV and Sydorenko's 18.3 eV.
* ``Al2O3``: DECLARED constants, not digitised from Tondu, Belhaj and Inguimbert 2011 (J. Appl. Phys.
  110, 093301, whose Al2O3 yield falls with electron exposure): maximum ``delta_max 6.4 at 650 eV`` from
  the Dawson 1966 (J. Appl. Phys. 37, 3644) alumina range 5-7 at 450-700 eV quoted by the USU / Dennison
  compilations, Vaughan's default threshold 12.5 eV, k_rise 0.56 (Vaughan 1993), and the Sydorenko
  low-energy elastic bump (``gamma_e,max 0.5`` at 7.5 eV, decay 10 eV) so the yield does not vanish
  below the threshold (no dielectric measured at low energy does).  First crossover 16 eV; flux-averaged
  yield 0.63 / 0.82 / 1.09 at T_e = 5 / 7 / 10 eV; critical temperature 8.7 eV - the more emissive wall,
  as the audit's "Al2O3 > BN in every effect" hypothesis requires.

Both presets carry their provenance string into ``config_sha256`` (``SEEConfig.to_dict``).

Ledger
------
Emitted kinetic energy is an injected term (``ke_see_emitted_j``), emitted axial momentum
``pz_see_emitted``; the impact energy stays in ``ke_absorbed_wall_j``.  Counts: ``see_impacts``
(electron impacts on the emitting wall), ``see_electrons`` (emitted, electron-induced),
``see_ion_induced_electrons``, ``see_backscattered`` (elastic + inelastic among the emitted),
``see_yield_sum`` (sum of delta over impacts: the expected emitted count), ``see_yield_clamped`` (impacts
whose integer yield hit ``max_emitted_per_impact``).  All keys are EXTRA ledger keys, present only when
SEE is on, so a configuration without SEE records exactly the v2.0.6 ledger.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite, pi
from typing import Any

import numpy as np

from .models import (
    ELECTRON_MASS_KG,
    ELEMENTARY_CHARGE_C,
    EV_J,
    Grid2D,
    ParticleArrays,
    PIC2DValidationError,
)

# Hobbs and Wesson 1967 (Plasma Phys. 9, 85): the classical sheath ceases to exist when the emitted flux
# reaches delta* = 1 - 8.3 sqrt(m_e / M); for xenon 0.983.  Diagnostic only (no cap is imposed).
XENON_MASS_KG = 2.1801714e-25
HOBBS_WESSON_CRITICAL_YIELD_XE = 1.0 - 8.3 * float(np.sqrt(ELECTRON_MASS_KG / XENON_MASS_KG))
# Hobbs-Wesson space-charge-limited sheath drop (cold emitted electrons, Bohm ions): ~1.02 T_e for xenon.
HOBBS_WESSON_SCL_DROP_TE = 1.02

# Wall-normal codes shared by the CPU reference and the Warp kernels (the inward normal of the crossed face).
NORMAL_MINUS_R = 0      # radial face of the last plasma cell: emit toward the axis
NORMAL_MINUS_Z = 1      # axial face at the high-z side of the cell: emit toward -z
NORMAL_PLUS_Z = 2       # axial face at the low-z side of the cell: emit toward +z

# Emission point offset from the crossed face, in cell fractions (inside the plasma cell).
FACE_NUDGE = 1.0e-6

# Hard cap on the integer yield per impact (device scratch sizing); clamped impacts are tallied.
DEFAULT_MAX_EMITTED_PER_IMPACT = 8

YIELD_MODELS = ("vaughan_components", "constant")


@dataclass(frozen=True, slots=True)
class SEEMaterial:
    """Declared SEE constants of one wall material (all energies in eV)."""

    name: str
    delta_max: float
    energy_max_ev: float
    energy_threshold_ev: float
    k_rise: float = 0.56
    k_fall: float = 0.25
    smoothness: float = 1.0
    elastic_fraction: float = 0.03          # Sydorenko 2006: r_e (elastically reflected share of the Vaughan yield)
    inelastic_fraction: float = 0.07        # Sydorenko 2006: r_i (inelastically backscattered share)
    low_energy_elastic_peak: float = 0.0    # Sydorenko 2006 eq. 3.16: gamma_e,max (0 = no bump)
    low_energy_elastic_peak_ev: float = 7.5
    low_energy_elastic_threshold_ev: float = 0.0
    low_energy_elastic_decay_ev: float = 10.0
    source: str = ""

    def __post_init__(self) -> None:
        for name in ("delta_max", "energy_max_ev", "k_rise", "k_fall", "low_energy_elastic_peak_ev", "low_energy_elastic_decay_ev"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise PIC2DValidationError(f"see material {self.name}: {name} must be positive")
        for name in ("energy_threshold_ev", "smoothness", "elastic_fraction", "inelastic_fraction", "low_energy_elastic_peak",
                     "low_energy_elastic_threshold_ev"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise PIC2DValidationError(f"see material {self.name}: {name} must be non-negative")
        if self.energy_threshold_ev >= self.energy_max_ev:
            raise PIC2DValidationError(f"see material {self.name}: energy_threshold_ev must be below energy_max_ev")
        if self.elastic_fraction + self.inelastic_fraction > 1.0:
            raise PIC2DValidationError(f"see material {self.name}: elastic + inelastic fractions must not exceed 1")
        if self.low_energy_elastic_threshold_ev >= self.low_energy_elastic_peak_ev:
            raise PIC2DValidationError(f"see material {self.name}: the elastic bump threshold must be below its peak energy")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name, "delta_max": self.delta_max, "energy_max_ev": self.energy_max_ev,
            "energy_threshold_ev": self.energy_threshold_ev, "k_rise": self.k_rise, "k_fall": self.k_fall,
            "smoothness": self.smoothness, "elastic_fraction": self.elastic_fraction, "inelastic_fraction": self.inelastic_fraction,
            "low_energy_elastic_peak": self.low_energy_elastic_peak, "low_energy_elastic_peak_ev": self.low_energy_elastic_peak_ev,
            "low_energy_elastic_threshold_ev": self.low_energy_elastic_threshold_ev,
            "low_energy_elastic_decay_ev": self.low_energy_elastic_decay_ev, "source": self.source,
        }


MATERIALS: dict[str, SEEMaterial] = {
    "BN": SEEMaterial(
        "BN", delta_max=2.016, energy_max_ev=299.0, energy_threshold_ev=0.0, k_rise=0.563,
        source=(
            "Vaughan fit of the BN total electron emission yield of Villemant, Belhaj, Sarrailh, Dadouch, Garrigues and "
            "Boniface 2019 (EPL 127, 23001, doi:10.1209/0295-5075/127/23001; 10-1000 eV) as tabulated by PICLas SEE "
            "model 13 (a 2.016, b 299 eV, c 0.563, W 0); reproduces the Dunaevsky, Raitses and Fisch 2003 (Phys. "
            "Plasmas 10, 2574, doi:10.1063/1.1568344) BN grade HP low-energy data: first crossover 35 eV (their "
            "power-fit E_1) and delta(10 eV) 0.51 vs their linear-fit sigma_0 0.54; component split r_e 0.03 / r_i "
            "0.07 per Sydorenko 2006 (Phys. Plasmas 13, 014501, doi:10.1063/1.2158698; thesis eqs 3.16-3.18)"
        ),
    ),
    "Al2O3": SEEMaterial(
        "Al2O3", delta_max=6.4, energy_max_ev=650.0, energy_threshold_ev=12.5, k_rise=0.56,
        low_energy_elastic_peak=0.5, low_energy_elastic_peak_ev=7.5, low_energy_elastic_threshold_ev=0.0,
        low_energy_elastic_decay_ev=10.0,
        source=(
            "DECLARED (not digitised): maximum 6.4 at 650 eV from the Dawson 1966 (J. Appl. Phys. 37, 3644, "
            "doi:10.1063/1.1708934) alumina range 5-7 at 450-700 eV as compiled by Dennison et al. (USU); Vaughan 1993 "
            "(IEEE TED 40, 830, doi:10.1109/16.202798) threshold 12.5 eV and k 0.56; low-energy elastic bump "
            "gamma_e,max 0.5 at 7.5 eV (decay 10 eV, declared) per Sydorenko 2006 thesis eq. 3.16; Tondu, Belhaj and "
            "Inguimbert 2011 (J. Appl. Phys. 110, 093301, doi:10.1063/1.3653820) report the Al2O3 yield FALLING with "
            "electron exposure - the value is a sensitivity bracket, not a measurement of the flight ceramic"
        ),
    ),
}


def vaughan_yield(
    energy_ev: np.ndarray | float, incidence_angle_rad: np.ndarray | float = 0.0, *,
    delta_max: float, energy_max_ev: float, energy_threshold_ev: float, smoothness: float = 1.0,
    k_rise: float = 0.56, k_fall: float = 0.25,
) -> np.ndarray:
    """Vaughan 1989 / 1993 secondary-electron yield ``delta(E, theta)`` (see the module docstring)."""

    energy = np.asarray(energy_ev, dtype=np.float64)
    theta = np.asarray(incidence_angle_rad, dtype=np.float64)
    e_max = energy_max_ev * (1.0 + smoothness * theta**2 / (2.0 * np.pi))
    d_max = delta_max * (1.0 + smoothness * theta**2 / np.pi)
    with np.errstate(invalid="ignore", divide="ignore"):
        v = (energy - energy_threshold_ev) / (e_max - energy_threshold_ev)
        k = np.where(v < 1.0, k_rise, k_fall)
        vv = np.maximum(v, 0.0)
        core = d_max * np.power(vv * np.exp(1.0 - vv), k)
        tail = d_max * 1.125 / np.power(np.maximum(v, 1e-300), 0.35)
        out = np.where(v <= 3.6, core, tail)
    return np.where(energy > energy_threshold_ev, out, 0.0)


def low_energy_elastic_yield(
    energy_ev: np.ndarray | float, *, peak: float, peak_ev: float, threshold_ev: float, decay_ev: float,
) -> np.ndarray:
    """Sydorenko 2006 (thesis eq. 3.16) elastic-reflection bump: ``peak * v_1 e^(1-v_1)`` up to the peak energy,
    ``peak * (1 + v_2) e^(-v_2)`` beyond it (``v_1 = (E - E_e0)/(E_e,max - E_e0)``, ``v_2 = (E - E_e,max)/Delta_e``)."""

    energy = np.asarray(energy_ev, dtype=np.float64)
    if peak <= 0.0:
        return np.zeros_like(energy)
    v1 = (energy - threshold_ev) / (peak_ev - threshold_ev)
    v2 = (energy - peak_ev) / decay_ev
    rising = peak * np.maximum(v1, 0.0) * np.exp(1.0 - np.maximum(v1, 0.0))
    with np.errstate(over="ignore"):
        falling = peak * (1.0 + np.maximum(v2, 0.0)) * np.exp(-np.maximum(v2, 0.0))
    out = np.where(energy < peak_ev, rising, falling)
    return np.where(energy > threshold_ev, out, 0.0)


@dataclass(frozen=True, slots=True)
class SEEConfig:
    """SEE from the dielectric wall (model v2.2.0).  Every field enters ``config_sha256`` through ``to_dict``.

    ``enabled = False`` records the block (virtual-yield diagnostics) without emission.  ``material`` selects a
    ``MATERIALS`` preset; ``overrides`` replaces individual preset constants (declared sensitivity brackets).
    ``yield_model = "constant"`` uses ``constant_yield`` at every energy and angle (tests; Brandt-style crude
    models), split by the material's elastic / inelastic fractions unless overridden.
    """

    enabled: bool = True
    material: str = "BN"
    yield_model: str = "vaughan_components"
    constant_yield: float = 0.0
    # constant model only: no emission below this impact energy (a constant yield >= 1 at EVERY energy would let the
    # secondaries returned by a space-charge-limited sheath breed without bound; the real curves fall well below 1 at
    # the emission energies - BN 0.14 at 1 eV - so the threshold stands in for that)
    constant_yield_threshold_ev: float = 0.0
    emission_temperature_ev: float = 2.0
    ion_induced_yield: float = 0.0
    max_emitted_per_impact: int = DEFAULT_MAX_EMITTED_PER_IMPACT
    space_charge_limit_yield: float = HOBBS_WESSON_CRITICAL_YIELD_XE
    overrides: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise PIC2DValidationError("see.enabled must be a bool")
        if self.material not in MATERIALS:
            raise PIC2DValidationError(f"see.material must be one of {sorted(MATERIALS)}")
        if self.yield_model not in YIELD_MODELS:
            raise PIC2DValidationError(f"see.yield_model must be one of {YIELD_MODELS}")
        for name in ("emission_temperature_ev", "space_charge_limit_yield"):
            value = getattr(self, name)
            if not isfinite(value) or value <= 0.0:
                raise PIC2DValidationError(f"see.{name} must be positive")
        for name in ("constant_yield", "constant_yield_threshold_ev", "ion_induced_yield"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0.0:
                raise PIC2DValidationError(f"see.{name} must be finite and non-negative")
        if isinstance(self.max_emitted_per_impact, bool) or not isinstance(self.max_emitted_per_impact, int) or self.max_emitted_per_impact < 1:
            raise PIC2DValidationError("see.max_emitted_per_impact must be a positive integer")
        if self.yield_model == "constant" and self.constant_yield > self.max_emitted_per_impact:
            raise PIC2DValidationError("see.constant_yield exceeds max_emitted_per_impact")
        allowed = {f.name for f in SEEMaterial.__dataclass_fields__.values()} - {"name", "source"}  # type: ignore[attr-defined]
        for key, value in self.overrides.items():
            if key not in allowed:
                raise PIC2DValidationError(f"see.overrides: unknown material constant {key!r}")
            if not isinstance(value, (int, float)) or isinstance(value, bool) or not isfinite(float(value)):
                raise PIC2DValidationError(f"see.overrides[{key!r}] must be a finite number")
        # validates the overridden constants as a whole
        self.resolved_material()

    def resolved_material(self) -> SEEMaterial:
        base = MATERIALS[self.material]
        if not self.overrides:
            return base
        values = base.to_dict()
        values.update({key: float(value) for key, value in self.overrides.items()})
        values["source"] = base.source + f"; overrides {dict(sorted(self.overrides.items()))}"
        return SEEMaterial(**values)

    def to_dict(self) -> dict[str, Any]:
        material = self.resolved_material()
        return {
            "model": "see_dielectric_v1", "enabled": self.enabled, "material": self.material, "yield_model": self.yield_model,
            "constant_yield": self.constant_yield, "constant_yield_threshold_ev": self.constant_yield_threshold_ev,
            "emission_temperature_ev": self.emission_temperature_ev,
            "ion_induced_yield": self.ion_induced_yield, "max_emitted_per_impact": self.max_emitted_per_impact,
            "space_charge_limit_yield": self.space_charge_limit_yield,
            "overrides": {key: float(value) for key, value in sorted(self.overrides.items())},
            "constants": material.to_dict(),
            "emission": {
                "integer_yield": "floor(delta) + Bernoulli(delta - floor(delta)); weight = impacting macro weight",
                "components": "elastic (impact speed kept) / inelastic (energy uniform in (0, E)) / true (flux half-Maxwellian at T_see, mean 2 T_see)",
                "angle": "cosine law about the inward wall normal (true secondaries: Rayleigh normal + Gaussian tangential sampler)",
                "position": "impact segment crossing of the last plasma cell face, nudged 1e-6 cell into the plasma",
                "surface_charge": "wall node stencil gains (-1 + n) e W per electron impact and (+1 + n_i) e W per ion impact",
                "space_charge_limit": "not imposed (emerges in the PIC); effective yield and wall potential recorded",
            },
        }

    # -- yield ----------------------------------------------------------------------------------------------
    def yield_components(self, energy_ev: np.ndarray | float, incidence_angle_rad: np.ndarray | float = 0.0,
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(total, elastic, inelastic)`` yields; true secondaries = total - elastic - inelastic."""

        material = self.resolved_material()
        energy = np.asarray(energy_ev, dtype=np.float64)
        theta = np.asarray(incidence_angle_rad, dtype=np.float64)
        if self.yield_model == "constant":
            total = np.where(energy > self.constant_yield_threshold_ev, float(self.constant_yield), 0.0) * np.ones(np.broadcast(energy, theta).shape)
            return total, total * material.elastic_fraction, total * material.inelastic_fraction
        base = vaughan_yield(energy, theta, delta_max=material.delta_max, energy_max_ev=material.energy_max_ev,
                             energy_threshold_ev=material.energy_threshold_ev, smoothness=material.smoothness,
                             k_rise=material.k_rise, k_fall=material.k_fall)
        bump = low_energy_elastic_yield(energy, peak=material.low_energy_elastic_peak, peak_ev=material.low_energy_elastic_peak_ev,
                                        threshold_ev=material.low_energy_elastic_threshold_ev, decay_ev=material.low_energy_elastic_decay_ev)
        elastic = material.elastic_fraction * base + bump
        inelastic = material.inelastic_fraction * base
        return base + bump, elastic, inelastic

    def yield_at(self, energy_ev: np.ndarray | float, incidence_angle_rad: np.ndarray | float = 0.0) -> np.ndarray:
        return self.yield_components(energy_ev, incidence_angle_rad)[0]


def first_crossover_ev(config: SEEConfig, *, upper_ev: float = 2000.0) -> float:
    """Lowest energy where the normal-incidence yield reaches 1 (bisection on a monotone-rising bracket)."""

    if config.yield_model == "constant":
        return 0.0 if config.constant_yield >= 1.0 else float("inf")
    energies = np.linspace(0.0, upper_ev, 20001)
    above = np.flatnonzero(config.yield_at(energies) >= 1.0)
    if above.size == 0:
        return float("inf")
    hi = float(energies[above[0]])
    lo = float(energies[max(above[0] - 1, 0)])
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if float(config.yield_at(mid)) < 1.0:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def maxwellian_flux_average_yield(config: SEEConfig, electron_temperature_ev: float, *, points: int = 20001) -> float:
    """Yield averaged over the impact-energy distribution of a Maxwellian flux at normal incidence,
    ``<delta> = int delta(E) E exp(-E/T) dE / T^2`` (a Maxwellian wall flux without the sheath's angular focusing)."""

    t = float(electron_temperature_ev)
    energies = np.linspace(0.0, 40.0 * t, points)
    weights = energies * np.exp(-energies / t) / t**2
    return float(np.trapezoid(config.yield_at(energies) * weights, energies))


def critical_temperature_ev(config: SEEConfig) -> float:
    """Electron temperature at which the flux-averaged yield reaches the Hobbs-Wesson limit (inf if never)."""

    lo, hi = 0.1, 400.0
    if maxwellian_flux_average_yield(config, hi) < config.space_charge_limit_yield:
        return float("inf")
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if maxwellian_flux_average_yield(config, mid) < config.space_charge_limit_yield:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def classical_sheath_drop_te(delta: float, ion_mass_kg: float = XENON_MASS_KG) -> float:
    """Floating-wall sheath drop in units of T_e for a Maxwellian plasma with Bohm ions and yield ``delta``:
    ``ln[(1 - delta) sqrt(M / 2 pi m_e)]`` (Hobbs and Wesson 1967 below the space-charge limit)."""

    if delta >= 1.0:
        raise PIC2DValidationError("the classical floating sheath requires delta < 1")
    return float(np.log((1.0 - delta) * np.sqrt(ion_mass_kg / (2.0 * pi * ELECTRON_MASS_KG))))


# -- sampling primitives (shared semantics with warp_see.py) --------------------------------------------------

def sample_integer_yield(delta: np.ndarray, u: np.ndarray, maximum: int) -> np.ndarray:
    """``floor(delta) + (u < frac(delta))`` clamped to ``maximum`` (unbiased below the clamp)."""

    delta = np.asarray(delta, dtype=np.float64)
    base = np.floor(delta)
    n = base.astype(np.int64) + (np.asarray(u) < (delta - base)).astype(np.int64)
    return np.minimum(n, int(maximum))


def wall_crossing(grid: Grid2D, plasma_cell: np.ndarray, r_old: np.ndarray, z_old: np.ndarray, r_new: np.ndarray, z_new: np.ndarray,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Emission point and inward-normal code of a wall impact from the pre- and post-push positions.

    The path is linearly interpolated in (r, z); the first face of the old (plasma) cell that the path crosses
    is the wall unless the cell behind it is plasma, in which case the second crossed face is the wall (the
    Courant limit allows at most one diagonal cell change).  The emission point is on the crossed face,
    ``FACE_NUDGE`` cells inside the old cell, with the tangential coordinate clamped into that cell.
    """

    dr, dz, z_min = grid.dr_m, grid.dz_m, grid.geometry.z_min_m
    nr, nz = grid.cell_shape
    fr0 = r_old / dr
    fz0 = (z_old - z_min) / dz
    fr1 = r_new / dr
    fz1 = (z_new - z_min) / dz
    i0 = np.clip(np.floor(fr0).astype(np.int64), 0, nr - 1)
    j0 = np.clip(np.floor(fz0).astype(np.int64), 0, nz - 1)
    inf = np.inf
    with np.errstate(divide="ignore", invalid="ignore"):
        t_r = np.where(fr1 > i0 + 1, (i0 + 1 - fr0) / (fr1 - fr0), inf)
        forward = fz1 >= j0 + 1
        backward = fz1 < j0
        t_z = np.where(forward, (j0 + 1 - fz0) / (fz1 - fz0), np.where(backward, (j0 - fz0) / (fz1 - fz0), inf))
    code_z = np.where(forward, NORMAL_MINUS_Z, NORMAL_PLUS_Z)
    j_behind = np.where(forward, j0 + 1, j0 - 1)
    # cell behind the radial face / behind the axial face (False when outside the box)
    radial_behind_plasma = np.zeros(fr0.shape, dtype=bool)
    ok = i0 + 1 < nr
    radial_behind_plasma[ok] = plasma_cell[i0[ok] + 1, j0[ok]]
    axial_behind_plasma = np.zeros(fr0.shape, dtype=bool)
    ok = (j_behind >= 0) & (j_behind < nz)
    axial_behind_plasma[ok] = plasma_cell[i0[ok], j_behind[ok]]
    has_r = np.isfinite(t_r)
    has_z = np.isfinite(t_z)
    radial_first = t_r <= t_z
    # radial wall: radial face crossed (first, and the cell behind it is solid or no axial face follows) or the
    # axial face was crossed first into a plasma cell and the radial face follows
    radial_wall = (has_r & radial_first & ~(radial_behind_plasma & has_z)) | (has_z & ~radial_first & axial_behind_plasma & has_r)
    axial_wall = ~radial_wall & has_z
    fallback = ~radial_wall & ~axial_wall
    t = np.where(radial_wall, t_r, np.where(axial_wall, t_z, 0.0))
    t = np.clip(np.where(np.isfinite(t), t, 0.0), 0.0, 1.0)
    r_e = r_old + t * (r_new - r_old)
    z_e = z_old + t * (z_new - z_old)
    code = np.where(radial_wall | fallback, NORMAL_MINUS_R, code_z).astype(np.int64)
    # snap onto the face (nudged into the old cell) and clamp the tangential coordinate into the old cell
    r_face = (i0 + 1 - FACE_NUDGE) * dr
    z_lo = z_min + (j0 + FACE_NUDGE) * dz
    z_hi = z_min + (j0 + 1 - FACE_NUDGE) * dz
    r_lo = i0 * dr
    r_e = np.where(radial_wall | fallback, r_face, np.clip(r_e, r_lo, r_face))
    z_e = np.where(axial_wall, np.where(code == NORMAL_MINUS_Z, z_hi, z_lo), np.clip(z_e, z_lo, z_hi))
    return r_e, z_e, code


def emission_velocities(speed: np.ndarray, code: np.ndarray, u1: np.ndarray, u2: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Cosine-law direction about the inward normal ``code`` at the given speed: ``cos theta = sqrt(u1)``."""

    cos_t = np.sqrt(u1)
    sin_t = np.sqrt(np.maximum(1.0 - u1, 0.0))
    phi = 2.0 * np.pi * u2
    v_n = speed * cos_t
    v_a = speed * sin_t * np.cos(phi)
    v_b = speed * sin_t * np.sin(phi)
    return _orient(v_n, v_a, v_b, code)


def _orient(v_n: np.ndarray, v_a: np.ndarray, v_b: np.ndarray, code: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map (normal, tangential a, tangential b) onto (v_r, v_theta, v_z) for the inward-normal code."""

    radial = code == NORMAL_MINUS_R
    minus_z = code == NORMAL_MINUS_Z
    vr = np.where(radial, -v_n, v_a)
    vt = np.where(radial, v_a, v_b)
    vz = np.where(radial, v_b, np.where(minus_z, -v_n, v_n))
    return vr, vt, vz


def thermal_emission_velocities(thermal_speed: float, code: np.ndarray, u_a: np.ndarray, u_b: np.ndarray, u_c: np.ndarray,
                                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flux-weighted half-Maxwellian at ``v_th = sqrt(T/m)``: normal Rayleigh, tangential Gaussian (cosine angular law,
    energy ``E e^(-E/T) / T^2``) - the cathode injector's sampler."""

    v_n = thermal_speed * np.sqrt(-2.0 * np.log(np.maximum(u_a, 1e-300)))
    radial = np.sqrt(-2.0 * np.log(np.maximum(u_b, 1e-300)))
    v_a = thermal_speed * radial * np.cos(2.0 * np.pi * u_c)
    v_b = thermal_speed * radial * np.sin(2.0 * np.pi * u_c)
    return _orient(v_n, v_a, v_b, code)


@dataclass(slots=True)
class SEEEmission:
    """Result of one CPU emission pass: the new macro-electrons plus the ledger and diagnostic tallies."""

    particles: ParticleArrays
    impacts: int                      # electron impacts on the emitting wall (0 for the ion pass)
    emitted: int                      # emitted macro-electrons
    backscattered: int                # elastic + inelastic among them
    yield_sum: float                  # sum of delta over impacts
    clamped: int                      # impacts whose integer yield hit the cap
    kinetic_energy_j: float           # W-scaled emitted kinetic energy
    momentum_z: float                 # W-scaled emitted axial momentum
    emitted_per_impact: np.ndarray    # integer yield of every impact (for the surface charge)
    column: np.ndarray                # axial cell of every emitted electron
    column_energy_j: np.ndarray       # its W-scaled kinetic energy


def emit_secondaries(
    config: SEEConfig, grid: Grid2D, plasma_cell: np.ndarray, *, is_electron: bool, old: ParticleArrays, hit: ParticleArrays,
    impact_kinetic_energy_j: np.ndarray, macro_weight: float, rng: np.random.Generator, emitting: np.ndarray | None = None,
) -> SEEEmission:
    """CPU reference emission for the wall impacts ``hit`` (post-push state) with pre-push state ``old``.

    ``impact_kinetic_energy_j`` is the W-scaled kinetic energy of each impact; ``emitting`` masks the impacts on the
    emitting (dielectric) part of the wall (None = all).  Random draws: one uniform per impact (integer yield), then
    six per emitted electron (component, two direction / three thermal) - the order is part of the CPU reference.
    """

    count = hit.count
    empty = SEEEmission(ParticleArrays.empty(), 0, 0, 0, 0.0, 0, 0.0, 0.0, np.zeros(count, dtype=np.int64), np.zeros(0, dtype=np.int64), np.zeros(0))
    if count == 0:
        return empty
    mask = np.ones(count, dtype=bool) if emitting is None else np.asarray(emitting, dtype=bool)
    r_e, z_e, code = wall_crossing(grid, plasma_cell, old.r_m, old.z_m, hit.r_m, hit.z_m)
    speed2 = hit.speed_squared()
    speed = np.sqrt(speed2)
    energy_ev = impact_kinetic_energy_j / (macro_weight * EV_J)
    if is_electron:
        v_normal = np.where(code == NORMAL_MINUS_R, hit.vr_m_per_s, hit.vz_m_per_s)
        with np.errstate(invalid="ignore", divide="ignore"):
            cos_inc = np.where(speed > 0.0, np.abs(v_normal) / np.maximum(speed, 1e-300), 1.0)
        theta = np.arccos(np.clip(cos_inc, 0.0, 1.0))
        total, elastic, inelastic = config.yield_components(energy_ev, theta)
    else:
        total = np.full(count, float(config.ion_induced_yield))
        elastic = np.zeros(count)
        inelastic = np.zeros(count)
    total = np.where(mask, total, 0.0)
    u_count = rng.random(count)
    n_emit = sample_integer_yield(total, u_count, config.max_emitted_per_impact)
    n_emit = np.where(mask, n_emit, 0)
    clamped = int(np.count_nonzero(mask & (total > config.max_emitted_per_impact)))
    n_total = int(n_emit.sum())
    impacts = int(np.count_nonzero(mask)) if is_electron else 0
    yield_sum = float(total.sum())
    if n_total == 0:
        return SEEEmission(ParticleArrays.empty(), impacts, 0, 0, yield_sum, clamped, 0.0, 0.0, n_emit, np.zeros(0, dtype=np.int64), np.zeros(0))
    parent = np.repeat(np.arange(count), n_emit)
    u = rng.random((6, n_total))
    with np.errstate(invalid="ignore", divide="ignore"):
        p_elastic = np.where(total > 0.0, elastic / np.maximum(total, 1e-300), 0.0)[parent]
        p_inelastic = np.where(total > 0.0, inelastic / np.maximum(total, 1e-300), 0.0)[parent]
    is_elastic = u[0] < p_elastic
    is_inelastic = ~is_elastic & (u[0] < p_elastic + p_inelastic)
    is_true = ~is_elastic & ~is_inelastic
    code_p = code[parent]
    # backscattered: cosine law at the impact speed (elastic) or at the speed of a uniform fraction of the energy
    back_speed = np.where(is_elastic, speed[parent], np.sqrt(np.maximum(u[3], 0.0)) * speed[parent])
    vr_b, vt_b, vz_b = emission_velocities(back_speed, code_p, u[1], u[2])
    thermal = float(np.sqrt(EV_J * config.emission_temperature_ev / ELECTRON_MASS_KG))
    vr_t, vt_t, vz_t = thermal_emission_velocities(thermal, code_p, u[3], u[4], u[5])
    vr = np.where(is_true, vr_t, vr_b)
    vt = np.where(is_true, vt_t, vt_b)
    vz = np.where(is_true, vz_t, vz_b)
    particles = ParticleArrays(r_e[parent], z_e[parent], vr, vt, vz)
    c2 = 299792458.0**2
    s2 = particles.speed_squared()
    ke = (s2 / c2 / (1.0 + np.sqrt(1.0 - s2 / c2))) * ELECTRON_MASS_KG * c2 * macro_weight
    column = np.clip(((hit.z_m[parent] - grid.geometry.z_min_m) / grid.dz_m).astype(np.int64), 0, grid.axial_cells - 1)
    return SEEEmission(
        particles, impacts, n_total, int(np.count_nonzero(~is_true)), yield_sum, clamped, float(ke.sum()),
        float(np.sum(vz)) * ELECTRON_MASS_KG * macro_weight, n_emit, column, ke,
    )


def electron_charge_per_emitted_c(macro_weight: float) -> float:
    """Surface charge left on the wall by one emitted macro-electron (``+e W``)."""

    return ELEMENTARY_CHARGE_C * macro_weight


def see_birth_bound(electrons: int) -> int:
    """Device slots reserved per step for the wall's secondaries (the Warp backend fails closed at the next host sync
    if a step exceeds it): the wall electron flux is a small fraction of the population per step (~1 macro-electron per
    step at the 33 um plateau, a few during the seed transient), so 0.1 % of the electrons plus a fixed floor is a wide
    margin that costs ~10 % extra capacity over a 100-step sync interval."""

    return 256 + int(electrons) // 1000


__all__ = [
    "DEFAULT_MAX_EMITTED_PER_IMPACT",
    "FACE_NUDGE",
    "HOBBS_WESSON_CRITICAL_YIELD_XE",
    "HOBBS_WESSON_SCL_DROP_TE",
    "MATERIALS",
    "NORMAL_MINUS_R",
    "NORMAL_MINUS_Z",
    "NORMAL_PLUS_Z",
    "YIELD_MODELS",
    "SEEConfig",
    "SEEEmission",
    "SEEMaterial",
    "classical_sheath_drop_te",
    "critical_temperature_ev",
    "electron_charge_per_emitted_c",
    "emission_velocities",
    "emit_secondaries",
    "first_crossover_ev",
    "low_energy_elastic_yield",
    "maxwellian_flux_average_yield",
    "sample_integer_yield",
    "see_birth_bound",
    "thermal_emission_velocities",
    "vaughan_yield",
    "wall_crossing",
]
