"""Physical constants and sheath coefficients used by the v2 sheath closure.

Every number here is either a CODATA/CIAAW constant or a coefficient with a
literature citation.  Nothing is fitted.

Citations (full records in ``docs/literature/reduced-models-cusp-topology-
blockers.md`` section 7 and ``docs/REFERENCES.md``):

* Lieberman, M. A., Lichtenberg, A. J., *Principles of Plasma Discharges and
  Materials Processing*, 2nd ed., Wiley (2005), eq. (6.2.17): the floating
  potential of a Maxwellian plasma against an absorbing wall,
  ``V_s = T_e ln[(M / (2 pi m_e))^(1/2)]``, obtained from the ambipolar flux
  balance ``(1/4) n_s v_e_bar exp(-V_s/T_e) = n_s u_B`` at a common sheath-edge
  density ``n_s`` (the density cancels).
* Hobbs, G. D., Wesson, J. A., "Heat flow through a Langmuir sheath in the
  presence of electron emission", *Plasma Physics* 9, 85-87 (1967): with a
  secondary-emission yield ``gamma`` the floating potential becomes
  ``T_e ln[(1-gamma) (M/(2 pi m_e))^(1/2)]`` until the space-charge limit,
  where the sheath saturates at ``~1.02 T_e`` and the critical yield is
  ``gamma_crit = 1 - 8.3 (m_e/M)^(1/2)``.
* Goebel, D. M., Katz, I., *Fundamentals of Electric Propulsion*, Wiley (2008),
  Ch. 4 (ring-cusp discharge model): plasma-electron loss to a cusp through the
  hybrid area ``sqrt(r_e r_i) L_c`` with the Boltzmann factor ``exp(-phi_s/T_e)``.
* Puca, N., Panelli, M., Battista, F., *Aerotecnica Missili & Spazio* 103(4),
  321-338 (2024), Table 3: the same closure adapted to the HEMP thruster with
  a leak-width prefactor of 1; Hershkowitz, Leung, Romesser, *Phys. Rev.
  Lett.* 35, 277 (1975) report leak widths of order four hybrid gyroradii.
"""

from __future__ import annotations

from math import log, pi, sqrt

# CODATA 2018/2022 values (identical to ``cft_revival.pic2d.models``).
ELEMENTARY_CHARGE_C = 1.602176634e-19
ELECTRON_MASS_KG = 9.1093837139e-31
ATOMIC_MASS_UNIT_KG = 1.66053906660e-27
XENON_ATOMIC_WEIGHT_U = 131.293  # CIAAW standard atomic weight
XENON_MASS_KG = XENON_ATOMIC_WEIGHT_U * ATOMIC_MASS_UNIT_KG

# Lieberman & Lichtenberg (6.2.17): flux-ratio factor of a Maxwellian floating
# sheath for xenon; the sheath drop without emission is ``T_e ln(K0)``.
MASS_FLUX_RATIO = sqrt(XENON_MASS_KG / (2.0 * pi * ELECTRON_MASS_KG))
FLOATING_SHEATH_COEFFICIENT = log(MASS_FLUX_RATIO)

# Hobbs & Wesson (1967): space-charge-limited sheath drop in units of T_e and
# the critical emission yield for xenon.
SPACE_CHARGE_LIMITED_COEFFICIENT = 1.02
CRITICAL_EMISSION_YIELD = 1.0 - 8.3 * sqrt(ELECTRON_MASS_KG / XENON_MASS_KG)

# Documented disagreement range of the cusp leak-width prefactor (in units of
# the hybrid gyroradius): Puca 2024 uses 1, Hershkowitz 1975 report ~4.
LEAK_WIDTH_PREFACTOR_RANGE = (1.0, 4.0)

__all__ = [
    "ATOMIC_MASS_UNIT_KG",
    "CRITICAL_EMISSION_YIELD",
    "ELECTRON_MASS_KG",
    "ELEMENTARY_CHARGE_C",
    "FLOATING_SHEATH_COEFFICIENT",
    "LEAK_WIDTH_PREFACTOR_RANGE",
    "MASS_FLUX_RATIO",
    "SPACE_CHARGE_LIMITED_COEFFICIENT",
    "XENON_ATOMIC_WEIGHT_U",
    "XENON_MASS_KG",
]
