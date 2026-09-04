"""Model v2.0.6 (2026-09-05): energy-ledger correction - ``inelastic_loss_j`` carries the macro weight.

Found by the external-validation v0 launch-1 diagnosis (036bd679): both backends added the MCC tally's per-MACRO-event
threshold energy ``(n_exc E_exc + n_ion E_ion) e`` to ``cumulative["inelastic_loss_j"]`` WITHOUT the macro weight W,
while every other ledger term (kinetic energies, absorbed / injected / born energies, field work) carries W.  The
recorded interval residual was therefore ``H - L_inel`` with ``H = field work + dU_field - electrode work`` the true
numerical energy creation and ``L_inel`` the W-scaled inelastic power: every recorded residual read too negative by the
inelastic power (7-14 % of the electrode work at the accepted plateaus; 1/W exactly on ss-v4, 047, 056-L1, attempts 7/8).

Regressions here:
* the tally stays per macro event (the operator does not know W); the ledger applies W and keeps the unscaled sum under
  ``inelastic_loss_per_weight_j``; the series record carries ``interval_inelastic_loss_j`` (W-scaled);
* the particle-side identity ``dKE = field work + injected - absorbed + born - W (n_exc E_exc + n_ion E_ion) e`` closes to
  round-off per record on the CPU reference, the Warp CPU backend and CUDA (when available), and the recorded residual
  equals ``H`` to the same round-off - the pre-v2.0.6 bookkeeping misses by the whole W-scaled loss;
* on a well-resolved, field-free, strongly collisional box (no electrode work, no heating source) the corrected residual is
  ~0 while the pre-v2.0.6 statistic reads ~ -100 % of the inelastic loss;
* the runner's series arrays gain ``interval_inelastic_loss_j`` (NaN on pre-v2.0.6 records); physics is untouched
  (diagnostic scalars only: same particles, same counts), and ``config_sha256`` does not change.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest

from cft_revival.pic2d import artifacts
from cft_revival.pic2d.fields import linear_psi_field_map, uniform_field_map
from cft_revival.pic2d.mcc import EV_J, MCCConfig, MCCTally, XenonCrossSections
from cft_revival.pic2d.models import (
    BoundaryPotentials,
    ChannelGeometry,
    Grid2D,
    PoissonConfig2D,
    StabilityLimits,
)
from cft_revival.pic2d.simulation import (
    CUMULATIVE_KEYS,
    INELASTIC_LOSS_PER_WEIGHT_KEY,
    InjectionConfig,
    PIC2DConfig,
    SeedPlasmaConfig,
    Simulation,
    empty_cumulative,
)
from experiments.pic2d_cft_steady_state_v1 import run as runner

CFT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 24.0e-3, 18.0e-3, 3.0e-3)
STRAIGHT_GEOMETRY = ChannelGeometry(2.0e-3, 0.0, 8.0e-3, 8.0e-3, 2.0e-3)


def _warp_backends() -> list[str]:
    try:
        from cft_revival.pic2d.warp_backend import device_available
    except ImportError:  # pragma: no cover
        return []
    return [name for name, device in (("warp-cpu", "cpu"), ("warp-cuda", "cuda:0")) if device_available(device)]


BACKENDS = ["cpu", *_warp_backends()]


def _discharge_config(grid: Grid2D, *, series: int = 25) -> PIC2DConfig:
    """The warp-parity discharge: 300 V, injection, MCC at 1e21, W 2e6 (every ledger term active)."""

    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(300.0, 0.0), dt_s=5e-12, macro_weight=2e6, seed=3,
        injection=InjectionConfig(0.05, 2.0), seed_plasma=SeedPlasmaConfig(1e16, 5.0), mcc=MCCConfig(1e21),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=1e16, reference_electron_temperature_ev=5.0,
        limits=StabilityLimits(max_cell_debye_ratio=2.0), series_interval_steps=series,
    )


def _interval_terms(sim: Simulation) -> dict[str, np.ndarray]:
    """Per-record interval quantities of the ledger identity (records 1..n)."""

    keys = ("field_work_j", "ke_injected_j", "ke_absorbed_anode_j", "ke_absorbed_exit_j", "ke_absorbed_wall_j", "ke_born_ions_j",
            "inelastic_loss_j", INELASTIC_LOSS_PER_WEIGHT_KEY)
    out: dict[str, list[float]] = {k: [] for k in ("dke", "rhs", "rhs_old", "residual", "h", "inelastic", "per_weight", "electrode", "interval_key")}
    for a, b in pairwise(sim.series):
        ca, cb = a.ledger["cumulative"], b.ledger["cumulative"]
        d = {key: float(cb.get(key, 0.0) - ca.get(key, 0.0)) for key in keys}
        sources = d["field_work_j"] + d["ke_injected_j"] - d["ke_absorbed_anode_j"] - d["ke_absorbed_exit_j"] - d["ke_absorbed_wall_j"] + d["ke_born_ions_j"]
        out["dke"].append((b.kinetic_electron_j + b.kinetic_ion_j) - (a.kinetic_electron_j + a.kinetic_ion_j))
        out["rhs"].append(sources - d["inelastic_loss_j"])
        out["rhs_old"].append(sources - d[INELASTIC_LOSS_PER_WEIGHT_KEY])          # the pre-v2.0.6 bookkeeping
        out["residual"].append(b.ledger["interval_residual_j"])
        out["h"].append(d["field_work_j"] + (b.field_energy_j - a.field_energy_j) - b.ledger["interval_electrode_work_j"])
        out["inelastic"].append(d["inelastic_loss_j"])
        out["per_weight"].append(d[INELASTIC_LOSS_PER_WEIGHT_KEY])
        out["electrode"].append(b.ledger["interval_electrode_work_j"])
        out["interval_key"].append(b.ledger["interval_inelastic_loss_j"])
    return {k: np.asarray(v, dtype=np.float64) for k, v in out.items()}


# -- the tally is per macro event; the ledger applies W -----------------------------------------------------------------

def test_tally_is_per_macro_event_and_the_ledger_applies_the_macro_weight_and_keeps_the_unscaled_sum():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    sim = Simulation(_discharge_config(grid), linear_psi_field_map(grid, 2.0), backend="cpu", cross_sections=xs)
    sim.run(200)
    thresholds = sim.backend.mcc.table.thresholds_ev
    assert thresholds[1] == pytest.approx(8.32) and thresholds[2] == pytest.approx(12.13)
    cumulative = sim.series[-1].ledger["cumulative"]
    n_exc, n_ion = cumulative["excitations"], cumulative["ionizations"]
    assert n_exc > 0 and n_ion > 0
    per_weight = (n_exc * thresholds[1] + n_ion * thresholds[2]) * EV_J
    assert cumulative[INELASTIC_LOSS_PER_WEIGHT_KEY] == pytest.approx(per_weight, rel=1e-12)          # the v2.0.5 quantity, kept
    assert cumulative["inelastic_loss_j"] == pytest.approx(per_weight * sim.config.macro_weight, rel=1e-12)   # v2.0.6: times W
    assert cumulative["inelastic_loss_j"] / cumulative[INELASTIC_LOSS_PER_WEIGHT_KEY] == pytest.approx(2e6, rel=1e-12)
    # the extra key is present from the start and never enters the fixed (checkpoint) key set
    assert INELASTIC_LOSS_PER_WEIGHT_KEY in empty_cumulative() and INELASTIC_LOSS_PER_WEIGHT_KEY not in CUMULATIVE_KEYS
    terms = _interval_terms(sim)
    assert np.allclose(terms["interval_key"], terms["inelastic"], rtol=1e-12, atol=0.0)          # series carries the W-scaled sink
    assert np.allclose(terms["inelastic"], terms["per_weight"] * 2e6, rtol=1e-12, atol=0.0)
    # the MCC tally itself is unchanged: per macro event, no W (the operator does not know the weight)
    tally = MCCTally(10, 3, 2, 1, 4, (2 * 8.32 + 1 * 12.13) * EV_J)
    assert tally.to_dict()["inelastic_energy_loss_j"] == pytest.approx((2 * 8.32 + 12.13) * EV_J)


# -- the particle-side identity closes to round-off on every backend --------------------------------------------------------

@pytest.mark.parametrize("backend", BACKENDS)
def test_particle_side_identity_closes_to_round_off_and_the_residual_equals_h(backend: str):
    """dKE = field work + injected - absorbed + born - W (n_exc E_exc + n_ion E_ion) e per record, and the recorded residual
    is H = field work + dU - electrode work.  Without W (the pre-v2.0.6 bookkeeping) the identity misses by the whole
    W-scaled loss: the residual was H - L_inel."""

    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    sim = Simulation(_discharge_config(grid), linear_psi_field_map(grid, 2.0), backend=backend, cross_sections=xs)
    sim.run(200)
    terms = _interval_terms(sim)
    scale = max(float(np.max(np.abs(terms["dke"]))), float(sim.series[-1].kinetic_electron_j))
    # the MCC removes the thresholds in the classical energy while the ledger is relativistic: the mismatch is
    # O(v^2/c^2) of the per-event loss ~1e-5 of L_inel, far inside 1e-6 of the kinetic energy scale
    closure = np.abs(terms["dke"] - terms["rhs"])
    assert float(closure.max()) <= 1e-6 * scale, (closure.max(), scale)
    assert np.allclose(terms["residual"], terms["h"], rtol=0.0, atol=1e-6 * scale)
    # the old bookkeeping does NOT close: it misses by (W - 1) L_inel / W ~ the whole W-scaled loss
    inelastic = float(terms["inelastic"].sum())
    assert inelastic > 1e-3 * scale                                             # collisions did happen at a visible level
    old_gap = float(np.abs(terms["dke"] - terms["rhs_old"]).sum())
    assert old_gap == pytest.approx(inelastic - float(terms["per_weight"].sum()), rel=1e-6)
    assert old_gap > 1e3 * float(closure.sum())
    # the recorded residual sums to H; the pre-v2.0.6 residual would have been H - L_inel (sums over n records: n x round-off)
    summed = 1e-6 * scale * terms["residual"].size
    assert float(terms["residual"].sum()) == pytest.approx(float(terms["h"].sum()), abs=summed)
    old_residual = terms["residual"] - (terms["inelastic"] - terms["per_weight"])
    expected_old = float(terms["h"].sum()) - inelastic + float(terms["per_weight"].sum())
    assert float(old_residual.sum()) == pytest.approx(expected_old, abs=summed)
    assert float(old_residual.sum()) < float(terms["residual"].sum()) - 0.99 * inelastic
    assert float(terms["electrode"].sum()) > 0.0


def _collisional_box_config(grid: Grid2D, backend_seed: int = 7) -> PIC2DConfig:
    """Field-free (0 V / 0 V), no injection, hot seed electrons (20 eV) on a dense static neutral background: the
    inelastic sink is the dominant energy flow, there is no electrode work and no heating source; the cells resolve the
    seed plasma's Debye length by a wide margin (n 3e15, 20 eV -> lambda_D 0.6 mm on 167 um cells)."""

    return PIC2DConfig(
        grid=grid, potentials=BoundaryPotentials(0.0, 0.0), dt_s=2e-12, macro_weight=1e5, seed=backend_seed,
        injection=None, seed_plasma=SeedPlasmaConfig(3e15, 20.0), mcc=MCCConfig(1e22),
        poisson=PoissonConfig2D(method="direct"), reference_density_per_m3=3e15, reference_electron_temperature_ev=20.0,
        max_electron_energy_ev=200.0, limits=StabilityLimits(max_cell_debye_ratio=4.0, max_omega_pe_dt=0.5),
        series_interval_steps=20,
    )


@pytest.mark.parametrize("backend", BACKENDS)
def test_corrected_residual_is_zero_on_a_no_heating_collisional_box_while_the_old_statistic_read_minus_the_loss(backend: str):
    grid = Grid2D(STRAIGHT_GEOMETRY, 12, 48)
    xs = XenonCrossSections.from_file()
    sim = Simulation(_collisional_box_config(grid), uniform_field_map(grid, 0.01), backend=backend, cross_sections=xs)
    sim.run(400)
    terms = _interval_terms(sim)
    loss = float(terms["inelastic"].sum())
    ke0 = float(sim.series[0].kinetic_electron_j)
    assert loss > 0.15 * ke0                                                   # the sink is the dominant energy flow
    assert float(np.abs(terms["electrode"]).max()) == 0.0                     # grounded box: no supply work
    corrected = float(terms["residual"].sum())
    old = float((terms["residual"] - (terms["inelastic"] - terms["per_weight"])).sum())
    assert abs(corrected) <= 0.005 * loss, (corrected, loss)                  # ~0: -0.01 ... -0.03 % of L_inel measured
    assert old == pytest.approx(-loss, rel=0.005)                             # the pre-v2.0.6 statistic: -100 % of L_inel
    cumulative = sim.series[-1].ledger["cumulative"]
    assert cumulative["excitations"] > 100 and cumulative["ionizations"] > 100


# -- physics untouched, identity untouched, series arrays ----------------------------------------------------------------

def test_ledger_fix_is_diagnostic_only_physics_and_config_identity_unchanged(tmp_path):
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    config = _discharge_config(grid)
    # the ledger fix has no configuration parameter: to_dict / config_sha256 are those of v2.0.5
    assert "inelastic" not in repr(config.to_dict())
    field = linear_psi_field_map(grid, 2.0)
    sim = Simulation(config, field, backend="cpu", cross_sections=xs)
    sim.run(100)
    state = sim.state
    # the particle state and the event counts are what they were: the fix only rescales one ledger scalar
    assert state.electrons.count > 0 and state.cumulative["ionizations"] > 0
    assert state.cumulative["inelastic_loss_j"] == pytest.approx(state.cumulative[INELASTIC_LOSS_PER_WEIGHT_KEY] * config.macro_weight, rel=1e-12)
    assert artifacts.config_identity(config) == artifacts.config_identity(_discharge_config(grid))
    # a checkpoint round trip carries the extra key like the v2.0 momentum keys (the fixed key set is untouched)
    json_path, _ = artifacts.save_checkpoint(tmp_path, "ck", state, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256, backend="cpu")
    metadata = artifacts.read_canonical_json(json_path)
    assert metadata["cumulative_keys"] == list(CUMULATIVE_KEYS) and INELASTIC_LOSS_PER_WEIGHT_KEY in metadata["cumulative_extra_keys"]
    loaded = artifacts.load_checkpoint(json_path, config, field_sha256=field.sha256, cross_section_sha256=xs.payload_sha256)
    assert loaded.cumulative[INELASTIC_LOSS_PER_WEIGHT_KEY] == state.cumulative[INELASTIC_LOSS_PER_WEIGHT_KEY]
    assert loaded.cumulative["inelastic_loss_j"] == state.cumulative["inelastic_loss_j"]


def test_records_to_arrays_carries_the_interval_inelastic_loss_and_tolerates_old_records():
    grid = Grid2D(CFT_GEOMETRY, 12, 96)
    xs = XenonCrossSections.from_file()
    sim = Simulation(_discharge_config(grid), linear_psi_field_map(grid, 2.0), backend="cpu", cross_sections=xs)
    sim.run(100)
    records = [record.to_dict() for record in sim.series]
    arrays = runner.records_to_arrays(records)
    assert "interval_inelastic_loss_j" in arrays and arrays["interval_inelastic_loss_j"].size == len(records)
    assert np.all(np.isfinite(arrays["interval_inelastic_loss_j"])) and float(arrays["interval_inelastic_loss_j"][1:].sum()) > 0.0
    assert float(arrays["interval_inelastic_loss_j"][1:].sum()) == pytest.approx(
        sim.series[-1].ledger["cumulative"]["inelastic_loss_j"] - sim.series[0].ledger["cumulative"]["inelastic_loss_j"], rel=1e-12)
    for record in records:            # pre-v2.0.6 records have no such key -> NaN, everything else unchanged
        record["ledger"].pop("interval_inelastic_loss_j")
    old = runner.records_to_arrays(records)
    assert np.all(np.isnan(old["interval_inelastic_loss_j"])) and np.array_equal(old["interval_residual_j"], arrays["interval_residual_j"])
