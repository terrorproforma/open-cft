"""L2 v2: cell partition from the v3.1 catalogue and Maxwellian rates from the PIC cross sections."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from cft_revival.hybrid.cells import (
    CellPartition,
    load_reference_partition,
    load_sealed_catalogue,
    synthetic_partition,
)
from cft_revival.hybrid.models import HybridValidationError
from cft_revival.hybrid.rates import build_rate_table, maxwellian_rate_m3_per_s
from cft_revival.pic2d.mcc import UniformSigmaTable, XenonCrossSections
from cft_revival.pic2d.models import ChannelGeometry, Grid2D

MODERN = Path(__file__).resolve().parents[2]
CATALOGUE_RESULTS = MODERN / "experiments" / "cusp_topology_search_v3_1" / "results"
PIC_PLANES = (0.006028, 0.012, 0.017972)


def pic_grid(nr: int = 60, nz: int = 480) -> Grid2D:
    return Grid2D(ChannelGeometry(0.002, 0.0, 0.024, 0.018, 0.003), nr, nz)


def test_synthetic_partition_tiles_and_maps_nodes() -> None:
    part = synthetic_partition(0.0, 0.024, [0.012, 0.006])
    assert part.cell_count == 3
    assert part.cusp_z_m == (0.006, 0.012)
    assert part.z_start_m == (0.0, 0.006, 0.012) and part.z_end_m == (0.006, 0.012, 0.024)
    cells = part.cell_of_z(np.array([0.0, 0.0059, 0.006, 0.0119, 0.012, 0.024, 0.03]))
    assert cells.tolist() == [0, 0, 1, 1, 2, 2, -1]
    nodes = part.node_cells(pic_grid(6, 48))
    assert nodes.shape == (7, 49) and nodes.min() == 0 and nodes.max() == 2
    with pytest.raises(HybridValidationError):
        synthetic_partition(0.0, 0.024, [0.03])


def test_partition_invariants_fail_closed() -> None:
    with pytest.raises(HybridValidationError):
        CellPartition("d", "s", "SYNTHETIC_PARTITION", ("a", "b"), ("anode_partial", "exit_partial"), (0.0, 0.01), (0.01, 0.02), (0.011,), None, "x")
    with pytest.raises(HybridValidationError):
        CellPartition("d", "s", "SYNTHETIC_PARTITION", ("a", "b"), ("anode_partial", "exit_partial"), (0.0, 0.012), (0.01, 0.02), (0.01,), None, "x")


def test_reference_partition_matches_the_pic_cusp_planes() -> None:
    part = load_reference_partition(CATALOGUE_RESULTS, set_id="p2_divergent_exit", design_id="divergent-exit-stack", grid=pic_grid(),
                                    declared_cusp_planes_m=PIC_PLANES)
    assert part.label == "P2_QUALIFIED_FIELD_SEPARATRIX_CUSP_TOPOLOGY"
    assert part.cell_count == 4 and part.kinds == ("anode_partial", "interior", "interior", "exit_partial")
    # the catalogue planes ARE the PIC's 6.03 / 12.00 / 17.97 mm planes (to a micron)
    assert np.allclose(part.cusp_z_m, PIC_PLANES, atol=1.0e-6)
    # extended to the electrodes (catalogue straight dielectric is 1.0-18.0 mm)
    assert part.z_start_m[0] == 0.0 and part.z_end_m[-1] == 0.024
    assert part.catalogue_sha256 is not None and len(part.catalogue_sha256) == 64


def test_reference_partition_refuses_wrong_declared_planes_or_design() -> None:
    with pytest.raises(HybridValidationError, match="differ from the declared PIC planes"):
        load_reference_partition(CATALOGUE_RESULTS, set_id="p2_divergent_exit", design_id="divergent-exit-stack", grid=pic_grid(),
                                 declared_cusp_planes_m=(0.006, 0.012, 0.0185))
    with pytest.raises(HybridValidationError, match="not in the catalogue"):
        load_reference_partition(CATALOGUE_RESULTS, set_id="p2_divergent_exit", design_id="no-such-design", grid=pic_grid())


def test_sealed_catalogue_refuses_tampered_bytes(tmp_path: Path) -> None:
    root = tmp_path / "results"
    (root / "artifacts").mkdir(parents=True)
    shutil.copy(CATALOGUE_RESULTS / "manifest.json", root / "manifest.json")
    raw = (CATALOGUE_RESULTS / "artifacts" / "cusp-cell-catalogue.json").read_bytes()
    (root / "artifacts" / "cusp-cell-catalogue.json").write_bytes(raw)
    catalogue, digest = load_sealed_catalogue(root)
    assert catalogue["design_count"] == 281 and len(digest) == 64
    edited = json.loads(raw)
    edited["entries"][0]["wall_cusps"][0]["z_c_m"] += 1e-4
    (root / "artifacts" / "cusp-cell-catalogue.json").write_bytes(json.dumps(edited).encode())
    with pytest.raises(HybridValidationError, match="differ from the sealed manifest"):
        load_sealed_catalogue(root)


def test_rate_table_is_bound_to_the_cross_sections_and_matches_the_oracle() -> None:
    xs = XenonCrossSections.synthetic_for_tests()
    table = build_rate_table(xs, temperature_points=41)
    assert table.cross_section_payload_sha256 == xs.payload_sha256
    k_cold = float(table.rate("ionization", 0.3))
    k_hot = float(table.rate("ionization", 20.0))
    assert k_cold < 1e-6 * k_hot
    rates = table.rate("ionization", table.temperature_ev)
    assert np.all(np.diff(rates[rates > 0]) > 0.0)
    grid_table = UniformSigmaTable.build(xs)
    energy = np.arange(grid_table.point_count) * grid_table.energy_step_ev
    t = float(table.temperature_ev[25])
    oracle = maxwellian_rate_m3_per_s(energy, grid_table.table_m2[2], t)
    assert np.isclose(float(table.rate("ionization", t)), oracle, rtol=1e-12)
    # log-log interpolation between grid points stays close to the direct integral
    t_mid = float(np.sqrt(table.temperature_ev[25] * table.temperature_ev[26]))
    assert np.isclose(float(table.rate("ionization", t_mid)), maxwellian_rate_m3_per_s(energy, grid_table.table_m2[2], t_mid), rtol=0.05)
    assert len(table.sha256()) == 64 and table.to_dict()["rate_table_sha256"] == table.sha256()


def test_real_cross_section_rates_have_the_expected_magnitude() -> None:
    table = build_rate_table(XenonCrossSections.from_file(), temperature_points=61)
    k_iz_8 = float(table.rate("ionization", 8.0))
    # Maxwellian xenon ionisation at 8 eV is of order 1e-14 m^3/s (Goebel-Katz), well below the PIC's effective 2.2e-14
    assert 3e-15 < k_iz_8 < 3e-14
    assert float(table.rate("elastic", 8.0)) > k_iz_8
