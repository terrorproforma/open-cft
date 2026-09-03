"""Catalogue binding, counts, the pure-Python Beta machinery and the frozen per-design sample."""

from __future__ import annotations

import math
import random
import shutil

import pytest

from experiments.mdo_l0_campaign_v2 import catalogue as c
from experiments.mdo_l0_campaign_v2.experiment import protocol

GIT = shutil.which("git") is not None


@pytest.fixture(scope="module")
def designs() -> tuple[c.CatalogueDesign, ...]:
    return c.load_catalogue()


def test_dataset_binding_passes_and_is_fail_closed() -> None:
    report = c.binding_report(use_git=GIT)
    assert report["passed"], report["checks"]
    assert report["dataset_bytes"] == 2659656
    assert report["manifest_state"] == "accepted_result"
    if GIT:
        assert report["checks"]["dataset_git_blob"] and report["checks"]["result_commit_is_ancestor"]
    tampered = dict(c.DATASET_BINDING)
    tampered["dataset_file_sha256"] = "0" * 64
    with pytest.raises(c.CatalogueBindingError, match="dataset_file_sha256"):
        c.require_binding(tampered, use_git=False)


def test_catalogue_has_96_ordered_converged_designs_with_consistent_counts(designs) -> None:
    assert len(designs) == 96
    assert [d.index for d in designs] == list(range(96))
    assert all(d.case_id == f"l1a-gs-v2-{d.index:03d}-{d.design_id[:10]}" for d in designs)
    assert all(d.converged for d in designs)
    for d in designs:
        assert d.cell_trials == (128, 128, 128, 128)
        assert d.pooled_trials == 512
        assert sum(d.cell_wall_hits) == d.pooled_wall_hits
        assert all(0 <= s <= 128 for s in d.cell_wall_hits)
        assert 0.0 <= d.pooled_point_estimate <= 1.0
        assert set(d.geometry) >= {"chamber_length_m", "wall_radius_m", "stage_count", "stage_pitch_m", "has_divergent_exit"}
    saturated = sum(1 for d in designs if 128 in d.cell_wall_hits)
    value = protocol()
    assert saturated == value["catalogue_binding_identity"]["designs_with_a_saturated_cell_128_of_128"]
    pooled = [d.pooled_point_estimate for d in designs]
    assert [min(pooled), max(pooled)] == value["catalogue_binding_identity"]["pooled_wall_hit_probability_range"]
    assert min(pooled) == 0.375 and max(pooled) == 0.869140625  # dataset headline
    assert c.catalogue_sha256(designs) == value["catalogue_binding_identity"]["catalogue_sha256"]
    assert c.catalogue_sha256(designs) == c.catalogue_sha256(c.load_catalogue())


def test_incomplete_beta_identities_and_quantile_inversion() -> None:
    assert c.regularized_incomplete_beta(1.0, 1.0, 0.3) == pytest.approx(0.3, abs=1e-15)
    assert c.regularized_incomplete_beta(2.0, 1.0, 0.5) == pytest.approx(0.25, abs=1e-15)
    assert c.regularized_incomplete_beta(0.5, 0.5, 0.5) == pytest.approx(0.5, abs=1e-14)
    rng = random.Random(7)
    for _ in range(200):
        a = rng.choice([0.5, 1.5, 12.5, 64.5, 128.5, 300.5, 511.5])
        b = rng.choice([0.5, 1.5, 12.5, 64.5, 128.5, 300.5, 511.5])
        x = rng.random()
        assert c.regularized_incomplete_beta(a, b, x) + c.regularized_incomplete_beta(b, a, 1.0 - x) == pytest.approx(1.0, abs=1e-13)
        u = rng.random()
        q = c.beta_quantile(u, a, b)
        assert 0.0 < q < 1.0
        # x-accuracy: u lies between the CDF values eight ulps either side of the quantile
        below, above = q, q
        for _ in range(8):
            below, above = math.nextafter(below, 0.0), math.nextafter(above, 1.0)
        assert c.regularized_incomplete_beta(a, b, below) - 1e-13 <= u <= c.regularized_incomplete_beta(a, b, above) + 1e-13
    # monotone in u
    values = [c.beta_quantile(u / 33.0, 96.5, 32.5) for u in range(1, 33)]
    assert values == sorted(values)
    with pytest.raises(ValueError):
        c.beta_quantile(0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        c.regularized_incomplete_beta(0.0, 1.0, 0.5)


def test_beta_functions_agree_with_scipy_when_available() -> None:
    special = pytest.importorskip("scipy.special")
    rng = random.Random(1)
    worst_i = worst_q = 0.0
    for _ in range(300):
        a = rng.choice([0.5, 1.5, 12.5, 64.5, 128.5, 300.5, 511.5])
        b = rng.choice([0.5, 1.5, 12.5, 64.5, 128.5, 300.5, 511.5])
        x, u = rng.random(), rng.random()
        worst_i = max(worst_i, abs(c.regularized_incomplete_beta(a, b, x) - float(special.betainc(a, b, x))))
        worst_q = max(worst_q, abs(c.beta_quantile(u, a, b) - float(special.betaincinv(a, b, u))))
    assert worst_i < 1e-12 and worst_q < 1e-12


def test_jeffreys_posterior_rule_is_unrounded_and_width_scaling_behaves() -> None:
    assert c.posterior_parameters(127, 128) == (127.5, 1.5)
    assert c.posterior_mean(127, 128) == 127.5 / 129.0
    assert c.posterior_mean(0, 128) == 0.5 / 129.0
    a, b = c.posterior_parameters(96, 128, width_scale=4.0)
    assert (a, b) == (384.5, 128.5)
    with pytest.raises(ValueError):
        c.posterior_parameters(129, 128)
    with pytest.raises(ValueError):
        c.posterior_parameters(10, 128, width_scale=0.0)
    rows = c.unit_qmc_rows()
    design = c.load_catalogue()[1]
    narrow = [t[c.CUSP_NAMES[1]] for t in c.design_theta_rows(design, rows=rows, width_scale=4.0)]
    campaign = [t[c.CUSP_NAMES[1]] for t in c.design_theta_rows(design, rows=rows, width_scale=1.0)]
    wide = [t[c.CUSP_NAMES[1]] for t in c.design_theta_rows(design, rows=rows, width_scale=0.25)]
    point = [t[c.CUSP_NAMES[1]] for t in c.design_theta_rows(design, rows=rows, width_scale=None)]
    spread = lambda xs: max(xs) - min(xs)  # noqa: E731
    assert spread(narrow) < spread(campaign) < spread(wide)
    assert len(set(point)) == 1 and point[0] == c.posterior_mean(design.cell_wall_hits[1], 128)
    mean = design.cell_wall_hits[1] / 128
    assert abs(sum(campaign) / len(campaign) - mean) < 0.03


def test_frozen_unit_rows_and_catalogue_sample_hash_to_the_protocol(designs) -> None:
    value = protocol()
    rows = c.unit_qmc_rows()
    assert len(rows) == 64 and all(len(row) == 7 for row in rows)
    assert all(0.0 < coordinate < 1.0 for row in rows for coordinate in row)
    assert c.unit_rows_sha256(rows) == value["uncertain_inputs"]["sample"]["unit_rows_sha256"]
    sample = c.catalogue_sample(designs)
    assert len(sample) == 96 and all(len(rows_k) == 64 for rows_k in sample)
    assert c.catalogue_sample_sha256(sample) == value["uncertain_inputs"]["sample"]["catalogue_sample_sha256"]
    for design, rows_k in zip(designs, sample, strict=True):
        for theta in rows_k:
            assert list(theta) == list(c.CUSP_NAMES) + [c.POOLED_NAME] + [n for n, *_ in c.SHARED_UNCERTAIN_INPUTS]
            assert all(0.0 < theta[name] < 1.0 for name in c.CUSP_NAMES)
            assert 0.0 < theta[c.POOLED_NAME] < 1.0
            for name, lower, upper, _units in c.SHARED_UNCERTAIN_INPUTS:
                assert lower <= theta[name] <= upper
        nominal = c.design_nominal_theta(design)
        for k, name in enumerate(c.CUSP_NAMES):
            assert nominal[name] == (design.cell_wall_hits[k] + 0.5) / 129.0
        assert nominal["ionized_number_fraction"] == pytest.approx(0.815)
    # the unit rows are v1's (same construction, same offset): spot-check the radical inverse
    assert rows[0][0] == c._radical_inverse(17 + c.QMC_SEED * 104_729 + 1, 2)
    assert math.isclose(sum(rows[0]) / 7.0, 0.5, abs_tol=0.5)
