"""Recording evaluator, fairness, fronts, paired tests and optimiser smokes."""

from __future__ import annotations

import pytest

from experiments.mdo_l0_campaign_v1 import experiment, model as m, optimizers as opt


def _ledger(strategy: str, budget: int, seed: int = 900909) -> opt.RunLedger:
    return opt.RunLedger(
        strategy=strategy,
        seed=seed,
        budget=budget,
        sample=m.uncertain_sample(),
        nominal=m.nominal_theta(),
        tail_fraction=m.CVAR_TAIL_FRACTION,
    )


def test_ledger_enforces_budget_and_monotone_hypervolume() -> None:
    ledger = _ledger("lhs", 12)
    opt.run_lhs(ledger, initial_count=4)
    assert len(ledger.records) == 12
    curve = [item["hypervolume"] for item in ledger.hypervolume_curve]
    assert all(later >= earlier for earlier, later in zip(curve, curve[1:], strict=False))
    assert [item["evaluations"] for item in ledger.hypervolume_curve] == list(range(1, 13))
    summary = ledger.summary()
    assert summary["evaluations"] == 12 and summary["budget"] == 12
    assert summary["feasible_evaluations"] + summary["infeasible_evaluations"] == 12
    assert summary["pareto_set_size"] <= summary["feasible_evaluations"]
    with pytest.raises(opt.BudgetExceededError):
        ledger.evaluate((300.0, 1.0, 1e-6), batch=99, provenance="extra")
    assert [record["index"] for record in ledger.records] == list(range(12))
    assert all(record["provenance"].startswith("lhs:seed=900909") for record in ledger.records)


def test_shared_initial_points_are_identical_across_strategies_and_seed_specific() -> None:
    first = opt.shared_initial_points(900909, 8)
    again = opt.shared_initial_points(900909, 8)
    assert first.tolist() == again.tolist()
    assert opt.lhs_points(900909, 20, 8)[:8].tolist() == first.tolist()
    assert opt.shared_initial_points(900910, 8).tolist() != first.tolist()
    assert first.shape == (8, 3)
    assert ((first >= 0.0) & (first <= 1.0)).all()
    # Latin-hypercube property: exactly one point per stratum in every dimension.
    for column in first.T:
        assert sorted(int(c * 8) for c in column) == list(range(8))
    second_stage = opt.lhs_points(900909, 20, 8)[8:]
    for column in second_stage.T:
        assert sorted(int(c * 12) for c in column) == list(range(12))
    with pytest.raises(ValueError):
        opt.lhs_points(1, 4, 8)
    values = opt.denormalize(first[0])
    for value, variable in zip(values, m.DESIGN_VARIABLES, strict=True):
        assert variable.lower <= value <= variable.upper
    assert opt.denormalize((0.0, 0.0, 0.0)) == tuple(v.lower for v in m.DESIGN_VARIABLES)
    assert opt.denormalize((1.0, 1.0, 1.0)) == tuple(v.upper for v in m.DESIGN_VARIABLES)


def test_pooled_fronts_and_separability_on_ledger_records() -> None:
    ledger = _ledger("lhs", 32)
    opt.run_lhs(ledger, initial_count=8)
    records = [dict(record, run="lhs:900909") for record in ledger.records]
    pooled = experiment.pooled_fronts(records)
    assert pooled["unique_designs"] == 32
    assert pooled["robust"]["front_size"] >= 1
    assert pooled["nominal"]["front_size"] >= 1
    assert pooled["robust"]["hypervolume"] == pytest.approx(ledger.hypervolume_curve[-1]["hypervolume"])
    assert pooled["nominal"]["hypervolume"] >= pooled["robust"]["hypervolume"]
    assert set(pooled["robust"]["design_ids"]) == {
        item["design_id"] for item in pooled["robust"]["designs"]
    }
    assert 0.0 <= pooled["jaccard_robust_nominal"] <= 1.0
    separability = experiment.separability_check(records)
    assert separability["passed"] is True
    assert separability["ratios"]["anode_input_power_w"]["ratio_min"] == pytest.approx(1.0)
    replay = experiment.replay_records(records)
    assert replay["passed"] is True and replay["replayed"] == 32
    tampered = [dict(records[0])]
    tampered[0]["constraints"] = dict(tampered[0]["constraints"])
    tampered[0]["constraints"][m.ROBUST_CONSTRAINT.name] += 1e-3
    assert experiment.replay_records(tampered)["passed"] is False


def test_paired_test_and_seed_variance() -> None:
    summaries = {
        "qlognehvi:1": {"final_hypervolume": 0.5},
        "lhs:1": {"final_hypervolume": 0.4},
        "qlognehvi:2": {"final_hypervolume": 0.3},
        "lhs:2": {"final_hypervolume": 0.4},
        "qlognehvi:3": {"final_hypervolume": 0.6},
        "lhs:3": {"final_hypervolume": 0.5},
    }
    result = experiment._paired_test(summaries, (1, 2, 3), "qlognehvi", "lhs")
    assert result["wins"] == 2 and result["required_wins"] == 2 and result["passed"] is True
    summaries["qlognehvi:3"]["final_hypervolume"] = 0.1
    assert experiment._paired_test(summaries, (1, 2, 3), "qlognehvi", "lhs")["passed"] is False
    variance = experiment._seed_variance([0.5, 0.3, 0.6])
    assert variance["minimum"] == 0.3 and variance["maximum"] == 0.6
    assert variance["mean"] == pytest.approx(0.4666666666666667)
    assert experiment._seed_variance([0.5])["sample_std"] is None


def test_dense_reference_and_cusp_sensitivity_small() -> None:
    reference = experiment.dense_reference(64, 20260903)
    assert reference["count"] == 64
    assert reference["feasible"] + reference["infeasible"] == 64
    compact = experiment.compact_records(reference["records"])
    assert compact["count"] == 64 and len(compact["values"]) == 64
    value = experiment.protocol()
    records = [dict(record, run="dense") for record in reference["records"]]
    campaign_ids = reference["fronts"]["robust"]["design_ids"]
    sensitivity = experiment.cusp_sensitivity(records, value, campaign_ids)
    priors = {item["cusp_upper"]: item for item in sensitivity["priors"]}
    assert set(priors) == {0.0, 0.2, 0.45, 0.7}
    # The campaign prior reproduces the campaign front exactly.
    assert priors[0.45]["identical_to_campaign_front"] is True
    assert priors[0.45]["jaccard_with_campaign_front"] == 1.0
    # Separability: on the common feasible set every prior yields the same front
    # (up to floating-point ties).
    assert all(item["identical_on_common_feasible_set_up_to_ties"] for item in sensitivity["priors"])
    assert all(
        item["tolerant_common_front_symmetric_difference"] == 0 for item in sensitivity["priors"]
    )
    # Fewer designs are feasible when the prior admits less wall loss (p = 0 always).
    assert priors[0.0]["feasible"] <= priors[0.45]["feasible"] <= priors[0.7]["feasible"]
    scenarios = {item["id"]: item for item in sensitivity["scenarios"]}
    assert scenarios["no_wall_loss"]["survival"] == 1.0
    assert scenarios["v4_per_cell_jeffreys"]["survival"] < 1e-6
    assert scenarios["wide_prior_upper"]["survival"] == pytest.approx(0.55**4)


def test_nsga3_smoke_matches_budget_and_shared_initial_design() -> None:
    pytest.importorskip("pymoo")
    ledger = _ledger("nsga3", 16)
    info = opt.run_nsga3(
        ledger, initial_count=8, population_size=8, generations=2, reference_direction_seed=1
    )
    assert len(ledger.records) == 16
    assert info["pymoo_reported_evaluations"] == 16
    shared = opt.shared_initial_points(900909, 8)
    assert [record["design"]["values"] for record in ledger.records[:8]] == [
        list(opt.denormalize(row)) for row in shared
    ]
    with pytest.raises(ValueError):
        opt.run_nsga3(_ledger("nsga3", 16), initial_count=8, population_size=4, generations=4, reference_direction_seed=1)


def test_qlognehvi_smoke_on_cpu() -> None:
    pytest.importorskip("botorch")
    ledger = _ledger("qlognehvi", 10)
    info = opt.run_qlognehvi(
        ledger,
        initial_count=8,
        batch_size=2,
        device="cpu",
        num_restarts=2,
        raw_samples=32,
        mc_samples=16,
        fit_noise_floor=1e-6,
        sequential=False,
        maxiter=20,
    )
    assert len(ledger.records) == 10
    assert info["iterations"] == 1
    assert info["iteration_log"][0]["training_points"] == 8
    assert ledger.records[8]["provenance"].startswith("qlognehvi:seed=900909:iteration=1")
    layout = opt.model_layout()
    assert layout.output_names == (*m.OBJECTIVE_NAMES, m.ROBUST_CONSTRAINT.name)
    probe = opt.torch_environment("cpu")
    assert probe["float64_cholesky_probe"][0] == 2.0
