import json

import pytest

from cft_revival.optimization.campaign import (
    Campaign,
    CampaignConfig,
    CampaignError,
    Proposal,
)
from cft_revival.optimization.domain import (
    Design,
    EvaluationRequest,
    EvaluationStatus,
    Fidelity,
    ObjectiveDirection,
    ObjectiveSpec,
    ObjectiveValue,
    Observation,
    Provenance,
    SolverFailure,
    Variable,
    stable_hash,
)


VARIABLES = (Variable("x", 0.0, 1.0, "1"),)
PROVENANCE = Provenance("test", "rev", "inputs")
OBJECTIVES = (ObjectiveSpec("score", ObjectiveDirection.MAXIMIZE, "1"),)
CONFIG = CampaignConfig(
    fidelity_budgets=((Fidelity.F0, 3), (Fidelity.F3, 1)),
    maximum_pending=2,
    mandatory_high_fidelity_validations=1,
    minimum_high_fidelity_fraction=0.25,
    highest_fidelity_attempt_limit=2,
    reserved_high_fidelity_retries=1,
    maximum_retries_per_promotion=1,
    maximum_equivalent_f3_cost=2.1,
)


def proposal(x: float, fidelity: Fidelity = Fidelity.F0, seed: int = 0) -> Proposal:
    request = EvaluationRequest(
        Design((x,), VARIABLES),
        fidelity,
        seed,
        1,
        OBJECTIVES,
    )
    return Proposal(request, "test-acquisition", x)


def success(request: EvaluationRequest) -> Observation:
    return Observation(
        request,
        EvaluationStatus.SUCCESS,
        (ObjectiveValue("score", request.design.values[0], "1"),),
        (),
        PROVENANCE,
        1.0,
        1.0 if request.fidelity is Fidelity.F3 else 1 / 256,
    )


def test_ask_prevents_duplicate_pending_and_respects_capacity() -> None:
    campaign = Campaign(CONFIG)
    first = proposal(0.1)
    accepted = campaign.ask((first, first, proposal(0.2), proposal(0.3)), max_jobs=4)
    assert len(accepted) == 2
    assert len({job.proposal.request.evaluation_key for job in accepted}) == 2
    assert campaign.ask((proposal(0.4),), max_jobs=1) == ()


def test_campaign_config_copies_mutable_budget_input() -> None:
    budgets = [[Fidelity.F0, 2], [Fidelity.F3, 1]]
    config = CampaignConfig(
        fidelity_budgets=budgets,
        mandatory_high_fidelity_validations=1,
        highest_fidelity_attempt_limit=2,
        reserved_high_fidelity_retries=1,
        maximum_retries_per_promotion=1,
        maximum_equivalent_f3_cost=2.1,
    )
    campaign_id = Campaign(config).campaign_id
    budgets[0][1] = 999
    assert config.fidelity_budgets == ((Fidelity.F0, 2), (Fidelity.F3, 1))
    assert Campaign(config).campaign_id == campaign_id


def test_impossible_retry_and_cost_configurations_fail_up_front() -> None:
    with pytest.raises(ValueError, match="initial budget plus retries"):
        CampaignConfig(
            fidelity_budgets=((Fidelity.F3, 1),),
            mandatory_high_fidelity_validations=1,
            highest_fidelity_attempt_limit=1,
            reserved_high_fidelity_retries=1,
            maximum_retries_per_promotion=1,
        )
    underfunded = CampaignConfig(
        fidelity_budgets=((Fidelity.F0, 3), (Fidelity.F3, 1)),
        mandatory_high_fidelity_validations=1,
        highest_fidelity_attempt_limit=2,
        reserved_high_fidelity_retries=1,
        maximum_retries_per_promotion=1,
        maximum_equivalent_f3_cost=2.0,
    )
    with pytest.raises(ValueError, match="cost ceiling cannot fund"):
        Campaign(underfunded)


def test_tell_requires_pending_and_completed_key_cannot_repeat() -> None:
    campaign = Campaign(CONFIG)
    request = proposal(0.1).request
    with pytest.raises(CampaignError, match="pending"):
        campaign.tell(success(request))
    campaign.ask((proposal(0.1),), max_jobs=1)
    campaign.tell(success(request))
    assert campaign.ask((proposal(0.1),), max_jobs=1) == ()
    with pytest.raises(CampaignError, match="duplicate tell"):
        campaign.tell(success(request))


def test_failure_remains_explicit_and_replay_is_deterministic() -> None:
    campaign = Campaign(CONFIG)
    request = proposal(0.1).request
    campaign.ask((proposal(0.1),), max_jobs=1)
    campaign.tell(
        Observation(
            request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            3.0,
            1 / 256,
            SolverFailure("TIMEOUT", "solver timed out", True, "field"),
        )
    )
    payload = campaign.to_jsonl()
    replayed = Campaign.from_jsonl(payload)
    assert replayed.to_jsonl() == payload
    assert replayed.observations[0].status is EvaluationStatus.FAILURE


def test_promoted_candidate_is_forced_to_highest_fidelity_and_quota_gates_stop() -> None:
    campaign = Campaign(CONFIG)
    low_job = campaign.ask((proposal(0.25),), max_jobs=1)[0]
    low_observation = success(low_job.proposal.request)
    campaign.tell(low_observation)
    validation = campaign.validation_proposal(
        low_observation.request.design,
        seed=99,
        promotion_source_id=low_observation.observation_id,
        score=10.0,
    )
    assert validation.request.fidelity is Fidelity.F3
    with pytest.raises(CampaignError, match="source observation's design"):
        campaign.validation_proposal(
            Design((0.75,), VARIABLES),
            seed=100,
            promotion_source_id=low_observation.observation_id,
            score=9.0,
        )
    job = campaign.ask((proposal(0.8), validation), max_jobs=1)[0]
    assert job.proposal.mandatory_validation
    assert not campaign.stopping_diagnostics(
        acquisition_converged=True,
        verified_hypervolume_stalled=True,
        surrogate_calibrated=True,
        promoted_candidates_guardrails_passed=True,
        iteration_policy_satisfied=True,
    ).should_stop
    campaign.tell(success(job.proposal.request))
    diagnostics = campaign.stopping_diagnostics(
        acquisition_converged=True,
        verified_hypervolume_stalled=True,
        surrogate_calibrated=True,
        promoted_candidates_guardrails_passed=True,
        iteration_policy_satisfied=True,
    )
    assert diagnostics.should_stop
    assert (
        diagnostics.high_fidelity_successes
        == diagnostics.mandatory_validation_target
        == 1
    )
    assert diagnostics.mandatory_validation_gate_met
    assert diagnostics.high_fidelity_fraction_gate_met


def test_failed_high_fidelity_run_does_not_satisfy_validation_quota() -> None:
    campaign = Campaign(CONFIG)
    request = proposal(0.4, Fidelity.F3).request
    campaign.ask((proposal(0.4, Fidelity.F3),), max_jobs=1)
    campaign.tell(
        Observation(
            request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            8.0,
            1.0,
            SolverFailure("DIVERGED", "PIC solve diverged", False, "pic"),
        )
    )
    diagnostics = campaign.stopping_diagnostics(
        acquisition_converged=True,
        verified_hypervolume_stalled=True,
        surrogate_calibrated=True,
        promoted_candidates_guardrails_passed=True,
        iteration_policy_satisfied=True,
    )
    assert diagnostics.high_fidelity_attempts == 1
    assert diagnostics.high_fidelity_successes == 0
    assert diagnostics.high_fidelity_failures == 1
    assert diagnostics.retry_capacity_remaining == 1
    assert diagnostics.retryable_failure_lineages == 0
    assert diagnostics.validation_exhausted
    assert diagnostics.should_stop


def test_failed_f3_costs_real_budget_and_has_one_explicit_bounded_retry() -> None:
    campaign = Campaign(CONFIG)
    first = proposal(0.4, Fidelity.F3)
    campaign.ask((first,), max_jobs=1)
    campaign.tell(
        Observation(
            first.request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            8.0,
            1.0,
            SolverFailure("DIVERGED", "PIC failed", True, "pic"),
        )
    )
    retry = campaign.retry_proposal(
        first.request.evaluation_key,
        seed=91,
        score=2.0,
    )
    campaign.ask((retry,), max_jobs=1)
    with pytest.raises(CampaignError, match="already has a retry"):
        campaign.retry_proposal(
            first.request.evaluation_key,
            seed=92,
            score=1.0,
        )
    campaign.tell(success(retry.request))
    diagnostics = campaign.stopping_diagnostics(
        acquisition_converged=True,
        verified_hypervolume_stalled=True,
        surrogate_calibrated=True,
        promoted_candidates_guardrails_passed=True,
        iteration_policy_satisfied=True,
    )
    assert diagnostics.high_fidelity_attempts == 2
    assert diagnostics.high_fidelity_successes == 1
    assert diagnostics.high_fidelity_failures == 1
    assert diagnostics.retry_capacity_remaining == 0
    assert diagnostics.equivalent_f3_cost_spent == 2.0
    assert diagnostics.should_stop


def test_nonretryable_failure_and_zero_cost_f3_are_rejected() -> None:
    campaign = Campaign(CONFIG)
    first = proposal(0.41, Fidelity.F3)
    campaign.ask((first,), max_jobs=1)
    with pytest.raises(CampaignError, match="positive charged cost"):
        campaign.tell(
            Observation(
                first.request,
                EvaluationStatus.FAILURE,
                (),
                (),
                PROVENANCE,
                1.0,
                0.0,
                SolverFailure("REJECTED", "preflight failed", False, "preflight"),
            )
        )
    campaign.reject_pending(
        first.request.evaluation_key,
        reason="preflight rejection before execution",
    )
    assert campaign.cost_spent == 0.0
    assert campaign.stopping_diagnostics(
        acquisition_converged=False,
        verified_hypervolume_stalled=False,
        surrogate_calibrated=False,
        promoted_candidates_guardrails_passed=False,
        iteration_policy_satisfied=False,
    ).high_fidelity_attempts == 0
    assert len(campaign.rejections) == 1
    assert Campaign.from_jsonl(campaign.to_jsonl()).rejections == campaign.rejections

    campaign = Campaign(CONFIG)
    first = proposal(0.41, Fidelity.F3)
    campaign.ask((first,), max_jobs=1)
    campaign.tell(
        Observation(
            first.request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            1.0,
            1.0,
            SolverFailure("INVALID", "nonretryable solve", False, "pic"),
        )
    )
    with pytest.raises(CampaignError, match="non-retryable"):
        campaign.retry_proposal(
            first.request.evaluation_key,
            seed=90,
            score=1.0,
        )


def test_successful_retry_is_also_required_to_be_paid() -> None:
    campaign = Campaign(CONFIG)
    first = proposal(0.42, Fidelity.F3)
    campaign.ask((first,), max_jobs=1)
    campaign.tell(
        Observation(
            first.request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            1.0,
            1.0,
            SolverFailure("FAILED", "retryable", True, "pic"),
        )
    )
    retry = campaign.retry_proposal(
        first.request.evaluation_key,
        seed=96,
        score=1.0,
    )
    campaign.ask((retry,), max_jobs=1)
    with pytest.raises(CampaignError, match="positive charged cost"):
        campaign.tell(
            Observation(
                retry.request,
                EvaluationStatus.SUCCESS,
                (ObjectiveValue("score", 0.42, "1"),),
                (),
                PROVENANCE,
                1.0,
                0.0,
            )
        )


def test_retry_lineage_replays_through_live_policy() -> None:
    campaign = Campaign(CONFIG)
    first = proposal(0.45, Fidelity.F3)
    campaign.ask((first,), max_jobs=1)
    campaign.tell(
        Observation(
            first.request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            4.0,
            1.0,
            SolverFailure("FAILED", "first attempt failed", True, "pic"),
        )
    )
    retry = campaign.retry_proposal(
        first.request.evaluation_key,
        seed=94,
        score=2.0,
    )
    campaign.ask((retry,), max_jobs=1)
    payload = campaign.to_jsonl()
    replayed = Campaign.from_jsonl(payload)
    assert replayed.to_jsonl() == payload
    with pytest.raises(CampaignError, match="already has a retry"):
        replayed.retry_proposal(
            first.request.evaluation_key,
            seed=95,
            score=1.0,
        )


def test_failed_retry_exhaustion_is_visible_and_terminal() -> None:
    campaign = Campaign(CONFIG)
    first = proposal(0.4, Fidelity.F3)
    campaign.ask((first,), max_jobs=1)
    for request in (first.request,):
        campaign.tell(
            Observation(
                request,
                EvaluationStatus.FAILURE,
                (),
                (),
                PROVENANCE,
                8.0,
                1.0,
                SolverFailure("DIVERGED", "PIC failed", True, "pic"),
            )
        )
    retry = campaign.retry_proposal(
        first.request.evaluation_key,
        seed=93,
        score=2.0,
    )
    campaign.ask((retry,), max_jobs=1)
    campaign.tell(
        Observation(
            retry.request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            8.0,
            1.0,
            SolverFailure("DIVERGED", "PIC retry failed", False, "pic"),
        )
    )
    diagnostics = campaign.stopping_diagnostics(
        acquisition_converged=False,
        verified_hypervolume_stalled=False,
        surrogate_calibrated=False,
        promoted_candidates_guardrails_passed=False,
        iteration_policy_satisfied=False,
    )
    assert diagnostics.validation_exhausted
    assert diagnostics.retry_capacity_remaining == 0
    assert diagnostics.should_stop


def test_fabricated_and_redundant_promotions_are_rejected() -> None:
    campaign = Campaign(CONFIG)
    fake = Proposal(
        EvaluationRequest(
            Design((0.2,), VARIABLES),
            Fidelity.F3,
            10,
            1,
            OBJECTIVES,
        ),
        "mandatory-pareto-validation",
        1.0,
        promotion_source_id="fabricated",
        promotion_lineage_id="fabricated-lineage",
        mandatory_validation=True,
    )
    with pytest.raises(CampaignError, match="promotion source"):
        campaign.ask((fake,), max_jobs=1)

    high = proposal(0.4, Fidelity.F3)
    campaign.ask((high,), max_jobs=1)
    observed = success(high.request)
    campaign.tell(observed)
    with pytest.raises(CampaignError, match="already at highest"):
        campaign.validation_proposal(
            observed.request.design,
            seed=11,
            promotion_source_id=observed.observation_id,
            score=1.0,
        )


def test_recorded_but_dominated_source_is_not_promotion_eligible() -> None:
    campaign = Campaign(CONFIG)
    jobs = campaign.ask((proposal(0.1), proposal(0.9)), max_jobs=2)
    for job in jobs:
        campaign.tell(success(job.proposal.request))
    dominated = next(
        item for item in campaign.observations if item.request.design.values == (0.1,)
    )
    with pytest.raises(CampaignError, match="eligibility"):
        campaign.validation_proposal(
            dominated.request.design,
            seed=20,
            promotion_source_id=dominated.observation_id,
            score=1.0,
        )


def test_duplicate_mandatory_jobs_for_one_source_are_rejected_across_seeds() -> None:
    campaign = Campaign(CONFIG)
    low = campaign.ask((proposal(0.5),), max_jobs=1)[0]
    observed = success(low.proposal.request)
    campaign.tell(observed)
    first = campaign.validation_proposal(
        observed.request.design,
        seed=30,
        promotion_source_id=observed.observation_id,
        score=2.0,
    )
    second = campaign.validation_proposal(
        observed.request.design,
        seed=31,
        promotion_source_id=observed.observation_id,
        score=1.0,
    )
    with pytest.raises(CampaignError, match="duplicate promotion-lineage"):
        campaign.ask((first, second), max_jobs=2)
    assert campaign.pending == ()
    campaign.ask((first,), max_jobs=1)
    with pytest.raises(CampaignError, match="already pending"):
        campaign.ask((second,), max_jobs=1)


def test_same_design_different_source_seeds_share_one_promotion_lineage() -> None:
    campaign = Campaign(CONFIG)
    jobs = campaign.ask(
        (proposal(0.55, seed=1), proposal(0.55, seed=2)),
        max_jobs=2,
    )
    observations = []
    for job in jobs:
        observation = success(job.proposal.request)
        campaign.tell(observation)
        observations.append(observation)
    first = campaign.validation_proposal(
        observations[0].request.design,
        seed=50,
        promotion_source_id=observations[0].observation_id,
        score=2.0,
    )
    second = campaign.validation_proposal(
        observations[1].request.design,
        seed=51,
        promotion_source_id=observations[1].observation_id,
        score=1.0,
    )
    assert first.promotion_source_id != second.promotion_source_id
    assert first.promotion_lineage_id == second.promotion_lineage_id
    with pytest.raises(CampaignError, match="duplicate promotion-lineage"):
        campaign.ask((first, second), max_jobs=2)


def test_cross_context_observation_cannot_block_promotion() -> None:
    campaign = Campaign(CONFIG)
    request_a = EvaluationRequest(
        Design((0.2,), VARIABLES),
        Fidelity.F0,
        1,
        1,
        OBJECTIVES,
        result_context=(("configuration", "A"),),
    )
    request_b = EvaluationRequest(
        Design((0.9,), VARIABLES),
        Fidelity.F0,
        2,
        1,
        OBJECTIVES,
        result_context=(("configuration", "B"),),
    )
    jobs = campaign.ask(
        (
            Proposal(request_a, "context-test", 1.0),
            Proposal(request_b, "context-test", 2.0),
        ),
        max_jobs=2,
    )
    requests = {job.proposal.request.evaluation_key: job.proposal.request for job in jobs}
    observation_a = Observation(
        requests[request_a.evaluation_key],
        EvaluationStatus.SUCCESS,
        (ObjectiveValue("score", 0.2, "1"),),
        (),
        Provenance("model-A", "revision-A", "input-A"),
        1.0,
        1 / 256,
    )
    observation_b = Observation(
        requests[request_b.evaluation_key],
        EvaluationStatus.SUCCESS,
        (ObjectiveValue("score", 0.9, "1"),),
        (),
        Provenance("model-B", "revision-B", "input-B"),
        1.0,
        1 / 256,
    )
    campaign.tell(observation_a)
    campaign.tell(observation_b)
    promotion_a = campaign.validation_proposal(
        observation_a.request.design,
        seed=60,
        promotion_source_id=observation_a.observation_id,
        score=1.0,
    )
    promotion_b = campaign.validation_proposal(
        observation_b.request.design,
        seed=61,
        promotion_source_id=observation_b.observation_id,
        score=1.0,
    )
    assert promotion_a.promotion_lineage_id != promotion_b.promotion_lineage_id


def test_mandatory_retry_preserves_source_lineage_and_requires_failure_key() -> None:
    campaign = Campaign(CONFIG)
    low = campaign.ask((proposal(0.6),), max_jobs=1)[0]
    source = success(low.proposal.request)
    campaign.tell(source)
    validation = campaign.validation_proposal(
        source.request.design,
        seed=40,
        promotion_source_id=source.observation_id,
        score=2.0,
    )
    campaign.ask((validation,), max_jobs=1)
    campaign.tell(
        Observation(
            validation.request,
            EvaluationStatus.FAILURE,
            (),
            (),
            PROVENANCE,
            5.0,
            1.0,
            SolverFailure("FAILED", "validation failed", True, "pic"),
        )
    )
    with pytest.raises(CampaignError, match="terminal failed validation"):
        campaign.validation_proposal(
            source.request.design,
            seed=41,
            promotion_source_id=source.observation_id,
            score=1.0,
        )
    retry = campaign.retry_proposal(
        validation.request.evaluation_key,
        seed=41,
        score=1.0,
    )
    assert retry.mandatory_validation
    assert retry.promotion_source_id == source.observation_id
    assert retry.retry_of_evaluation_key == validation.request.evaluation_key


def test_replay_rejects_malformed_tampered_and_policy_bypassing_logs() -> None:
    campaign = Campaign(CONFIG)
    campaign.ask((proposal(0.1),), max_jobs=1)
    payload = campaign.to_jsonl()
    with pytest.raises(CampaignError, match="malformed JSON"):
        Campaign.from_jsonl(payload + "\n{")
    with pytest.raises(CampaignError, match="payload must be a string"):
        Campaign.from_jsonl(None)  # type: ignore[arg-type]
    with pytest.raises(CampaignError, match="duplicate object key 'sequence'"):
        Campaign.from_jsonl('{"sequence":-1,"sequence":-1}')
    with pytest.raises(CampaignError, match="duplicate object key 'maximum_pending'"):
        Campaign.from_jsonl(
            payload.replace(
                '"maximum_pending":2',
                '"maximum_pending":2,"maximum_pending":2',
            )
        )
    with pytest.raises(CampaignError, match="duplicate object key 'score'"):
        Campaign.from_jsonl(
            payload.replace(
                '"score":0.1',
                '"score":0.1,"score":0.1',
            )
        )
    for constant in ("NaN", "Infinity", "-Infinity"):
        with pytest.raises(CampaignError, match="malformed JSON"):
            Campaign.from_jsonl(
                payload.replace(
                    '"maximum_pending":2',
                    f'"maximum_pending":{constant}',
                )
            )
    for overflow_payload in (
        '{"nested":{"values":[1e999]}}',
        '[{"outer":[{"inner":-1e999}]}]',
        payload.replace('"score":0.1', '"score":1e999'),
    ):
        with pytest.raises(CampaignError, match="non-finite number"):
            Campaign.from_jsonl(overflow_payload)

    records = [json.loads(line) for line in payload.splitlines()]
    records[0]["config"]["maximum_pending"] = 99
    tampered_header = "\n".join(json.dumps(item) for item in records)
    with pytest.raises(CampaignError, match="hash"):
        Campaign.from_jsonl(tampered_header)

    records = [json.loads(line) for line in payload.splitlines()]
    records[1]["proposal"]["score"] = 999.0
    tampered_event = "\n".join(json.dumps(item) for item in records)
    with pytest.raises(CampaignError, match="event hash"):
        Campaign.from_jsonl(tampered_event)

    records = [json.loads(line) for line in payload.splitlines()]
    duplicate = dict(records[1])
    duplicate["sequence"] = 1
    duplicate["previous_event_hash"] = records[1]["event_hash"]
    duplicate["event_hash"] = stable_hash(
        {key: value for key, value in duplicate.items() if key != "event_hash"}
    )
    policy_bypass = "\n".join(json.dumps(item) for item in (*records, duplicate))
    with pytest.raises(CampaignError, match="safety policy"):
        Campaign.from_jsonl(policy_bypass)

    replayed_pending = Campaign.from_jsonl(payload)
    assert len(replayed_pending.pending) == 1

    malformed_observation = Campaign(CONFIG)
    job = malformed_observation.ask((proposal(0.2),), max_jobs=1)[0]
    malformed_observation.tell(success(job.proposal.request))
    records = [
        json.loads(line)
        for line in malformed_observation.to_jsonl().splitlines()
    ]
    records[2]["observation"] = []
    records[2]["event_hash"] = stable_hash(
        {
            key: value
            for key, value in records[2].items()
            if key != "event_hash"
        }
    )
    with pytest.raises(CampaignError, match="event values are invalid"):
        Campaign.from_jsonl("\n".join(json.dumps(item) for item in records))


def test_all_stopping_evidence_gates_are_enforced() -> None:
    campaign = Campaign(CONFIG)
    diagnostics = campaign.stopping_diagnostics(
        acquisition_converged=True,
        verified_hypervolume_stalled=True,
        surrogate_calibrated=False,
        promoted_candidates_guardrails_passed=False,
        iteration_policy_satisfied=False,
    )
    assert not diagnostics.should_stop
    assert "surrogate calibration gate is unmet" in diagnostics.reasons
    assert "promoted-candidate guardrails gate is unmet" in diagnostics.reasons
    assert "iteration acquisition policy gate is unmet" in diagnostics.reasons


def test_mandatory_count_and_success_fraction_are_independent_gates() -> None:
    high_fraction_config = CampaignConfig(
        fidelity_budgets=((Fidelity.F0, 3), (Fidelity.F3, 2)),
        mandatory_high_fidelity_validations=1,
        minimum_high_fidelity_fraction=0.75,
        highest_fidelity_attempt_limit=3,
        reserved_high_fidelity_retries=1,
        maximum_retries_per_promotion=1,
        maximum_equivalent_f3_cost=3.1,
    )
    campaign = Campaign(high_fraction_config)
    jobs = campaign.ask(
        (proposal(0.2, Fidelity.F3), proposal(0.3, Fidelity.F0)),
        max_jobs=2,
    )
    for job in jobs:
        campaign.tell(success(job.proposal.request))
    diagnostics = campaign.stopping_diagnostics(
        acquisition_converged=True,
        verified_hypervolume_stalled=True,
        surrogate_calibrated=True,
        promoted_candidates_guardrails_passed=True,
        iteration_policy_satisfied=True,
    )
    assert diagnostics.mandatory_validation_gate_met
    assert diagnostics.high_fidelity_success_fraction == 0.5
    assert not diagnostics.high_fidelity_fraction_gate_met
    assert not diagnostics.should_stop

    count_config = CampaignConfig(
        fidelity_budgets=((Fidelity.F3, 2),),
        mandatory_high_fidelity_validations=2,
        minimum_high_fidelity_fraction=0.5,
        highest_fidelity_attempt_limit=3,
        reserved_high_fidelity_retries=1,
        maximum_retries_per_promotion=1,
        maximum_equivalent_f3_cost=3.0,
    )
    campaign = Campaign(count_config)
    high = proposal(0.8, Fidelity.F3)
    campaign.ask((high,), max_jobs=1)
    campaign.tell(success(high.request))
    diagnostics = campaign.stopping_diagnostics(
        acquisition_converged=True,
        verified_hypervolume_stalled=True,
        surrogate_calibrated=True,
        promoted_candidates_guardrails_passed=True,
        iteration_policy_satisfied=True,
    )
    assert diagnostics.high_fidelity_fraction_gate_met
    assert not diagnostics.mandatory_validation_gate_met
    assert not diagnostics.should_stop
