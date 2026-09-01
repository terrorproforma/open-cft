"""Asynchronous, event-sourced ask/tell campaign orchestration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from math import isfinite
from numbers import Real
from typing import Any, Iterable, Mapping, Sequence

from .domain import (
    DEFAULT_SOURCES,
    Design,
    DomainError,
    EvaluationRequest,
    EvaluationStatus,
    Fidelity,
    InformationSource,
    Observation,
    request_from_dict,
    request_to_dict,
    source_for,
    stable_hash,
)
from .pareto import promotion_metadata


class CampaignError(RuntimeError):
    """A campaign transition would violate an invariant."""


def _validate_decoded_json(value: Any, path: str = "$") -> None:
    """Reject non-finite or non-JSON decoded values before any hashing."""
    if value is None or isinstance(value, (bool, str, int)):
        return
    if isinstance(value, float):
        if not isfinite(value):
            raise CampaignError(f"campaign JSON contains non-finite number at {path}")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_decoded_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise CampaignError(f"campaign JSON object key is not text at {path}")
            _validate_decoded_json(item, f"{path}.{key}")
        return
    raise CampaignError(f"campaign JSON contains unsupported value at {path}")


@dataclass(frozen=True)
class CampaignConfig:
    campaign_spec_id: str = "cft-8d-4objective-multifidelity@1.4"
    fidelity_budgets: tuple[tuple[Fidelity, int], ...] = (
        (Fidelity.F0, 256),
        (Fidelity.F1, 96),
        (Fidelity.F2, 32),
        (Fidelity.F3, 12),
    )
    maximum_pending: int = 32
    mandatory_high_fidelity_validations: int = 12
    minimum_high_fidelity_fraction: float = 0.03
    highest_fidelity_attempt_limit: int = 16
    reserved_high_fidelity_retries: int = 4
    maximum_retries_per_promotion: int = 1
    maximum_equivalent_f3_cost: float = 19.0
    highest_available_fidelity: Fidelity = Fidelity.F3

    def __post_init__(self) -> None:
        object.__setattr__(self, "campaign_spec_id", str(self.campaign_spec_id))
        if not self.campaign_spec_id:
            raise ValueError("campaign specification identity is required")
        try:
            raw_budgets = tuple(self.fidelity_budgets)
            if any(
                isinstance(budget, bool) or not isinstance(budget, int)
                for _, budget in raw_budgets
            ):
                raise ValueError("fidelity budgets must be integers")
            fidelity_budgets = tuple(
                (Fidelity(fidelity), budget)
                for fidelity, budget in raw_budgets
            )
            highest = Fidelity(self.highest_available_fidelity)
        except (TypeError, ValueError) as exc:
            raise ValueError("campaign fidelity configuration is invalid") from exc
        object.__setattr__(self, "fidelity_budgets", fidelity_budgets)
        object.__setattr__(self, "highest_available_fidelity", highest)
        budgets = dict(fidelity_budgets)
        if len(budgets) != len(fidelity_budgets) or any(
            value < 0 for value in budgets.values()
        ):
            raise ValueError("fidelity budgets must be unique and non-negative")
        if (
            isinstance(self.maximum_pending, bool)
            or not isinstance(self.maximum_pending, int)
            or isinstance(self.mandatory_high_fidelity_validations, bool)
            or not isinstance(self.mandatory_high_fidelity_validations, int)
            or isinstance(self.highest_fidelity_attempt_limit, bool)
            or not isinstance(self.highest_fidelity_attempt_limit, int)
            or isinstance(self.reserved_high_fidelity_retries, bool)
            or not isinstance(self.reserved_high_fidelity_retries, int)
            or isinstance(self.maximum_retries_per_promotion, bool)
            or not isinstance(self.maximum_retries_per_promotion, int)
            or self.maximum_pending < 1
            or self.mandatory_high_fidelity_validations < 0
            or self.highest_fidelity_attempt_limit < 0
            or self.reserved_high_fidelity_retries < 0
            or self.maximum_retries_per_promotion < 0
        ):
            raise ValueError("invalid pending or validation limit")
        if (
            isinstance(self.minimum_high_fidelity_fraction, bool)
            or not isinstance(self.minimum_high_fidelity_fraction, Real)
            or not isfinite(float(self.minimum_high_fidelity_fraction))
            or not 0.0 <= self.minimum_high_fidelity_fraction <= 1.0
        ):
            raise ValueError("minimum high-fidelity fraction must lie in [0, 1]")
        if (
            isinstance(self.maximum_equivalent_f3_cost, bool)
            or not isinstance(self.maximum_equivalent_f3_cost, Real)
            or not isfinite(float(self.maximum_equivalent_f3_cost))
            or self.maximum_equivalent_f3_cost <= 0.0
        ):
            raise ValueError("maximum cost must be positive")
        object.__setattr__(
            self,
            "minimum_high_fidelity_fraction",
            float(self.minimum_high_fidelity_fraction),
        )
        object.__setattr__(
            self,
            "maximum_equivalent_f3_cost",
            float(self.maximum_equivalent_f3_cost),
        )
        if self.highest_available_fidelity not in budgets:
            raise ValueError("highest available fidelity requires a configured budget")
        if (
            self.mandatory_high_fidelity_validations
            > budgets[self.highest_available_fidelity]
        ):
            raise ValueError("validation quota cannot exceed highest-fidelity budget")
        required_attempt_capacity = (
            budgets[self.highest_available_fidelity]
            + self.reserved_high_fidelity_retries
        )
        if self.highest_fidelity_attempt_limit != required_attempt_capacity:
            raise ValueError(
                "highest-fidelity attempt limit must equal initial budget plus retries"
            )
        if (
            self.reserved_high_fidelity_retries > 0
            and self.maximum_retries_per_promotion < 1
        ):
            raise ValueError("reserved retries require per-promotion retry capacity")

    @property
    def budget_by_fidelity(self) -> dict[Fidelity, int]:
        return dict(self.fidelity_budgets)


@dataclass(frozen=True)
class Proposal:
    request: EvaluationRequest
    acquisition: str
    score: float
    promotion_source_id: str | None = None
    promotion_lineage_id: str | None = None
    mandatory_validation: bool = False
    retry_of_evaluation_key: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.request, EvaluationRequest):
            raise ValueError("proposal request must be an EvaluationRequest")
        object.__setattr__(self, "acquisition", str(self.acquisition))
        if self.promotion_source_id is not None:
            object.__setattr__(
                self, "promotion_source_id", str(self.promotion_source_id)
            )
        if self.promotion_lineage_id is not None:
            object.__setattr__(
                self, "promotion_lineage_id", str(self.promotion_lineage_id)
            )
        if self.retry_of_evaluation_key is not None:
            object.__setattr__(
                self,
                "retry_of_evaluation_key",
                str(self.retry_of_evaluation_key),
            )
        if not isinstance(self.mandatory_validation, bool):
            raise ValueError("mandatory validation flag must be boolean")
        if (
            not self.acquisition
            or isinstance(self.score, bool)
            or not isinstance(self.score, Real)
            or not isfinite(float(self.score))
        ):
            raise ValueError("proposals require an acquisition name and finite score")
        object.__setattr__(self, "score", float(self.score))
        if self.mandatory_validation and (
            self.promotion_source_id is None
            or self.promotion_lineage_id is None
        ):
            raise ValueError(
                "mandatory validation requires source and promotion lineage"
            )


@dataclass(frozen=True)
class PendingJob:
    proposal: Proposal
    ask_index: int

    def __post_init__(self) -> None:
        if not isinstance(self.proposal, Proposal):
            raise ValueError("pending job requires a Proposal")
        if (
            isinstance(self.ask_index, bool)
            or not isinstance(self.ask_index, int)
            or self.ask_index < 0
        ):
            raise ValueError("pending job ask index must be a non-negative integer")


@dataclass(frozen=True)
class PreExecutionRejection:
    evaluation_key: str
    reason: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "evaluation_key", str(self.evaluation_key))
        object.__setattr__(self, "reason", str(self.reason))
        if not self.evaluation_key or not self.reason:
            raise ValueError("pre-execution rejection requires key and reason")


@dataclass(frozen=True)
class StoppingDiagnostics:
    should_stop: bool
    reasons: tuple[str, ...]
    completed: int
    failed: int
    pending: int
    equivalent_f3_cost_spent: float
    equivalent_f3_cost_committed: float
    high_fidelity_attempts: int
    high_fidelity_successes: int
    high_fidelity_failures: int
    high_fidelity_attempt_limit: int
    retry_capacity_remaining: int
    retryable_failure_lineages: int
    mandatory_validation_target: int
    mandatory_validation_gate_met: bool
    high_fidelity_success_fraction: float
    minimum_high_fidelity_fraction: float
    high_fidelity_fraction_gate_met: bool
    validation_exhausted: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "reasons", tuple(self.reasons))


def _proposal_to_dict(proposal: Proposal) -> dict[str, Any]:
    return {
        "request": request_to_dict(proposal.request),
        "acquisition": proposal.acquisition,
        "score": proposal.score,
        "promotion_source_id": proposal.promotion_source_id,
        "promotion_lineage_id": proposal.promotion_lineage_id,
        "mandatory_validation": proposal.mandatory_validation,
        "retry_of_evaluation_key": proposal.retry_of_evaluation_key,
    }


def _proposal_from_dict(raw: Mapping[str, Any]) -> Proposal:
    return Proposal(
        request=request_from_dict(raw["request"]),
        acquisition=str(raw["acquisition"]),
        score=float(raw["score"]),
        promotion_source_id=(
            str(raw["promotion_source_id"]) if raw.get("promotion_source_id") is not None else None
        ),
        promotion_lineage_id=(
            str(raw["promotion_lineage_id"])
            if raw.get("promotion_lineage_id") is not None
            else None
        ),
        mandatory_validation=bool(raw.get("mandatory_validation", False)),
        retry_of_evaluation_key=(
            str(raw["retry_of_evaluation_key"])
            if raw.get("retry_of_evaluation_key") is not None
            else None
        ),
    )


class Campaign:
    """Mutable coordinator over immutable requests, observations, and events."""

    def __init__(
        self,
        config: CampaignConfig = CampaignConfig(),
        sources: Sequence[InformationSource] = DEFAULT_SOURCES,
    ) -> None:
        if not isinstance(config, CampaignConfig):
            raise ValueError("campaign config must be a CampaignConfig")
        self.config = config
        self.sources = tuple(sources)
        if any(not isinstance(source, InformationSource) for source in self.sources):
            raise ValueError("campaign sources must be InformationSource records")
        source_fidelities = [source.fidelity for source in self.sources]
        if len(source_fidelities) != len(set(source_fidelities)):
            raise ValueError("information source fidelities must be unique")
        missing = set(config.budget_by_fidelity) - set(source_fidelities)
        if missing:
            raise ValueError(f"budgets have no information sources: {sorted(missing)}")
        minimum_campaign_cost = sum(
            (
                config.highest_fidelity_attempt_limit
                if fidelity is config.highest_available_fidelity
                else budget
            )
            * source_for(fidelity, self.sources).equivalent_cost
            for fidelity, budget in config.fidelity_budgets
        )
        if config.maximum_equivalent_f3_cost + 1e-12 < minimum_campaign_cost:
            raise ValueError(
                "cost ceiling cannot fund configured budgets and retry attempts"
            )
        self._pending: dict[str, PendingJob] = {}
        self._observations: dict[str, Observation] = {}
        self._promotion_lineage: dict[str, str] = {}
        self._promotion_lineage_ids: dict[str, str] = {}
        self._retry_keys: set[str] = set()
        self._retry_parent: dict[str, str] = {}
        self._rejections: list[PreExecutionRejection] = []
        self._events: list[dict[str, Any]] = []
        self._ask_counter = 0

    @property
    def pending(self) -> tuple[PendingJob, ...]:
        return tuple(sorted(self._pending.values(), key=lambda item: item.ask_index))

    @property
    def observations(self) -> tuple[Observation, ...]:
        return tuple(self._observations.values())

    @property
    def rejections(self) -> tuple[PreExecutionRejection, ...]:
        return tuple(self._rejections)

    @property
    def campaign_id(self) -> str:
        return stable_hash(
            {
                "config": {
                    **asdict(self.config),
                    "fidelity_budgets": [
                        [fidelity.value, budget]
                        for fidelity, budget in self.config.fidelity_budgets
                    ],
                    "highest_available_fidelity": self.config.highest_available_fidelity.value,
                },
                "sources": [
                    {
                        **asdict(source),
                        "fidelity": source.fidelity.value,
                    }
                    for source in self.sources
                ],
            }
        )

    def _record_event(self, payload: Mapping[str, Any]) -> None:
        event = {
            "sequence": len(self._events),
            **payload,
            "previous_event_hash": (
                self._events[-1]["event_hash"]
                if self._events
                else self.campaign_id
            ),
        }
        event["event_hash"] = stable_hash(event)
        self._events.append(event)

    def _used_count(self, fidelity: Fidelity) -> int:
        return sum(
            item.request.fidelity is fidelity for item in self._observations.values()
        ) + sum(
            item.proposal.request.fidelity is fidelity for item in self._pending.values()
        )

    def _attempt_limit(self, fidelity: Fidelity) -> int:
        if fidelity is self.config.highest_available_fidelity:
            return self.config.highest_fidelity_attempt_limit
        return self.config.budget_by_fidelity.get(fidelity, 0)

    def _expected_cost(self, fidelity: Fidelity) -> float:
        return source_for(fidelity, self.sources).equivalent_cost

    @property
    def cost_spent(self) -> float:
        return sum(item.charged_cost for item in self._observations.values())

    @property
    def cost_committed(self) -> float:
        return self.cost_spent + sum(
            self._expected_cost(item.proposal.request.fidelity) for item in self._pending.values()
        )

    def _high_fidelity_success_count(self) -> int:
        highest = self.config.highest_available_fidelity
        return sum(
            item.request.fidelity is highest
            and item.status is EvaluationStatus.SUCCESS
            for item in self._observations.values()
        )

    def _high_fidelity_attempt_count(self) -> int:
        highest = self.config.highest_available_fidelity
        return sum(
            item.request.fidelity is highest
            for item in self._observations.values()
        ) + sum(
            item.proposal.request.fidelity is highest
            for item in self._pending.values()
        )

    def _initial_high_fidelity_attempt_count(self) -> int:
        highest = self.config.highest_available_fidelity
        return sum(
            item.request.fidelity is highest
            and key not in self._retry_keys
            for key, item in self._observations.items()
        ) + sum(
            item.proposal.request.fidelity is highest
            and key not in self._retry_keys
            for key, item in self._pending.items()
        )

    def _promotion_attempt_keys(self, lineage_id: str) -> tuple[str, ...]:
        return tuple(
            key
            for key, recorded_lineage_id in self._promotion_lineage_ids.items()
            if recorded_lineage_id == lineage_id
        )

    def _comparable_context_id(self, observation: Observation) -> str:
        information_source = source_for(
            observation.request.fidelity,
            self.sources,
        )
        return stable_hash(
            {
                "campaign_spec_id": self.config.campaign_spec_id,
                "fidelity": observation.request.fidelity.value,
                "information_source": {
                    **asdict(information_source),
                    "fidelity": information_source.fidelity.value,
                },
                "objective_specs": [
                    {
                        **asdict(specification),
                        "direction": specification.direction.value,
                    }
                    for specification in observation.request.objective_specs
                ],
                "constraint_specs": [
                    {
                        **asdict(specification),
                        "sense": specification.sense.value,
                    }
                    for specification in observation.request.constraint_specs
                ],
                "result_context": observation.request.result_context,
                "model_version": observation.provenance.model_version,
                "code_revision": observation.provenance.code_revision,
            }
        )

    def _promotion_lineage_id(self, source: Observation) -> str:
        return stable_hash(
            {
                "design_id": source.request.design.design_id,
                "target_fidelity": self.config.highest_available_fidelity.value,
                "comparable_context_id": self._comparable_context_id(source),
            }
        )

    def _retryable_failure_lineage_count(self) -> int:
        roots: set[str] = set()
        for key, observation in self._observations.items():
            if (
                observation.request.fidelity
                is not self.config.highest_available_fidelity
                or observation.status is not EvaluationStatus.FAILURE
                or observation.failure is None
                or not observation.failure.retryable
            ):
                continue
            root = self._retry_root(key)
            lineage_keys = {
                candidate
                for candidate in (
                    *self._observations.keys(),
                    *self._pending.keys(),
                )
                if self._retry_root(candidate) == root
            }
            if any(
                candidate in self._observations
                and self._observations[candidate].status
                is EvaluationStatus.SUCCESS
                for candidate in lineage_keys
            ):
                continue
            if any(candidate in self._pending for candidate in lineage_keys):
                continue
            retries_used = sum(
                self._retry_root(candidate) == root
                for candidate in self._retry_keys
            )
            if retries_used < self.config.maximum_retries_per_promotion:
                roots.add(root)
        return len(roots)

    def _retry_root(self, key: str) -> str:
        while key in self._retry_parent:
            key = self._retry_parent[key]
        return key

    def _validate_retry_proposal(self, proposal: Proposal) -> None:
        parent_key = proposal.retry_of_evaluation_key
        if parent_key is None:
            return
        if parent_key not in self._observations:
            raise CampaignError("retry must reference a completed evaluation")
        parent = self._observations[parent_key]
        if parent.status is not EvaluationStatus.FAILURE:
            raise CampaignError("retry must reference a terminal failed evaluation")
        assert parent.failure is not None
        if not parent.failure.retryable:
            raise CampaignError("solver failure is explicitly non-retryable")
        request = proposal.request
        if parent.request.fidelity is not self.config.highest_available_fidelity:
            raise CampaignError("reserved retry capacity is highest-fidelity only")
        if (
            request.fidelity is not parent.request.fidelity
            or request.design.design_id != parent.request.design.design_id
            or request.requested_replicates != parent.request.requested_replicates
            or request.objective_specs != parent.request.objective_specs
            or request.constraint_specs != parent.request.constraint_specs
            or request.result_context != parent.request.result_context
        ):
            raise CampaignError("retry cannot change result-defining request inputs")
        if any(
            existing_parent == parent_key
            for existing_parent in self._retry_parent.values()
        ):
            raise CampaignError("failed evaluation already has a retry attempt")
        root = self._retry_root(parent_key)
        retries_for_root = sum(
            self._retry_root(key) == root for key in self._retry_keys
        )
        if retries_for_root >= self.config.maximum_retries_per_promotion:
            raise CampaignError("retry limit for evaluation lineage is exhausted")
        root_pending = any(
            self._retry_root(key) == root for key in self._pending
        )
        if root_pending:
            raise CampaignError("evaluation lineage already has a pending retry")
        parent_source = self._promotion_lineage.get(parent_key)
        parent_lineage = self._promotion_lineage_ids.get(parent_key)
        if parent_source is None:
            if (
                proposal.mandatory_validation
                or proposal.promotion_source_id is not None
                or proposal.promotion_lineage_id is not None
            ):
                raise CampaignError("ordinary F3 retry cannot fabricate promotion lineage")
        elif (
            not proposal.mandatory_validation
            or proposal.promotion_source_id != parent_source
            or proposal.promotion_lineage_id != parent_lineage
        ):
            raise CampaignError("mandatory retry must preserve promotion-source lineage")

    def _validate_mandatory_proposal(self, proposal: Proposal) -> None:
        if not proposal.mandatory_validation:
            return
        source_id = proposal.promotion_source_id
        sources = {
            observation.observation_id: observation
            for observation in self._observations.values()
        }
        if source_id not in sources:
            raise CampaignError("promotion source must be an observed immutable result")
        source = sources[source_id]
        if source.status is not EvaluationStatus.SUCCESS:
            raise CampaignError("failed observations are not eligible for promotion")
        target = proposal.request
        if target.design.design_id != source.request.design.design_id:
            raise CampaignError("promotion must validate the source observation's design")
        if source.request.fidelity is self.config.highest_available_fidelity:
            raise CampaignError("source is already at highest available fidelity")
        expected_lineage_id = self._promotion_lineage_id(source)
        if proposal.promotion_lineage_id != expected_lineage_id:
            raise CampaignError(
                "promotion lineage does not match design, target, and model context"
            )
        if target.fidelity is not self.config.highest_available_fidelity:
            raise CampaignError("promotion must target highest available fidelity")
        if source.request.fidelity.ordinal >= target.fidelity.ordinal:
            raise CampaignError("promotion target must be higher fidelity than source")
        if (
            target.objective_specs != source.request.objective_specs
            or target.constraint_specs != source.request.constraint_specs
            or target.requested_replicates != source.request.requested_replicates
            or target.result_context != source.request.result_context
        ):
            raise CampaignError("promotion cannot change result-defining request schema")
        prior_keys = self._promotion_attempt_keys(expected_lineage_id)
        pending_keys = set(self._pending)
        if any(key in pending_keys for key in prior_keys):
            raise CampaignError(
                "a mandatory validation for this promotion source is already pending"
            )
        prior_observations = tuple(
            self._observations[key]
            for key in prior_keys
            if key in self._observations
        )
        if any(
            observation.status is EvaluationStatus.SUCCESS
            for observation in prior_observations
        ):
            raise CampaignError("promotion source already has successful validation")
        failed_keys = {
            observation.request.evaluation_key
            for observation in prior_observations
            if observation.status is EvaluationStatus.FAILURE
        }
        if not prior_keys:
            if proposal.retry_of_evaluation_key is not None:
                raise CampaignError("first validation attempt cannot be marked as a retry")
        else:
            if proposal.retry_of_evaluation_key not in failed_keys:
                raise CampaignError(
                    "retry must identify a terminal failed validation attempt"
                )
            retries_used = len(prior_keys) - 1
            if retries_used >= self.config.maximum_retries_per_promotion:
                raise CampaignError("per-promotion retry limit is exhausted")
        comparable = tuple(
            observation
            for observation in self._observations.values()
            if observation.status is EvaluationStatus.SUCCESS
            and self._comparable_context_id(observation)
            == self._comparable_context_id(source)
        )
        eligibility = {
            item.observation_id: item
            for item in promotion_metadata(
                comparable,
                source.request.objective_specs,
                source.request.constraint_specs,
                highest_available_fidelity=self.config.highest_available_fidelity,
            )
        }[source.observation_id]
        if not eligibility.eligible_for_promotion:
            raise CampaignError("source observation does not pass promotion eligibility")

    def ask(self, proposals: Iterable[Proposal], *, max_jobs: int) -> tuple[PendingJob, ...]:
        """Accept a deterministic subset while respecting pending, cost, and count budgets."""
        if isinstance(max_jobs, bool) or not isinstance(max_jobs, int) or max_jobs < 0:
            raise ValueError("max_jobs must be a non-negative integer")
        capacity = min(max_jobs, self.config.maximum_pending - len(self._pending))
        if capacity <= 0:
            return ()
        unique: dict[str, Proposal] = {}
        for proposal in proposals:
            key = proposal.request.evaluation_key
            if key in self._pending or key in self._observations:
                continue
            incumbent = unique.get(key)
            if incumbent is None or (proposal.mandatory_validation, proposal.score) > (
                incumbent.mandatory_validation,
                incumbent.score,
            ):
                unique[key] = proposal
        ordered = sorted(
            unique.values(),
            key=lambda item: (
                not item.mandatory_validation,
                -item.score,
                item.request.evaluation_key,
            ),
        )
        promotion_sources = [
            item.promotion_lineage_id
            for item in ordered
            if item.mandatory_validation
        ]
        if len(promotion_sources) != len(set(promotion_sources)):
            raise CampaignError(
                "one ask call cannot schedule duplicate promotion-lineage validation"
            )
        retry_parents = [
            item.retry_of_evaluation_key
            for item in ordered
            if item.retry_of_evaluation_key is not None
        ]
        if len(retry_parents) != len(set(retry_parents)):
            raise CampaignError(
                "one ask call cannot schedule duplicate retries of one failure"
            )
        accepted: list[PendingJob] = []
        for proposal in ordered:
            if len(accepted) >= capacity:
                break
            fidelity = proposal.request.fidelity
            self._validate_retry_proposal(proposal)
            self._validate_mandatory_proposal(proposal)
            if (
                proposal.mandatory_validation
                and fidelity is not self.config.highest_available_fidelity
            ):
                raise CampaignError(
                    "mandatory promoted candidates must use highest available fidelity"
                )
            if self._used_count(fidelity) >= self._attempt_limit(fidelity):
                continue
            if (
                fidelity is self.config.highest_available_fidelity
                and proposal.retry_of_evaluation_key is None
                and self._initial_high_fidelity_attempt_count()
                >= self.config.budget_by_fidelity[fidelity]
            ):
                continue
            if (
                self.cost_committed + self._expected_cost(fidelity)
                > self.config.maximum_equivalent_f3_cost
            ):
                continue
            job = PendingJob(proposal, self._ask_counter)
            self._ask_counter += 1
            self._pending[proposal.request.evaluation_key] = job
            if proposal.mandatory_validation:
                assert proposal.promotion_source_id is not None
                self._promotion_lineage[
                    proposal.request.evaluation_key
                ] = proposal.promotion_source_id
                assert proposal.promotion_lineage_id is not None
                self._promotion_lineage_ids[
                    proposal.request.evaluation_key
                ] = proposal.promotion_lineage_id
            if proposal.retry_of_evaluation_key is not None:
                self._retry_keys.add(proposal.request.evaluation_key)
                self._retry_parent[
                    proposal.request.evaluation_key
                ] = proposal.retry_of_evaluation_key
            self._record_event(
                {
                    "type": "ask",
                    "ask_index": job.ask_index,
                    "proposal": _proposal_to_dict(proposal),
                }
            )
            accepted.append(job)
        return tuple(accepted)

    def tell(self, observation: Observation) -> None:
        key = observation.request.evaluation_key
        if key in self._observations:
            raise CampaignError(f"duplicate tell for evaluation {key}")
        if key not in self._pending:
            raise CampaignError(f"tell does not match a pending request: {key}")
        if observation.request != self._pending[key].proposal.request:
            raise CampaignError("tell request payload differs from its pending request")
        pending_proposal = self._pending[key].proposal
        if (
            (
                observation.status is EvaluationStatus.FAILURE
                and observation.request.fidelity
                is self.config.highest_available_fidelity
            )
            or pending_proposal.retry_of_evaluation_key is not None
        ) and observation.charged_cost <= 0.0:
            raise CampaignError(
                "failed F3 and retry attempts require finite positive charged cost"
            )
        self._pending.pop(key)
        self._observations[key] = observation
        self._record_event(
            {
                "type": "tell",
                "observation": observation.to_dict(),
            }
        )

    def reject_pending(self, evaluation_key: str, *, reason: str) -> None:
        """Record a zero-cost pre-execution rejection that is not an attempt."""
        rejection = PreExecutionRejection(evaluation_key, reason)
        if rejection.evaluation_key not in self._pending:
            raise CampaignError("pre-execution rejection must reference pending work")
        self._pending.pop(rejection.evaluation_key)
        self._promotion_lineage.pop(rejection.evaluation_key, None)
        self._promotion_lineage_ids.pop(rejection.evaluation_key, None)
        self._retry_keys.discard(rejection.evaluation_key)
        self._retry_parent.pop(rejection.evaluation_key, None)
        self._rejections.append(rejection)
        self._record_event(
            {
                "type": "reject",
                "evaluation_key": rejection.evaluation_key,
                "reason": rejection.reason,
            }
        )

    def validation_proposal(
        self,
        design: Design,
        *,
        seed: int,
        promotion_source_id: str,
        score: float,
        retry_failed_evaluation_key: str | None = None,
    ) -> Proposal:
        source_by_id = {
            observation.observation_id: observation
            for observation in self._observations.values()
        }
        if promotion_source_id not in source_by_id:
            raise CampaignError("promotion source must be an observed immutable result")
        source = source_by_id[promotion_source_id]
        lineage_id = self._promotion_lineage_id(source)
        proposal = Proposal(
            EvaluationRequest(
                design,
                self.config.highest_available_fidelity,
                seed,
                source.request.requested_replicates,
                source.request.objective_specs,
                source.request.constraint_specs,
                source.request.result_context,
            ),
            acquisition="mandatory-pareto-validation",
            score=score,
            promotion_source_id=promotion_source_id,
            promotion_lineage_id=lineage_id,
            mandatory_validation=True,
            retry_of_evaluation_key=retry_failed_evaluation_key,
        )
        self._validate_mandatory_proposal(proposal)
        return proposal

    def retry_proposal(
        self,
        failed_evaluation_key: str,
        *,
        seed: int,
        score: float,
    ) -> Proposal:
        """Create one explicit bounded F3 retry preserving all result lineage."""
        if failed_evaluation_key not in self._observations:
            raise CampaignError("retry must reference a recorded evaluation")
        parent = self._observations[failed_evaluation_key]
        source_id = self._promotion_lineage.get(failed_evaluation_key)
        lineage_id = self._promotion_lineage_ids.get(failed_evaluation_key)
        proposal = Proposal(
            EvaluationRequest(
                parent.request.design,
                parent.request.fidelity,
                seed,
                parent.request.requested_replicates,
                parent.request.objective_specs,
                parent.request.constraint_specs,
                parent.request.result_context,
            ),
            acquisition="explicit-high-fidelity-retry",
            score=score,
            promotion_source_id=source_id,
            promotion_lineage_id=lineage_id,
            mandatory_validation=source_id is not None,
            retry_of_evaluation_key=failed_evaluation_key,
        )
        self._validate_retry_proposal(proposal)
        if proposal.mandatory_validation:
            self._validate_mandatory_proposal(proposal)
        return proposal

    def stopping_diagnostics(
        self,
        *,
        acquisition_converged: bool,
        verified_hypervolume_stalled: bool,
        surrogate_calibrated: bool,
        promoted_candidates_guardrails_passed: bool,
        iteration_policy_satisfied: bool,
    ) -> StoppingDiagnostics:
        gates = {
            "acquisition_converged": acquisition_converged,
            "verified_hypervolume_stalled": verified_hypervolume_stalled,
            "surrogate_calibrated": surrogate_calibrated,
            "promoted_candidates_guardrails_passed": (
                promoted_candidates_guardrails_passed
            ),
            "iteration_policy_satisfied": iteration_policy_satisfied,
        }
        if any(not isinstance(value, bool) for value in gates.values()):
            raise CampaignError("stopping gate evidence must be boolean")
        completed = len(self._observations)
        failures = sum(item.failure is not None for item in self._observations.values())
        high_attempts = self._high_fidelity_attempt_count()
        high_successes = self._high_fidelity_success_count()
        high_failures = sum(
            observation.request.fidelity
            is self.config.highest_available_fidelity
            and observation.status is EvaluationStatus.FAILURE
            for observation in self._observations.values()
        )
        successful_total = sum(
            observation.status is EvaluationStatus.SUCCESS
            for observation in self._observations.values()
        )
        high_fraction = (
            high_successes / successful_total if successful_total else 0.0
        )
        mandatory_gate_met = (
            high_successes >= self.config.mandatory_high_fidelity_validations
        )
        fraction_gate_met = (
            high_fraction >= self.config.minimum_high_fidelity_fraction
        )
        retry_attempts = len(self._retry_keys)
        retry_capacity = max(
            0,
            self.config.reserved_high_fidelity_retries - retry_attempts,
        )
        retryable_lineages = self._retryable_failure_lineage_count()
        initial_attempt_capacity = max(
            0,
            self.config.budget_by_fidelity[
                self.config.highest_available_fidelity
            ]
            - self._initial_high_fidelity_attempt_count(),
        )
        pending_high = any(
            job.proposal.request.fidelity
            is self.config.highest_available_fidelity
            for job in self._pending.values()
        )
        validation_exhausted = (
            initial_attempt_capacity == 0
            and (retry_capacity == 0 or retryable_lineages == 0)
            and not pending_high
            and not (mandatory_gate_met and fraction_gate_met)
        )
        reasons: list[str] = []
        if self._pending:
            reasons.append("pending evaluations remain")
        if not mandatory_gate_met:
            reasons.append("mandatory highest-fidelity success quota is unmet")
        if not fraction_gate_met:
            reasons.append(
                "highest-fidelity successful-evaluation fraction is unmet"
            )
        if validation_exhausted:
            reasons.append(
                "highest-fidelity attempts exhausted before validation gates"
            )
        if not acquisition_converged:
            reasons.append("acquisition convergence gate is unmet")
        if not verified_hypervolume_stalled:
            reasons.append("verified hypervolume has not stalled")
        if not surrogate_calibrated:
            reasons.append("surrogate calibration gate is unmet")
        if not promoted_candidates_guardrails_passed:
            reasons.append("promoted-candidate guardrails gate is unmet")
        if not iteration_policy_satisfied:
            reasons.append("iteration acquisition policy gate is unmet")
        if self.cost_spent >= self.config.maximum_equivalent_f3_cost:
            reasons.append("equivalent-F3 cost ceiling reached")
        stop = (
            not self._pending
            and mandatory_gate_met
            and fraction_gate_met
            and all(gates.values())
        ) or (
            self.cost_spent >= self.config.maximum_equivalent_f3_cost
            or validation_exhausted
        )
        return StoppingDiagnostics(
            should_stop=stop,
            reasons=tuple(reasons),
            completed=completed,
            failed=failures,
            pending=len(self._pending),
            equivalent_f3_cost_spent=self.cost_spent,
            equivalent_f3_cost_committed=self.cost_committed,
            high_fidelity_attempts=high_attempts,
            high_fidelity_successes=high_successes,
            high_fidelity_failures=high_failures,
            high_fidelity_attempt_limit=self.config.highest_fidelity_attempt_limit,
            retry_capacity_remaining=retry_capacity,
            retryable_failure_lineages=retryable_lineages,
            mandatory_validation_target=(
                self.config.mandatory_high_fidelity_validations
            ),
            mandatory_validation_gate_met=mandatory_gate_met,
            high_fidelity_success_fraction=high_fraction,
            minimum_high_fidelity_fraction=(
                self.config.minimum_high_fidelity_fraction
            ),
            high_fidelity_fraction_gate_met=fraction_gate_met,
            validation_exhausted=validation_exhausted,
        )

    def to_jsonl(self) -> str:
        header = {
            "sequence": -1,
            "type": "campaign",
            "campaign_id": self.campaign_id,
            "config": {
                **asdict(self.config),
                "fidelity_budgets": [
                    [fidelity.value, budget] for fidelity, budget in self.config.fidelity_budgets
                ],
                "highest_available_fidelity": self.config.highest_available_fidelity.value,
            },
            "sources": [
                {**asdict(source), "fidelity": source.fidelity.value} for source in self.sources
            ],
        }
        return "\n".join(
            json.dumps(item, sort_keys=True, separators=(",", ":"), allow_nan=False)
            for item in (header, *self._events)
        )

    @classmethod
    def from_jsonl(cls, payload: str) -> "Campaign":
        if not isinstance(payload, str):
            raise CampaignError("campaign log payload must be a string")

        def reject_nonstandard_constant(value: str) -> None:
            raise ValueError(f"nonstandard JSON constant {value}")

        def reject_duplicate_object_keys(
            pairs: list[tuple[str, Any]],
        ) -> dict[str, Any]:
            result: dict[str, Any] = {}
            for key, value in pairs:
                if key in result:
                    raise CampaignError(
                        f"campaign JSON contains duplicate object key {key!r}"
                    )
                result[key] = value
            return result

        try:
            records = [
                json.loads(
                    line,
                    parse_constant=reject_nonstandard_constant,
                    object_pairs_hook=reject_duplicate_object_keys,
                )
                for line in payload.splitlines()
                if line.strip()
            ]
        except (
            json.JSONDecodeError,
            AttributeError,
            TypeError,
            ValueError,
        ) as exc:
            raise CampaignError("campaign log contains malformed JSON") from exc
        for index, record in enumerate(records):
            _validate_decoded_json(record, f"$[{index}]")
        if not records or not isinstance(records[0], dict):
            raise CampaignError("campaign log has no valid header")
        header = records[0]
        if set(header) != {
            "sequence",
            "type",
            "campaign_id",
            "config",
            "sources",
        } or header.get("sequence") != -1 or header.get("type") != "campaign":
            raise CampaignError("campaign header schema is invalid")
        try:
            config_raw = dict(header["config"])
            if set(config_raw) != {
                "campaign_spec_id",
                "fidelity_budgets",
                "maximum_pending",
                "mandatory_high_fidelity_validations",
                "minimum_high_fidelity_fraction",
                "highest_fidelity_attempt_limit",
                "reserved_high_fidelity_retries",
                "maximum_retries_per_promotion",
                "maximum_equivalent_f3_cost",
                "highest_available_fidelity",
            }:
                raise CampaignError("campaign configuration schema is invalid")
            config_raw["fidelity_budgets"] = tuple(
                (fidelity, budget)
                for fidelity, budget in config_raw["fidelity_budgets"]
            )
            sources = []
            for item in header["sources"]:
                if set(item) != {
                    "fidelity",
                    "name",
                    "description",
                    "equivalent_cost",
                    "default_uncertainty",
                }:
                    raise CampaignError("information source schema is invalid")
                sources.append(InformationSource(**item))
            campaign = cls(CampaignConfig(**config_raw), tuple(sources))
        except CampaignError:
            raise
        except (
            AttributeError,
            IndexError,
            KeyError,
            OverflowError,
            TypeError,
            ValueError,
            DomainError,
        ) as exc:
            raise CampaignError("campaign header values are invalid") from exc
        if campaign.campaign_id != header["campaign_id"]:
            raise CampaignError("campaign header hash does not match configuration")
        expected_sequence = 0
        for event in records[1:]:
            if not isinstance(event, dict):
                raise CampaignError("campaign event must be an object")
            event_type = event.get("type")
            expected_keys = (
                {
                    "sequence",
                    "type",
                    "ask_index",
                    "proposal",
                    "previous_event_hash",
                    "event_hash",
                }
                if event_type == "ask"
                else {
                    "sequence",
                    "type",
                    "observation",
                    "previous_event_hash",
                    "event_hash",
                }
                if event_type == "tell"
                else {
                    "sequence",
                    "type",
                    "evaluation_key",
                    "reason",
                    "previous_event_hash",
                    "event_hash",
                }
                if event_type == "reject"
                else set()
            )
            if not expected_keys or set(event) != expected_keys:
                raise CampaignError("campaign event schema is invalid")
            if event.get("sequence") != expected_sequence:
                raise CampaignError("campaign event sequence is not contiguous")
            expected_previous_hash = (
                records[expected_sequence]["event_hash"]
                if expected_sequence > 0
                else campaign.campaign_id
            )
            if event["previous_event_hash"] != expected_previous_hash:
                raise CampaignError("campaign event hash chain is broken")
            unhashed_event = {
                key: value for key, value in event.items() if key != "event_hash"
            }
            if stable_hash(unhashed_event) != event["event_hash"]:
                raise CampaignError("campaign event hash is invalid")
            expected_sequence += 1
            try:
                if event_type == "ask":
                    proposal = _proposal_from_dict(event["proposal"])
                    accepted = campaign.ask((proposal,), max_jobs=1)
                    if len(accepted) != 1:
                        raise CampaignError(
                            "replayed ask violates campaign safety policy"
                        )
                    if accepted[0].ask_index != event["ask_index"]:
                        raise CampaignError("replayed ask index was tampered")
                elif event_type == "tell":
                    observation = Observation.from_dict(event["observation"])
                    campaign.tell(observation)
                else:
                    campaign.reject_pending(
                        event["evaluation_key"],
                        reason=event["reason"],
                    )
            except CampaignError:
                raise
            except (
                AttributeError,
                IndexError,
                KeyError,
                OverflowError,
                TypeError,
                ValueError,
                DomainError,
            ) as exc:
                raise CampaignError("campaign event values are invalid") from exc
            generated = json.dumps(
                campaign._events[-1],
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            supplied = json.dumps(
                event,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
            if generated != supplied:
                raise CampaignError("campaign event is not canonical or was tampered")
        return campaign
