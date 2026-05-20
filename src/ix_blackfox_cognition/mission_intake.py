"""Mission intake logic for IX-BlackFox-Cognition.

Mission intake converts human intent into a bounded mission envelope before the
system may form plan graphs, route model roles, propose memory updates, or
prepare BlackFox-compatible handoffs.

The intake layer is conservative by design:
- blank goals fail closed,
- every mission receives explicit constraints,
- every mission receives evidence-oriented acceptance criteria,
- higher-risk missions receive review checkpoints,
- forbidden actions remain visible instead of implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ix_blackfox_cognition.core import FailureMode, RiskLevel, WorkState, require_invariant
from ix_blackfox_cognition.mission import (
    AcceptanceCriterion,
    HumanGoal,
    MissionAssumption,
    MissionConstraint,
    MissionConstraintKind,
    MissionEnvelope,
    MissionReviewTrigger,
    MissionRisk,
    ReviewCheckpoint,
    RollbackNeed,
)


DEFAULT_FORBIDDEN_ACTIONS = (
    "self_authorize",
    "self_approve",
    "silently_mutate_state",
    "bypass_policy",
    "execute_operational_action_without_blackfox_handoff",
    "promote_memory_without_review",
    "promote_self_improvement_without_review",
    "claim_agi",
    "claim_autonomous_agi",
    "claim_self_awareness",
    "claim_certification",
    "claim_production_readiness",
    "claim_government_or_defense_affiliation",
)


@dataclass(frozen=True, slots=True)
class MissionIntakeDefaults:
    """Default boundaries added to every structured mission."""

    require_evidence_constraint: bool = True
    require_human_authority_constraint: bool = True
    require_no_direct_execution_constraint: bool = True
    require_default_acceptance_criterion: bool = True
    require_default_rollback_need: bool = True
    high_risk_review_threshold: RiskLevel = RiskLevel.HIGH
    default_forbidden_actions: tuple[str, ...] = DEFAULT_FORBIDDEN_ACTIONS

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.default_forbidden_actions),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission intake defaults must include explicit forbidden actions.",
        )


@dataclass(frozen=True, slots=True)
class MissionIntakeRequest:
    """Request to convert human intent into a bounded mission envelope."""

    goal_id: str
    statement: str
    requester: str = "human"
    context: str | None = None
    mission_id: str | None = None
    summary: str | None = None
    constraints: tuple[MissionConstraint, ...] = field(default_factory=tuple)
    assumptions: tuple[MissionAssumption, ...] = field(default_factory=tuple)
    risks: tuple[MissionRisk, ...] = field(default_factory=tuple)
    acceptance_criteria: tuple[AcceptanceCriterion, ...] = field(default_factory=tuple)
    rollback_needs: tuple[RollbackNeed, ...] = field(default_factory=tuple)
    review_checkpoints: tuple[ReviewCheckpoint, ...] = field(default_factory=tuple)
    forbidden_actions: tuple[str, ...] = field(default_factory=tuple)
    evidence_requirement_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.goal_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Mission intake goal id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission intake statement cannot be blank.",
        )
        require_invariant(
            bool(self.requester.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Mission intake requester cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MissionIntakeResult:
    """Result of mission intake."""

    envelope: MissionEnvelope
    added_default_constraints: tuple[str, ...]
    added_default_acceptance_criteria: tuple[str, ...]
    added_default_review_checkpoints: tuple[str, ...]
    added_default_rollback_needs: tuple[str, ...]

    @property
    def bounded(self) -> bool:
        """Return whether the produced mission envelope is bounded."""

        return self.envelope.is_bounded


class MissionIntakeEngine:
    """Deterministic mission intake engine."""

    def __init__(self, defaults: MissionIntakeDefaults | None = None) -> None:
        self.defaults = defaults or MissionIntakeDefaults()

    def structure(self, request: MissionIntakeRequest) -> MissionIntakeResult:
        """Convert a mission intake request into a bounded mission envelope."""

        goal = HumanGoal(
            goal_id=request.goal_id,
            statement=request.statement,
            requester=request.requester,
            context=request.context,
        )

        constraints, added_constraint_ids = self._with_default_constraints(
            request=request,
        )
        acceptance_criteria, added_acceptance_ids = self._with_default_acceptance_criteria(
            request=request,
        )
        rollback_needs, added_rollback_ids = self._with_default_rollback_needs(
            request=request,
        )
        review_checkpoints, added_review_ids = self._with_default_review_checkpoints(
            request=request,
        )

        forbidden_actions = self._merge_forbidden_actions(request.forbidden_actions)

        mission_id = request.mission_id or f"mission:{request.goal_id}"
        summary = request.summary or self._summary_from_goal(request.statement)

        envelope = MissionEnvelope(
            mission_id=mission_id,
            goal=goal,
            summary=summary,
            status=WorkState.DRAFT,
            constraints=constraints,
            assumptions=request.assumptions,
            risks=request.risks,
            acceptance_criteria=acceptance_criteria,
            rollback_needs=rollback_needs,
            review_checkpoints=review_checkpoints,
            forbidden_actions=forbidden_actions,
            evidence_requirement_ids=request.evidence_requirement_ids,
        )

        return MissionIntakeResult(
            envelope=envelope,
            added_default_constraints=added_constraint_ids,
            added_default_acceptance_criteria=added_acceptance_ids,
            added_default_review_checkpoints=added_review_ids,
            added_default_rollback_needs=added_rollback_ids,
        )

    def _with_default_constraints(
        self,
        request: MissionIntakeRequest,
    ) -> tuple[tuple[MissionConstraint, ...], tuple[str, ...]]:
        constraints = list(request.constraints)
        added_ids: list[str] = []

        if self.defaults.require_evidence_constraint:
            constraint = MissionConstraint(
                constraint_id=f"constraint:{request.goal_id}:evidence-required",
                kind=MissionConstraintKind.EVIDENCE,
                statement=(
                    "Mission outputs must distinguish evidence-backed claims from "
                    "assumptions, unsupported claims, and model confidence."
                ),
            )
            constraints.append(constraint)
            added_ids.append(constraint.constraint_id)

        if self.defaults.require_human_authority_constraint:
            constraint = MissionConstraint(
                constraint_id=f"constraint:{request.goal_id}:human-authority",
                kind=MissionConstraintKind.HUMAN_AUTHORITY,
                statement=(
                    "Human authority remains explicit; cognition may structure and "
                    "propose but may not self-authorize or self-approve."
                ),
            )
            constraints.append(constraint)
            added_ids.append(constraint.constraint_id)

        if self.defaults.require_no_direct_execution_constraint:
            constraint = MissionConstraint(
                constraint_id=f"constraint:{request.goal_id}:blackfox-handoff",
                kind=MissionConstraintKind.POLICY,
                statement=(
                    "Risk-bearing action must be prepared as a BlackFox-compatible "
                    "handoff candidate instead of directly executed by cognition."
                ),
            )
            constraints.append(constraint)
            added_ids.append(constraint.constraint_id)

        return tuple(constraints), tuple(added_ids)

    def _with_default_acceptance_criteria(
        self,
        request: MissionIntakeRequest,
    ) -> tuple[tuple[AcceptanceCriterion, ...], tuple[str, ...]]:
        criteria = list(request.acceptance_criteria)
        added_ids: list[str] = []

        if self.defaults.require_default_acceptance_criterion:
            criterion = AcceptanceCriterion(
                criterion_id=f"criterion:{request.goal_id}:bounded-envelope",
                statement=(
                    "The mission must produce a bounded envelope with visible scope, "
                    "constraints, evidence expectations, forbidden actions, and human "
                    "review boundaries before planning continues."
                ),
                evidence_required=True,
            )
            criteria.append(criterion)
            added_ids.append(criterion.criterion_id)

        return tuple(criteria), tuple(added_ids)

    def _with_default_rollback_needs(
        self,
        request: MissionIntakeRequest,
    ) -> tuple[tuple[RollbackNeed, ...], tuple[str, ...]]:
        rollback_needs = list(request.rollback_needs)
        added_ids: list[str] = []

        if self.defaults.require_default_rollback_need:
            rollback_need = RollbackNeed(
                rollback_id=f"rollback:{request.goal_id}:no-silent-promotion",
                statement=(
                    "Any later memory promotion, self-improvement promotion, or "
                    "BlackFox handoff must preserve a reviewable rollback path."
                ),
            )
            rollback_needs.append(rollback_need)
            added_ids.append(rollback_need.rollback_id)

        return tuple(rollback_needs), tuple(added_ids)

    def _with_default_review_checkpoints(
        self,
        request: MissionIntakeRequest,
    ) -> tuple[tuple[ReviewCheckpoint, ...], tuple[str, ...]]:
        checkpoints = list(request.review_checkpoints)
        added_ids: list[str] = []

        if self._highest_risk(request).rank() >= self.defaults.high_risk_review_threshold.rank():
            checkpoint = ReviewCheckpoint(
                checkpoint_id=f"review:{request.goal_id}:high-risk",
                trigger=MissionReviewTrigger.HIGH_RISK,
                reason="High-risk missions require explicit human review before action handoff.",
            )
            checkpoints.append(checkpoint)
            added_ids.append(checkpoint.checkpoint_id)

        return tuple(checkpoints), tuple(added_ids)

    def _merge_forbidden_actions(self, requested_forbidden: tuple[str, ...]) -> tuple[str, ...]:
        merged: list[str] = []

        for action in (*self.defaults.default_forbidden_actions, *requested_forbidden):
            cleaned = action.strip()
            if cleaned and cleaned not in merged:
                merged.append(cleaned)

        require_invariant(
            bool(merged),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission intake must preserve at least one forbidden action.",
        )
        return tuple(merged)

    def _highest_risk(self, request: MissionIntakeRequest) -> RiskLevel:
        if not request.risks:
            return RiskLevel.LOW

        return max((risk.level for risk in request.risks), key=lambda level: level.rank())

    def _summary_from_goal(self, statement: str) -> str:
        cleaned = " ".join(statement.strip().split())
        if len(cleaned) <= 160:
            return cleaned
        return f"{cleaned[:157].rstrip()}..."


def structure_mission(request: MissionIntakeRequest) -> MissionIntakeResult:
    """Structure human intent into a bounded mission envelope using default rules."""

    return MissionIntakeEngine().structure(request)
