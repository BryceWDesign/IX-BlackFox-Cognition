"""Mission envelope models for IX-BlackFox-Cognition.

A mission envelope converts human intent into a bounded cognition object before
planning, model routing, memory updates, or BlackFox-compatible handoff can
begin. This module is intentionally data-model focused; mission intake logic is
introduced in the next commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import FailureMode, RiskLevel, WorkState, require_invariant


class MissionConstraintKind(StrEnum):
    """Kinds of constraints that can bound a mission."""

    SCOPE = "scope"
    SAFETY = "safety"
    POLICY = "policy"
    EVIDENCE = "evidence"
    TIME = "time"
    RESOURCE = "resource"
    LEGAL = "legal"
    LICENSE = "license"
    HUMAN_AUTHORITY = "human_authority"


class MissionAssumptionState(StrEnum):
    """Status values for assumptions inside a mission envelope."""

    UNVERIFIED = "unverified"
    NEEDS_EVIDENCE = "needs_evidence"
    ACCEPTED_FOR_PLANNING = "accepted_for_planning"
    CONTRADICTED = "contradicted"
    REJECTED = "rejected"


class MissionReviewTrigger(StrEnum):
    """Events that should force or request human review."""

    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"
    UNCLEAR_SCOPE = "unclear_scope"
    MISSING_EVIDENCE = "missing_evidence"
    MEMORY_PROMOTION = "memory_promotion"
    SELF_IMPROVEMENT = "self_improvement"
    BLACKFOX_HANDOFF = "blackfox_handoff"
    POLICY_CONFLICT = "policy_conflict"
    HUMAN_REQUESTED = "human_requested"


@dataclass(frozen=True, slots=True)
class HumanGoal:
    """Human-supplied goal before mission structuring."""

    goal_id: str
    statement: str
    requester: str = "human"
    context: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.goal_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Human goal id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Human goal statement cannot be blank.",
        )
        require_invariant(
            bool(self.requester.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Human goal requester cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MissionConstraint:
    """A bounded constraint that limits mission interpretation or execution."""

    constraint_id: str
    kind: MissionConstraintKind
    statement: str
    mandatory: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.constraint_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission constraint id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission constraint statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MissionAssumption:
    """An assumption that must remain visible instead of becoming hidden context."""

    assumption_id: str
    statement: str
    state: MissionAssumptionState = MissionAssumptionState.UNVERIFIED
    evidence_required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.assumption_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Mission assumption id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Mission assumption statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MissionRisk:
    """A known or anticipated mission risk."""

    risk_id: str
    statement: str
    level: RiskLevel
    mitigation: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.risk_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission risk id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission risk statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class AcceptanceCriterion:
    """A criterion that must be satisfied before a mission can be considered complete."""

    criterion_id: str
    statement: str
    evidence_required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.criterion_id.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Acceptance criterion id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Acceptance criterion statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class RollbackNeed:
    """A rollback or reversal need identified during mission setup."""

    rollback_id: str
    statement: str
    required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.rollback_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Rollback need id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Rollback need statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class ReviewCheckpoint:
    """A point where the mission must stop for review or explicit approval."""

    checkpoint_id: str
    trigger: MissionReviewTrigger
    reason: str
    required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.checkpoint_id.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Review checkpoint id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Review checkpoint reason cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MissionEnvelope:
    """Bounded mission object derived from a human goal."""

    mission_id: str
    goal: HumanGoal
    summary: str
    status: WorkState = WorkState.DRAFT
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
            bool(self.mission_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission id cannot be blank.",
        )
        require_invariant(
            bool(self.summary.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission summary cannot be blank.",
        )
        require_invariant(
            bool(self.constraints),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Mission envelope must include at least one explicit constraint.",
        )
        require_invariant(
            bool(self.acceptance_criteria),
            FailureMode.MISSING_EVIDENCE,
            "Mission envelope must include at least one acceptance criterion.",
        )

    @property
    def highest_risk(self) -> RiskLevel:
        """Return the highest mission risk, or low risk if no risks are recorded."""

        if not self.risks:
            return RiskLevel.LOW

        return max((risk.level for risk in self.risks), key=lambda level: level.rank())

    @property
    def requires_human_review(self) -> bool:
        """Return whether the mission has a required review checkpoint."""

        return any(checkpoint.required for checkpoint in self.review_checkpoints)

    @property
    def has_unverified_assumptions(self) -> bool:
        """Return whether any mission assumption remains unverified."""

        return any(
            assumption.state
            in (
                MissionAssumptionState.UNVERIFIED,
                MissionAssumptionState.NEEDS_EVIDENCE,
            )
            for assumption in self.assumptions
        )

    @property
    def is_bounded(self) -> bool:
        """Return whether the envelope contains the minimum bounded-mission shape."""

        return bool(self.constraints and self.acceptance_criteria and self.summary.strip())
