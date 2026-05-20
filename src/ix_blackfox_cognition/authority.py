"""Authority kernel models for IX-BlackFox-Cognition.

The authority kernel defines what the cognition layer may propose, what must be
escalated to humans, and what remains forbidden. This module is intentionally
data-model focused; firewall decision logic is introduced in the next commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import (
    ActorKind,
    DecisionOutcome,
    FailureMode,
    RiskLevel,
    require_invariant,
)


class CognitionPermission(StrEnum):
    """Actions the cognition layer may be allowed to perform under policy."""

    THINK = "think"
    STRUCTURE_GOAL = "structure_goal"
    PROPOSE_PLAN = "propose_plan"
    PROPOSE_WORK_PACKAGE = "propose_work_package"
    PROPOSE_MEMORY_UPDATE = "propose_memory_update"
    PROPOSE_SELF_IMPROVEMENT = "propose_self_improvement"
    PREPARE_BLACKFOX_HANDOFF = "prepare_blackfox_handoff"
    REQUEST_HUMAN_REVIEW = "request_human_review"
    RECORD_EVIDENCE_REFERENCE = "record_evidence_reference"
    RECORD_RECEIPT_REFERENCE = "record_receipt_reference"


class ForbiddenAction(StrEnum):
    """Actions that are forbidden without explicit human authority or entirely blocked."""

    SELF_AUTHORIZE = "self_authorize"
    SELF_APPROVE = "self_approve"
    SILENTLY_MUTATE_STATE = "silently_mutate_state"
    BYPASS_POLICY = "bypass_policy"
    EXECUTE_OPERATIONAL_ACTION = "execute_operational_action"
    MODIFY_POLICY_WITHOUT_APPROVAL = "modify_policy_without_approval"
    PROMOTE_MEMORY_WITHOUT_REVIEW = "promote_memory_without_review"
    PROMOTE_SELF_IMPROVEMENT_WITHOUT_REVIEW = "promote_self_improvement_without_review"
    CLAIM_AGI = "claim_agi"
    CLAIM_AUTONOMOUS_AGI = "claim_autonomous_agi"
    CLAIM_SELF_AWARENESS = "claim_self_awareness"
    CLAIM_CERTIFICATION = "claim_certification"
    CLAIM_PRODUCTION_READINESS = "claim_production_readiness"
    CLAIM_GOVERNMENT_OR_DEFENSE_AFFILIATION = "claim_government_or_defense_affiliation"


class AuthorityLevel(StrEnum):
    """Authority levels used by the governed cognition layer."""

    OBSERVE = "observe"
    THINK_ONLY = "think_only"
    PROPOSE_ONLY = "propose_only"
    STRUCTURE_ONLY = "structure_only"
    PREPARE_HANDOFF = "prepare_handoff"
    HUMAN_APPROVED = "human_approved"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class HumanAuthorityRequirement:
    """A requirement that an action be reviewed or approved by a human."""

    requirement_id: str
    reason: str
    required_for: tuple[CognitionPermission | ForbiddenAction | str, ...]
    minimum_risk: RiskLevel = RiskLevel.MODERATE
    explicit_approval_required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.requirement_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Human authority requirement id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Human authority requirement reason cannot be blank.",
        )
        require_invariant(
            bool(self.required_for),
            FailureMode.UNCLEAR_AUTHORITY,
            "Human authority requirement must identify at least one covered action.",
        )


@dataclass(frozen=True, slots=True)
class AuthorityBoundary:
    """The boundary describing what an actor may do inside cognition."""

    actor_kind: ActorKind
    actor_id: str
    authority_level: AuthorityLevel
    allowed_permissions: tuple[CognitionPermission, ...] = field(default_factory=tuple)
    forbidden_actions: tuple[ForbiddenAction, ...] = field(default_factory=tuple)
    human_requirements: tuple[HumanAuthorityRequirement, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.actor_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority boundary actor id cannot be blank.",
        )
        require_invariant(
            self.authority_level != AuthorityLevel.HUMAN_APPROVED
            or self.actor_kind == ActorKind.HUMAN,
            FailureMode.SELF_APPROVAL_ATTEMPT,
            "Only a human actor may hold direct human-approved authority.",
        )

    def allows_permission(self, permission: CognitionPermission) -> bool:
        """Return whether the boundary explicitly allows a permission."""

        return permission in self.allowed_permissions and self.authority_level != AuthorityLevel.BLOCKED

    def forbids_action(self, action: ForbiddenAction) -> bool:
        """Return whether the boundary explicitly forbids an action."""

        return action in self.forbidden_actions


@dataclass(frozen=True, slots=True)
class AuthorityRequest:
    """A request to perform a bounded cognition operation."""

    request_id: str
    actor_kind: ActorKind
    actor_id: str
    requested_permission: CognitionPermission
    justification: str
    risk_level: RiskLevel
    target: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    human_approval_id: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.request_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority request id cannot be blank.",
        )
        require_invariant(
            bool(self.actor_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority request actor id cannot be blank.",
        )
        require_invariant(
            bool(self.justification.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority request justification cannot be blank.",
        )

    @property
    def has_human_approval(self) -> bool:
        """Return whether this request includes an explicit human approval id."""

        return self.human_approval_id is not None and bool(self.human_approval_id.strip())


@dataclass(frozen=True, slots=True)
class AuthorityDecisionRecord:
    """Immutable record of an authority decision."""

    decision_id: str
    request_id: str
    outcome: DecisionOutcome
    actor_kind: ActorKind
    reason: str
    risk_level: RiskLevel
    failure_modes: tuple[FailureMode, ...] = field(default_factory=tuple)
    required_human_review: bool = False

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.decision_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority decision id cannot be blank.",
        )
        require_invariant(
            bool(self.request_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority decision request id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority decision reason cannot be blank.",
        )

    @property
    def allowed(self) -> bool:
        """Return whether the decision grants permission."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def failed_closed(self) -> bool:
        """Return whether the decision failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED


@dataclass(frozen=True, slots=True)
class AuthoritySnapshot:
    """A frozen snapshot of known authority boundaries."""

    snapshot_id: str
    boundaries: tuple[AuthorityBoundary, ...]
    doctrine: str = (
        "Model thinks → Cognition structures → BlackFox governs → humans authorize "
        "→ evidence decides trust."
    )

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.snapshot_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Authority snapshot id cannot be blank.",
        )

    def boundary_for(self, actor_kind: ActorKind, actor_id: str) -> AuthorityBoundary | None:
        """Return the authority boundary for an actor, if one exists."""

        for boundary in self.boundaries:
            if boundary.actor_kind == actor_kind and boundary.actor_id == actor_id:
                return boundary
        return None
