"""Model-role tribunal models for IX-BlackFox-Cognition.

The model-role tribunal prevents one model or role from becoming the unchecked
authority for planning, implementation, review, evidence evaluation, memory
promotion, self-improvement, or BlackFox-compatible handoff preparation.

This module defines routing and tribunal data models only. Separation and
self-approval enforcement logic is introduced in the next commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, RiskLevel, require_invariant
from ix_blackfox_cognition.work_packages import WorkPackageKind, WorkPackageOutputKind


class ModelRoleKind(StrEnum):
    """Separated cognition roles used by the tribunal."""

    PLANNER = "planner"
    IMPLEMENTER = "implementer"
    CRITIC = "critic"
    SECURITY_REVIEWER = "security_reviewer"
    POLICY_REVIEWER = "policy_reviewer"
    EVIDENCE_REVIEWER = "evidence_reviewer"
    MEMORY_REVIEWER = "memory_reviewer"
    ADVERSARIAL_REVIEWER = "adversarial_reviewer"
    DOCUMENTATION_REVIEWER = "documentation_reviewer"
    HUMAN_REVIEW_COORDINATOR = "human_review_coordinator"


class ModelProviderKind(StrEnum):
    """Kinds of reasoning providers that may occupy model roles."""

    FRONTIER_REMOTE = "frontier_remote"
    LOCAL_OPEN_WEIGHT = "local_open_weight"
    SPECIALIZED_MODEL = "specialized_model"
    SYMBOLIC_ENGINE = "symbolic_engine"
    HUMAN_OPERATOR = "human_operator"
    UNKNOWN = "unknown"


class ModelCapability(StrEnum):
    """Capabilities a model role may claim or be assigned."""

    GOAL_STRUCTURING = "goal_structuring"
    PLANNING = "planning"
    IMPLEMENTATION = "implementation"
    CRITIQUE = "critique"
    SECURITY_REVIEW = "security_review"
    POLICY_REVIEW = "policy_review"
    EVIDENCE_REVIEW = "evidence_review"
    MEMORY_REVIEW = "memory_review"
    ADVERSARIAL_REVIEW = "adversarial_review"
    DOCUMENTATION_REVIEW = "documentation_review"
    SELF_IMPROVEMENT_REVIEW = "self_improvement_review"
    BLACKFOX_HANDOFF_REVIEW = "blackfox_handoff_review"


class RoutePurpose(StrEnum):
    """Purposes for routing a cognitive work package to a role."""

    GENERATE = "generate"
    REVIEW = "review"
    CRITIQUE = "critique"
    RED_TEAM = "red_team"
    EVIDENCE_CHECK = "evidence_check"
    POLICY_CHECK = "policy_check"
    MEMORY_CHECK = "memory_check"
    HUMAN_ESCALATION = "human_escalation"


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    """Identity of a reasoning provider assigned to a role."""

    model_id: str
    provider_kind: ModelProviderKind
    provider_name: str
    model_name: str
    local_only: bool = False
    notes: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.model_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model identity id cannot be blank.",
        )
        require_invariant(
            bool(self.provider_name.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model identity provider name cannot be blank.",
        )
        require_invariant(
            bool(self.model_name.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model identity model name cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class ModelRoleSpec:
    """A separated role a model may perform inside governed cognition."""

    role_id: str
    role_kind: ModelRoleKind
    description: str
    allowed_package_kinds: tuple[WorkPackageKind, ...]
    required_capabilities: tuple[ModelCapability, ...]
    allowed_output_kinds: tuple[WorkPackageOutputKind, ...]
    may_generate: bool = True
    may_review: bool = False
    may_approve: bool = False
    human_authority_role: bool = False

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.role_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model role id cannot be blank.",
        )
        require_invariant(
            bool(self.description.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model role description cannot be blank.",
        )
        require_invariant(
            bool(self.allowed_package_kinds),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Model role must allow at least one work package kind.",
        )
        require_invariant(
            bool(self.required_capabilities),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Model role must require at least one capability.",
        )
        require_invariant(
            bool(self.allowed_output_kinds),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Model role must allow at least one output kind.",
        )

        if self.human_authority_role:
            require_invariant(
                not self.may_generate,
                FailureMode.SELF_APPROVAL_ATTEMPT,
                "Human authority roles cannot also be generation roles.",
            )

        if self.may_approve:
            require_invariant(
                self.human_authority_role,
                FailureMode.SELF_APPROVAL_ATTEMPT,
                "Only explicit human authority roles may approve.",
            )

    def can_handle_package_kind(self, package_kind: WorkPackageKind) -> bool:
        """Return whether this role can handle a package kind."""

        return package_kind in self.allowed_package_kinds

    def can_emit_output_kind(self, output_kind: WorkPackageOutputKind) -> bool:
        """Return whether this role can emit an output kind."""

        return output_kind in self.allowed_output_kinds


@dataclass(frozen=True, slots=True)
class ModelAssignment:
    """Assignment of a model identity to a separated role."""

    assignment_id: str
    role: ModelRoleSpec
    model: ModelIdentity
    maximum_risk: RiskLevel = RiskLevel.MODERATE
    enabled: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.assignment_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model assignment id cannot be blank.",
        )

        if self.role.human_authority_role:
            require_invariant(
                self.model.provider_kind == ModelProviderKind.HUMAN_OPERATOR,
                FailureMode.SELF_APPROVAL_ATTEMPT,
                "Human authority roles must be assigned to a human operator identity.",
            )

        if self.model.provider_kind == ModelProviderKind.HUMAN_OPERATOR:
            require_invariant(
                self.role.human_authority_role,
                FailureMode.UNCLEAR_AUTHORITY,
                "Human operator identities must be assigned only to human authority roles.",
            )

    def can_accept_risk(self, risk_level: RiskLevel) -> bool:
        """Return whether this assignment can accept a requested risk level."""

        return self.enabled and risk_level.rank() <= self.maximum_risk.rank()


@dataclass(frozen=True, slots=True)
class ModelRouteRequest:
    """Request to route a cognitive work package to a separated role."""

    route_request_id: str
    package_id: str
    package_kind: WorkPackageKind
    expected_outputs: tuple[WorkPackageOutputKind, ...]
    purpose: RoutePurpose
    risk_level: RiskLevel
    originating_role_id: str | None = None
    prior_output_id: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.route_request_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model route request id cannot be blank.",
        )
        require_invariant(
            bool(self.package_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Model route request package id cannot be blank.",
        )
        require_invariant(
            bool(self.expected_outputs),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Model route request must define expected outputs.",
        )


@dataclass(frozen=True, slots=True)
class ModelRouteDecision:
    """Decision assigning or rejecting a route request."""

    decision_id: str
    route_request_id: str
    outcome: DecisionOutcome
    reason: str
    selected_assignment_id: str | None = None
    selected_role_id: str | None = None
    selected_model_id: str | None = None
    failure_modes: tuple[FailureMode, ...] = field(default_factory=tuple)
    requires_human_review: bool = False

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.decision_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model route decision id cannot be blank.",
        )
        require_invariant(
            bool(self.route_request_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model route decision request id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model route decision reason cannot be blank.",
        )

        if self.outcome == DecisionOutcome.ALLOW:
            require_invariant(
                self.selected_assignment_id is not None
                and bool(self.selected_assignment_id.strip())
                and self.selected_role_id is not None
                and bool(self.selected_role_id.strip())
                and self.selected_model_id is not None
                and bool(self.selected_model_id.strip()),
                FailureMode.UNCLEAR_AUTHORITY,
                "Allowed model route decisions require assignment, role, and model ids.",
            )

    @property
    def allowed(self) -> bool:
        """Return whether the route was allowed."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def failed_closed(self) -> bool:
        """Return whether the route failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED


@dataclass(frozen=True, slots=True)
class ModelTribunal:
    """Immutable registry of separated model-role assignments."""

    tribunal_id: str
    assignments: tuple[ModelAssignment, ...]
    required_separation: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.tribunal_id.strip()),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model tribunal id cannot be blank.",
        )
        require_invariant(
            bool(self.assignments),
            FailureMode.UNCLEAR_AUTHORITY,
            "Model tribunal must contain at least one assignment.",
        )

        assignment_ids = [assignment.assignment_id for assignment in self.assignments]
        require_invariant(
            len(assignment_ids) == len(set(assignment_ids)),
            FailureMode.CONTRADICTION,
            "Model tribunal cannot contain duplicate assignment ids.",
        )

        role_ids = [assignment.role.role_id for assignment in self.assignments]
        require_invariant(
            len(role_ids) == len(set(role_ids)),
            FailureMode.CONTRADICTION,
            "Model tribunal cannot contain duplicate role ids.",
        )

    def assignment_by_id(self, assignment_id: str) -> ModelAssignment | None:
        """Return an assignment by id, if present."""

        for assignment in self.assignments:
            if assignment.assignment_id == assignment_id:
                return assignment
        return None

    def assignments_for_package(
        self,
        package_kind: WorkPackageKind,
        risk_level: RiskLevel,
    ) -> tuple[ModelAssignment, ...]:
        """Return enabled assignments that can handle a package kind and risk level."""

        return tuple(
            assignment
            for assignment in self.assignments
            if assignment.role.can_handle_package_kind(package_kind)
            and assignment.can_accept_risk(risk_level)
        )
