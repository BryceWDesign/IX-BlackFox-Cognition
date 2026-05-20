"""Cognitive work package models for IX-BlackFox-Cognition.

A cognitive work package is the bounded unit of thought/work produced from a
proof-carrying plan node. Work packages do not execute operational action. They
define scope, evidence needs, authority requirements, review gates, rollback
needs, and expected outputs so later routing, memory, sentinel, and BlackFox
handoff layers can reason safely.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import FailureMode, RiskLevel, WorkState, require_invariant
from ix_blackfox_cognition.evidence import EvidenceContract
from ix_blackfox_cognition.planning import PlanNode, PlanNodeKind


class WorkPackageKind(StrEnum):
    """Kinds of cognitive work packages."""

    RESEARCH = "research"
    DESIGN = "design"
    IMPLEMENTATION = "implementation"
    TEST = "test"
    REVIEW = "review"
    SECURITY_REVIEW = "security_review"
    POLICY_REVIEW = "policy_review"
    MEMORY_REVIEW = "memory_review"
    EVIDENCE_REVIEW = "evidence_review"
    DOCUMENTATION = "documentation"
    MEMORY_UPDATE = "memory_update"
    SELF_IMPROVEMENT_PROPOSAL = "self_improvement_proposal"
    BLACKFOX_HANDOFF = "blackfox_handoff"


class WorkPackageOutputKind(StrEnum):
    """Expected output classes for cognitive work packages."""

    CLAIMS = "claims"
    EVIDENCE_REFERENCES = "evidence_references"
    PLAN_UPDATE = "plan_update"
    REVIEW_RECORD = "review_record"
    TEST_PLAN = "test_plan"
    RISK_REGISTER = "risk_register"
    MEMORY_UPDATE_PROPOSAL = "memory_update_proposal"
    SELF_IMPROVEMENT_PROPOSAL = "self_improvement_proposal"
    BLACKFOX_ACTION_CANDIDATE = "blackfox_action_candidate"
    DOCUMENTATION = "documentation"


@dataclass(frozen=True, slots=True)
class WorkPackageReviewGate:
    """Review gate required before a work package can be accepted or promoted."""

    gate_id: str
    reviewer_role: str
    reason: str
    human_required: bool = False

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.gate_id.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Work package review gate id cannot be blank.",
        )
        require_invariant(
            bool(self.reviewer_role.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Work package review gate reviewer role cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Work package review gate reason cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class WorkPackageDependency:
    """Dependency between work packages or source plan nodes."""

    dependency_id: str
    required_before: str
    required_after: str
    reason: str

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.dependency_id.strip()),
            FailureMode.CONTRADICTION,
            "Work package dependency id cannot be blank.",
        )
        require_invariant(
            bool(self.required_before.strip()),
            FailureMode.CONTRADICTION,
            "Work package dependency required-before id cannot be blank.",
        )
        require_invariant(
            bool(self.required_after.strip()),
            FailureMode.CONTRADICTION,
            "Work package dependency required-after id cannot be blank.",
        )
        require_invariant(
            self.required_before != self.required_after,
            FailureMode.CONTRADICTION,
            "Work package dependency cannot point an item at itself.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.CONTRADICTION,
            "Work package dependency reason cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class CognitiveWorkPackage:
    """Bounded unit of cognition derived from a plan node."""

    package_id: str
    kind: WorkPackageKind
    objective: str
    scope: str
    status: WorkState = WorkState.DRAFT
    risk_level: RiskLevel = RiskLevel.LOW
    source_plan_node_id: str | None = None
    required_belief_ids: tuple[str, ...] = field(default_factory=tuple)
    required_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_requirement_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_contract: EvidenceContract | None = None
    review_gates: tuple[WorkPackageReviewGate, ...] = field(default_factory=tuple)
    rollback_requirements: tuple[str, ...] = field(default_factory=tuple)
    expected_outputs: tuple[WorkPackageOutputKind, ...] = field(default_factory=tuple)
    forbidden_actions: tuple[str, ...] = field(default_factory=tuple)
    dependencies: tuple[WorkPackageDependency, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.package_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Cognitive work package id cannot be blank.",
        )
        require_invariant(
            bool(self.objective.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Cognitive work package objective cannot be blank.",
        )
        require_invariant(
            bool(self.scope.strip()),
            FailureMode.SCOPE_CREEP,
            "Cognitive work package scope cannot be blank.",
        )
        require_invariant(
            bool(self.expected_outputs),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Cognitive work package must define at least one expected output.",
        )
        require_invariant(
            bool(self.evidence_requirement_ids) or self.evidence_contract is not None,
            FailureMode.MISSING_EVIDENCE,
            "Cognitive work package must define evidence requirements or an evidence contract.",
        )
        require_invariant(
            bool(self.forbidden_actions),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Cognitive work package must preserve explicit forbidden actions.",
        )

        if self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            require_invariant(
                self.requires_review,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "High-risk and critical work packages require review gates.",
            )

        if self.is_action_adjacent:
            require_invariant(
                self.requires_review,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "Action-adjacent work packages require review gates.",
            )
            require_invariant(
                bool(self.rollback_requirements),
                FailureMode.UNKNOWN_ACTION_TYPE,
                "Action-adjacent work packages require rollback requirements.",
            )

        if self.kind in (
            WorkPackageKind.MEMORY_UPDATE,
            WorkPackageKind.SELF_IMPROVEMENT_PROPOSAL,
            WorkPackageKind.BLACKFOX_HANDOFF,
        ):
            require_invariant(
                self.requires_human_review,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "Memory update, self-improvement, and BlackFox handoff packages require human review.",
            )

    @property
    def requires_review(self) -> bool:
        """Return whether the package has at least one review gate."""

        return bool(self.review_gates)

    @property
    def requires_human_review(self) -> bool:
        """Return whether the package has at least one human-required review gate."""

        return any(gate.human_required for gate in self.review_gates)

    @property
    def is_action_adjacent(self) -> bool:
        """Return whether the package is close to executable or promotable action."""

        return self.kind in (
            WorkPackageKind.IMPLEMENTATION,
            WorkPackageKind.TEST,
            WorkPackageKind.MEMORY_UPDATE,
            WorkPackageKind.SELF_IMPROVEMENT_PROPOSAL,
            WorkPackageKind.BLACKFOX_HANDOFF,
        )

    @property
    def has_satisfied_evidence_contract(self) -> bool:
        """Return whether the package has a satisfied, non-falsified evidence contract."""

        return self.evidence_contract is not None and self.evidence_contract.trust_eligible

    @classmethod
    def from_plan_node(
        cls,
        *,
        package_id: str,
        node: PlanNode,
        expected_outputs: tuple[WorkPackageOutputKind, ...],
        forbidden_actions: tuple[str, ...],
        review_gates: tuple[WorkPackageReviewGate, ...] = (),
        rollback_requirements: tuple[str, ...] = (),
        dependencies: tuple[WorkPackageDependency, ...] = (),
    ) -> CognitiveWorkPackage:
        """Create a cognitive work package from a proof-carrying plan node."""

        require_invariant(
            node.scope is not None and bool(node.scope.strip()),
            FailureMode.SCOPE_CREEP,
            "Source plan node must define scope before work package conversion.",
        )

        return cls(
            package_id=package_id,
            kind=_kind_from_plan_node(node.kind),
            objective=node.objective,
            scope=node.scope,
            status=node.status,
            risk_level=node.risk_level,
            source_plan_node_id=node.node_id,
            required_belief_ids=node.required_belief_ids,
            required_claim_ids=node.required_claim_ids,
            evidence_requirement_ids=node.proof_requirement_ids,
            evidence_contract=node.evidence_contract,
            review_gates=review_gates,
            rollback_requirements=rollback_requirements,
            expected_outputs=expected_outputs,
            forbidden_actions=forbidden_actions,
            dependencies=dependencies,
        )


@dataclass(frozen=True, slots=True)
class WorkPackageBatch:
    """Immutable batch of cognitive work packages for a mission or plan graph."""

    batch_id: str
    mission_id: str
    plan_graph_id: str
    packages: tuple[CognitiveWorkPackage, ...]
    dependencies: tuple[WorkPackageDependency, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.batch_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Work package batch id cannot be blank.",
        )
        require_invariant(
            bool(self.mission_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Work package batch mission id cannot be blank.",
        )
        require_invariant(
            bool(self.plan_graph_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Work package batch plan graph id cannot be blank.",
        )
        require_invariant(
            bool(self.packages),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Work package batch must contain at least one package.",
        )

        package_ids = [package.package_id for package in self.packages]
        require_invariant(
            len(package_ids) == len(set(package_ids)),
            FailureMode.CONTRADICTION,
            "Work package batch cannot contain duplicate package ids.",
        )

        known_package_ids = set(package_ids)
        for dependency in self.dependencies:
            require_invariant(
                dependency.required_before in known_package_ids
                and dependency.required_after in known_package_ids,
                FailureMode.UNSUPPORTED_CLAIM,
                "Work package batch dependencies must reference known packages.",
            )

    @property
    def packages_requiring_human_review(self) -> tuple[CognitiveWorkPackage, ...]:
        """Return packages that require human review."""

        return tuple(package for package in self.packages if package.requires_human_review)

    @property
    def action_adjacent_packages(self) -> tuple[CognitiveWorkPackage, ...]:
        """Return packages that are close to action or promotion."""

        return tuple(package for package in self.packages if package.is_action_adjacent)


def _kind_from_plan_node(kind: PlanNodeKind) -> WorkPackageKind:
    mapping = {
        PlanNodeKind.RESEARCH: WorkPackageKind.RESEARCH,
        PlanNodeKind.DESIGN: WorkPackageKind.DESIGN,
        PlanNodeKind.IMPLEMENTATION: WorkPackageKind.IMPLEMENTATION,
        PlanNodeKind.TEST: WorkPackageKind.TEST,
        PlanNodeKind.REVIEW: WorkPackageKind.REVIEW,
        PlanNodeKind.SECURITY_REVIEW: WorkPackageKind.SECURITY_REVIEW,
        PlanNodeKind.POLICY_REVIEW: WorkPackageKind.POLICY_REVIEW,
        PlanNodeKind.MEMORY_REVIEW: WorkPackageKind.MEMORY_REVIEW,
        PlanNodeKind.EVIDENCE_REVIEW: WorkPackageKind.EVIDENCE_REVIEW,
        PlanNodeKind.DOCUMENTATION: WorkPackageKind.DOCUMENTATION,
        PlanNodeKind.SELF_IMPROVEMENT_PROPOSAL: WorkPackageKind.SELF_IMPROVEMENT_PROPOSAL,
        PlanNodeKind.BLACKFOX_HANDOFF: WorkPackageKind.BLACKFOX_HANDOFF,
    }
    return mapping[kind]
