"""Proof-carrying plan graph models for IX-BlackFox-Cognition.

Plans are not text blobs. A governed cognition plan must be inspectable before
action is eligible. Every meaningful plan node carries scope, risk, proof
obligations, evidence requirements, review requirements, dependencies, rollback
conditions, and falsification conditions.

This module defines the plan-graph data model only. Validation logic is
introduced in the next commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import FailureMode, RiskLevel, WorkState, require_invariant
from ix_blackfox_cognition.evidence import (
    EvidenceContract,
    FalsificationCondition,
    ProofObligation,
)


class PlanNodeKind(StrEnum):
    """Kinds of nodes in a proof-carrying plan graph."""

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
    SELF_IMPROVEMENT_PROPOSAL = "self_improvement_proposal"
    BLACKFOX_HANDOFF = "blackfox_handoff"


class PlanDependencyKind(StrEnum):
    """Kinds of dependency edges between plan nodes."""

    REQUIRES = "requires"
    BLOCKS = "blocks"
    ENABLES = "enables"
    VERIFIES = "verifies"
    REVIEWS = "reviews"
    FALSIFIES = "falsifies"
    ROLLS_BACK = "rolls_back"
    HANDS_OFF_TO = "hands_off_to"


class PlanReviewTrigger(StrEnum):
    """Triggers that force plan-level human review."""

    HIGH_RISK = "high_risk"
    CRITICAL_RISK = "critical_risk"
    BLACKFOX_HANDOFF = "blackfox_handoff"
    SELF_IMPROVEMENT = "self_improvement"
    MEMORY_PROMOTION = "memory_promotion"
    POLICY_CHANGE = "policy_change"
    CONTRADICTION = "contradiction"
    MISSING_EVIDENCE = "missing_evidence"
    OPERATOR_REQUEST = "operator_request"


@dataclass(frozen=True, slots=True)
class PlanReviewRequirement:
    """Human or model-role review required before a plan node can proceed."""

    requirement_id: str
    trigger: PlanReviewTrigger
    reviewer_role: str
    reason: str
    human_required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.requirement_id.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Plan review requirement id cannot be blank.",
        )
        require_invariant(
            bool(self.reviewer_role.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Plan review requirement reviewer role cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "Plan review requirement reason cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class PlanRollbackCondition:
    """Condition describing when a plan node must be reversed or abandoned."""

    rollback_id: str
    statement: str
    required_artifact: str | None = None
    human_review_required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.rollback_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Plan rollback condition id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Plan rollback condition statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class PlanNode:
    """A single proof-carrying node inside a governed plan graph."""

    node_id: str
    kind: PlanNodeKind
    objective: str
    status: WorkState = WorkState.DRAFT
    risk_level: RiskLevel = RiskLevel.LOW
    scope: str | None = None
    assumptions: tuple[str, ...] = field(default_factory=tuple)
    blockers: tuple[str, ...] = field(default_factory=tuple)
    required_belief_ids: tuple[str, ...] = field(default_factory=tuple)
    required_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    proof_obligations: tuple[ProofObligation, ...] = field(default_factory=tuple)
    evidence_contract: EvidenceContract | None = None
    falsification_conditions: tuple[FalsificationCondition, ...] = field(default_factory=tuple)
    review_requirements: tuple[PlanReviewRequirement, ...] = field(default_factory=tuple)
    rollback_conditions: tuple[PlanRollbackCondition, ...] = field(default_factory=tuple)
    output_contract: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.node_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Plan node id cannot be blank.",
        )
        require_invariant(
            bool(self.objective.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Plan node objective cannot be blank.",
        )

        if self.status in (
            WorkState.READY_FOR_REVIEW,
            WorkState.APPROVED,
            WorkState.COMPLETED,
        ):
            require_invariant(
                self.has_proof_obligations,
                FailureMode.MISSING_EVIDENCE,
                "Plan nodes past draft status require proof obligations.",
            )

        if self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            require_invariant(
                self.requires_review,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "High-risk and critical plan nodes require review requirements.",
            )

        if self.kind in (
            PlanNodeKind.BLACKFOX_HANDOFF,
            PlanNodeKind.SELF_IMPROVEMENT_PROPOSAL,
        ):
            require_invariant(
                self.requires_review,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "BlackFox handoff and self-improvement nodes require review requirements.",
            )
            require_invariant(
                bool(self.rollback_conditions),
                FailureMode.UNKNOWN_ACTION_TYPE,
                "BlackFox handoff and self-improvement nodes require rollback conditions.",
            )

    @property
    def has_proof_obligations(self) -> bool:
        """Return whether this node carries proof obligations."""

        return bool(self.proof_obligations)

    @property
    def has_evidence_contract(self) -> bool:
        """Return whether this node has a direct evidence contract."""

        return self.evidence_contract is not None

    @property
    def requires_review(self) -> bool:
        """Return whether this node has at least one review requirement."""

        return bool(self.review_requirements)

    @property
    def is_action_candidate(self) -> bool:
        """Return whether this node is close to action handoff."""

        return self.kind in (
            PlanNodeKind.IMPLEMENTATION,
            PlanNodeKind.TEST,
            PlanNodeKind.SELF_IMPROVEMENT_PROPOSAL,
            PlanNodeKind.BLACKFOX_HANDOFF,
        )

    @property
    def proof_requirement_ids(self) -> tuple[str, ...]:
        """Return all evidence requirement ids referenced by proof obligations."""

        ids: list[str] = []
        for obligation in self.proof_obligations:
            ids.extend(obligation.requirement_ids)
        return tuple(ids)


@dataclass(frozen=True, slots=True)
class PlanEdge:
    """A directed dependency relationship between two plan nodes."""

    edge_id: str
    source_node_id: str
    target_node_id: str
    dependency: PlanDependencyKind
    reason: str
    required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.edge_id.strip()),
            FailureMode.CONTRADICTION,
            "Plan edge id cannot be blank.",
        )
        require_invariant(
            bool(self.source_node_id.strip()),
            FailureMode.CONTRADICTION,
            "Plan edge source node id cannot be blank.",
        )
        require_invariant(
            bool(self.target_node_id.strip()),
            FailureMode.CONTRADICTION,
            "Plan edge target node id cannot be blank.",
        )
        require_invariant(
            self.source_node_id != self.target_node_id,
            FailureMode.CONTRADICTION,
            "Plan edge cannot point a plan node at itself.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.CONTRADICTION,
            "Plan edge reason cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class PlanGraph:
    """Immutable proof-carrying plan graph."""

    graph_id: str
    mission_id: str
    nodes: tuple[PlanNode, ...] = field(default_factory=tuple)
    edges: tuple[PlanEdge, ...] = field(default_factory=tuple)
    root_node_ids: tuple[str, ...] = field(default_factory=tuple)
    terminal_node_ids: tuple[str, ...] = field(default_factory=tuple)
    status: WorkState = WorkState.DRAFT

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.graph_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Plan graph id cannot be blank.",
        )
        require_invariant(
            bool(self.mission_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Plan graph mission id cannot be blank.",
        )

        node_ids = [node.node_id for node in self.nodes]
        require_invariant(
            len(node_ids) == len(set(node_ids)),
            FailureMode.CONTRADICTION,
            "Plan graph cannot contain duplicate node ids.",
        )

        edge_ids = [edge.edge_id for edge in self.edges]
        require_invariant(
            len(edge_ids) == len(set(edge_ids)),
            FailureMode.CONTRADICTION,
            "Plan graph cannot contain duplicate edge ids.",
        )

        known_node_ids = set(node_ids)
        for edge in self.edges:
            require_invariant(
                edge.source_node_id in known_node_ids and edge.target_node_id in known_node_ids,
                FailureMode.UNSUPPORTED_CLAIM,
                "Plan graph edges must reference known plan nodes.",
            )

        for root_node_id in self.root_node_ids:
            require_invariant(
                root_node_id in known_node_ids,
                FailureMode.UNSUPPORTED_CLAIM,
                "Plan graph root node ids must reference known plan nodes.",
            )

        for terminal_node_id in self.terminal_node_ids:
            require_invariant(
                terminal_node_id in known_node_ids,
                FailureMode.UNSUPPORTED_CLAIM,
                "Plan graph terminal node ids must reference known plan nodes.",
            )

        if self.status in (
            WorkState.READY_FOR_REVIEW,
            WorkState.APPROVED,
            WorkState.COMPLETED,
        ):
            require_invariant(
                bool(self.nodes),
                FailureMode.UNKNOWN_ACTION_TYPE,
                "Plan graphs past draft status must contain plan nodes.",
            )
            require_invariant(
                bool(self.root_node_ids),
                FailureMode.UNKNOWN_ACTION_TYPE,
                "Plan graphs past draft status must identify root nodes.",
            )
            require_invariant(
                bool(self.terminal_node_ids),
                FailureMode.UNKNOWN_ACTION_TYPE,
                "Plan graphs past draft status must identify terminal nodes.",
            )

    def node_by_id(self, node_id: str) -> PlanNode | None:
        """Return a plan node by id, if present."""

        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    @property
    def highest_risk(self) -> RiskLevel:
        """Return the highest node risk in the graph, or low risk for empty graphs."""

        if not self.nodes:
            return RiskLevel.LOW

        return max((node.risk_level for node in self.nodes), key=lambda level: level.rank())

    @property
    def nodes_requiring_review(self) -> tuple[PlanNode, ...]:
        """Return plan nodes that explicitly require review."""

        return tuple(node for node in self.nodes if node.requires_review)

    @property
    def action_candidate_nodes(self) -> tuple[PlanNode, ...]:
        """Return nodes that are close to action handoff."""

        return tuple(node for node in self.nodes if node.is_action_candidate)

    @property
    def nodes_missing_proof(self) -> tuple[PlanNode, ...]:
        """Return nodes missing proof obligations."""

        return tuple(node for node in self.nodes if not node.has_proof_obligations)
