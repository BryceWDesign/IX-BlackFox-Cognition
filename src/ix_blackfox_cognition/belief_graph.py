"""Belief graph models for IX-BlackFox-Cognition.

The belief graph is the inspectable world-model layer for governed cognition.
It records what the system thinks it knows, what remains unsupported, what is
stale, what is contradicted, and what depends on evidence.

The belief graph does not make model output true. It only structures cognition
so claims, assumptions, contradictions, and evidence dependencies remain visible
before planning or action handoff.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import FailureMode, require_invariant


class BeliefKind(StrEnum):
    """Kinds of beliefs tracked by the cognition world model."""

    FACT = "fact"
    ASSUMPTION = "assumption"
    INFERENCE = "inference"
    REQUIREMENT = "requirement"
    RISK = "risk"
    DECISION = "decision"
    POLICY_FACT = "policy_fact"
    MEMORY_FACT = "memory_fact"
    REPO_FACT = "repo_fact"
    WORLD_FACT = "world_fact"
    UNKNOWN = "unknown"


class BeliefStatus(StrEnum):
    """Lifecycle status for a belief node."""

    PROPOSED = "proposed"
    UNSUPPORTED = "unsupported"
    EVIDENCE_REQUIRED = "evidence_required"
    SUPPORTED = "supported"
    VERIFIED = "verified"
    HUMAN_APPROVED = "human_approved"
    CONTRADICTED = "contradicted"
    REJECTED = "rejected"
    STALE = "stale"
    QUARANTINED = "quarantined"


class BeliefRelationKind(StrEnum):
    """Kinds of edges between belief nodes."""

    SUPPORTS = "supports"
    DEPENDS_ON = "depends_on"
    CONTRADICTS = "contradicts"
    SUPERSEDES = "supersedes"
    REFINES = "refines"
    DERIVED_FROM = "derived_from"
    BLOCKS = "blocks"
    REQUIRES_EVIDENCE = "requires_evidence"
    SAME_AS = "same_as"


@dataclass(frozen=True, slots=True)
class BeliefNode:
    """A single inspectable belief inside the cognition world model."""

    belief_id: str
    kind: BeliefKind
    statement: str
    status: BeliefStatus = BeliefStatus.PROPOSED
    source_claim_id: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    memory_record_ids: tuple[str, ...] = field(default_factory=tuple)
    uncertainty: str | None = None
    human_approval_id: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.belief_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Belief id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Belief statement cannot be blank.",
        )

        if self.status == BeliefStatus.VERIFIED:
            require_invariant(
                bool(self.evidence_ids),
                FailureMode.MISSING_EVIDENCE,
                "Verified beliefs require at least one evidence id.",
            )

        if self.status == BeliefStatus.HUMAN_APPROVED:
            require_invariant(
                self.human_approval_id is not None and bool(self.human_approval_id.strip()),
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "Human-approved beliefs require a human approval id.",
            )

        if self.status in (BeliefStatus.UNSUPPORTED, BeliefStatus.EVIDENCE_REQUIRED):
            require_invariant(
                not self.evidence_ids,
                FailureMode.MODEL_CONFIDENCE_AS_EVIDENCE,
                "Unsupported beliefs cannot carry evidence ids as if they were trusted.",
            )

    @property
    def has_evidence(self) -> bool:
        """Return whether the belief has evidence identifiers attached."""

        return bool(self.evidence_ids)

    @property
    def has_human_approval(self) -> bool:
        """Return whether the belief has explicit human approval."""

        return self.human_approval_id is not None and bool(self.human_approval_id.strip())

    @property
    def requires_evidence(self) -> bool:
        """Return whether the belief needs evidence before bounded trust."""

        return self.status in (
            BeliefStatus.PROPOSED,
            BeliefStatus.UNSUPPORTED,
            BeliefStatus.EVIDENCE_REQUIRED,
            BeliefStatus.QUARANTINED,
        )

    @property
    def trust_eligible(self) -> bool:
        """Return whether the belief is eligible for bounded trust."""

        return self.status in (
            BeliefStatus.SUPPORTED,
            BeliefStatus.VERIFIED,
            BeliefStatus.HUMAN_APPROVED,
        )

    @property
    def blocked_by_status(self) -> bool:
        """Return whether the belief status blocks promotion into trusted planning."""

        return self.status in (
            BeliefStatus.CONTRADICTED,
            BeliefStatus.REJECTED,
            BeliefStatus.STALE,
            BeliefStatus.QUARANTINED,
        )


@dataclass(frozen=True, slots=True)
class BeliefEdge:
    """A directed relationship between two belief nodes."""

    edge_id: str
    source_belief_id: str
    target_belief_id: str
    relation: BeliefRelationKind
    statement: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.edge_id.strip()),
            FailureMode.CONTRADICTION,
            "Belief edge id cannot be blank.",
        )
        require_invariant(
            bool(self.source_belief_id.strip()),
            FailureMode.CONTRADICTION,
            "Belief edge source id cannot be blank.",
        )
        require_invariant(
            bool(self.target_belief_id.strip()),
            FailureMode.CONTRADICTION,
            "Belief edge target id cannot be blank.",
        )
        require_invariant(
            self.source_belief_id != self.target_belief_id,
            FailureMode.CONTRADICTION,
            "Belief edge cannot point a belief at itself.",
        )


@dataclass(frozen=True, slots=True)
class BeliefContradiction:
    """A contradiction between belief nodes that must remain inspectable."""

    contradiction_id: str
    belief_ids: tuple[str, ...]
    statement: str
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    requires_review: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.contradiction_id.strip()),
            FailureMode.CONTRADICTION,
            "Belief contradiction id cannot be blank.",
        )
        require_invariant(
            len(self.belief_ids) >= 2,
            FailureMode.CONTRADICTION,
            "Belief contradiction must reference at least two beliefs.",
        )
        require_invariant(
            len(self.belief_ids) == len(set(self.belief_ids)),
            FailureMode.CONTRADICTION,
            "Belief contradiction cannot reference the same belief more than once.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.CONTRADICTION,
            "Belief contradiction statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class StaleBeliefMarker:
    """Marker showing that a belief is stale and must not be used as current truth."""

    marker_id: str
    belief_id: str
    reason: str
    superseded_by: str | None = None
    requires_review: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.marker_id.strip()),
            FailureMode.STALE_MEMORY,
            "Stale belief marker id cannot be blank.",
        )
        require_invariant(
            bool(self.belief_id.strip()),
            FailureMode.STALE_MEMORY,
            "Stale belief marker belief id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.STALE_MEMORY,
            "Stale belief marker reason cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class BeliefGraph:
    """Immutable graph of beliefs, relationships, contradictions, and staleness."""

    graph_id: str
    nodes: tuple[BeliefNode, ...] = field(default_factory=tuple)
    edges: tuple[BeliefEdge, ...] = field(default_factory=tuple)
    contradictions: tuple[BeliefContradiction, ...] = field(default_factory=tuple)
    stale_markers: tuple[StaleBeliefMarker, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.graph_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Belief graph id cannot be blank.",
        )

        node_ids = [node.belief_id for node in self.nodes]
        require_invariant(
            len(node_ids) == len(set(node_ids)),
            FailureMode.CONTRADICTION,
            "Belief graph cannot contain duplicate belief ids.",
        )

        edge_ids = [edge.edge_id for edge in self.edges]
        require_invariant(
            len(edge_ids) == len(set(edge_ids)),
            FailureMode.CONTRADICTION,
            "Belief graph cannot contain duplicate edge ids.",
        )

        known_ids = set(node_ids)
        for edge in self.edges:
            require_invariant(
                edge.source_belief_id in known_ids and edge.target_belief_id in known_ids,
                FailureMode.UNSUPPORTED_CLAIM,
                "Belief graph edge must reference known belief nodes.",
            )

        for contradiction in self.contradictions:
            require_invariant(
                all(belief_id in known_ids for belief_id in contradiction.belief_ids),
                FailureMode.CONTRADICTION,
                "Belief contradiction must reference known belief nodes.",
            )

        for marker in self.stale_markers:
            require_invariant(
                marker.belief_id in known_ids,
                FailureMode.STALE_MEMORY,
                "Stale belief marker must reference a known belief node.",
            )

    def node_by_id(self, belief_id: str) -> BeliefNode | None:
        """Return a belief node by id, if present."""

        for node in self.nodes:
            if node.belief_id == belief_id:
                return node
        return None

    @property
    def unsupported_nodes(self) -> tuple[BeliefNode, ...]:
        """Return belief nodes that still require evidence."""

        return tuple(node for node in self.nodes if node.requires_evidence)

    @property
    def trust_eligible_nodes(self) -> tuple[BeliefNode, ...]:
        """Return belief nodes eligible for bounded trust."""

        return tuple(node for node in self.nodes if node.trust_eligible)

    @property
    def blocked_nodes(self) -> tuple[BeliefNode, ...]:
        """Return belief nodes that are blocked by contradiction, rejection, or staleness."""

        return tuple(node for node in self.nodes if node.blocked_by_status)
