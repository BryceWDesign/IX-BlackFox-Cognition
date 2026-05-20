"""Belief graph logic for IX-BlackFox-Cognition.

The belief graph is immutable. Every operation returns a new graph plus a
reviewable decision record. This prevents silent state mutation while preserving
an inspectable trail of belief promotion, rejection, contradiction, and staleness
handling.

Core invariant:

Unsupported claims cannot become trusted beliefs without evidence or explicit
human approval, and contradicted or stale beliefs cannot be silently promoted.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from ix_blackfox_cognition.belief_graph import (
    BeliefContradiction,
    BeliefEdge,
    BeliefGraph,
    BeliefNode,
    BeliefRelationKind,
    BeliefStatus,
    StaleBeliefMarker,
)
from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, require_invariant


class BeliefDecisionKind(StrEnum):
    """Kinds of belief graph decisions."""

    ADD_NODE = "add_node"
    ADD_EDGE = "add_edge"
    PROMOTE = "promote"
    REJECT = "reject"
    QUARANTINE = "quarantine"
    RECORD_CONTRADICTION = "record_contradiction"
    MARK_STALE = "mark_stale"


@dataclass(frozen=True, slots=True)
class BeliefGraphDecision:
    """Reviewable decision record for a belief graph operation."""

    decision_id: str
    graph_id: str
    belief_id: str | None
    kind: BeliefDecisionKind
    outcome: DecisionOutcome
    reason: str
    failure_modes: tuple[FailureMode, ...] = ()

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.decision_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Belief graph decision id cannot be blank.",
        )
        require_invariant(
            bool(self.graph_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Belief graph decision graph id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Belief graph decision reason cannot be blank.",
        )

    @property
    def allowed(self) -> bool:
        """Return whether the belief graph operation was allowed."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def failed_closed(self) -> bool:
        """Return whether the belief graph operation failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED


@dataclass(frozen=True, slots=True)
class BeliefGraphOperationResult:
    """Result of a belief graph operation."""

    graph: BeliefGraph
    decision: BeliefGraphDecision


class BeliefGraphEngine:
    """Immutable belief graph operation engine."""

    def add_node(self, graph: BeliefGraph, node: BeliefNode) -> BeliefGraphOperationResult:
        """Return a graph containing a new unique belief node."""

        if graph.node_by_id(node.belief_id) is not None:
            return self._fail_closed(
                graph=graph,
                belief_id=node.belief_id,
                kind=BeliefDecisionKind.ADD_NODE,
                reason="Belief graph cannot add a duplicate belief node.",
                failure_modes=(FailureMode.CONTRADICTION,),
            )

        updated_graph = replace(graph, nodes=(*graph.nodes, node))
        return BeliefGraphOperationResult(
            graph=updated_graph,
            decision=self._allowed(
                graph=updated_graph,
                belief_id=node.belief_id,
                kind=BeliefDecisionKind.ADD_NODE,
                reason="Belief node was added without mutating the original graph.",
            ),
        )

    def add_edge(self, graph: BeliefGraph, edge: BeliefEdge) -> BeliefGraphOperationResult:
        """Return a graph containing a new unique edge between known beliefs."""

        existing_edge_ids = {existing.edge_id for existing in graph.edges}
        if edge.edge_id in existing_edge_ids:
            return self._fail_closed(
                graph=graph,
                belief_id=edge.source_belief_id,
                kind=BeliefDecisionKind.ADD_EDGE,
                reason="Belief graph cannot add a duplicate edge id.",
                failure_modes=(FailureMode.CONTRADICTION,),
            )

        if graph.node_by_id(edge.source_belief_id) is None:
            return self._fail_closed(
                graph=graph,
                belief_id=edge.source_belief_id,
                kind=BeliefDecisionKind.ADD_EDGE,
                reason="Belief edge source node is unknown.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if graph.node_by_id(edge.target_belief_id) is None:
            return self._fail_closed(
                graph=graph,
                belief_id=edge.target_belief_id,
                kind=BeliefDecisionKind.ADD_EDGE,
                reason="Belief edge target node is unknown.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        updated_graph = replace(graph, edges=(*graph.edges, edge))
        return BeliefGraphOperationResult(
            graph=updated_graph,
            decision=self._allowed(
                graph=updated_graph,
                belief_id=edge.source_belief_id,
                kind=BeliefDecisionKind.ADD_EDGE,
                reason="Belief edge was added without mutating the original graph.",
            ),
        )

    def promote_belief(
        self,
        graph: BeliefGraph,
        belief_id: str,
        target_status: BeliefStatus,
        *,
        reason: str,
        evidence_ids: tuple[str, ...] = (),
        human_approval_id: str | None = None,
    ) -> BeliefGraphOperationResult:
        """Promote a belief only when evidence and contradiction checks allow it."""

        node = graph.node_by_id(belief_id)
        if node is None:
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.PROMOTE,
                reason="Cannot promote an unknown belief.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if not reason.strip():
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.PROMOTE,
                reason="Belief promotion reason cannot be blank.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if target_status not in (
            BeliefStatus.SUPPORTED,
            BeliefStatus.VERIFIED,
            BeliefStatus.HUMAN_APPROVED,
        ):
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.PROMOTE,
                reason="Belief promotion target must be supported, verified, or human-approved.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if self._belief_has_contradiction(graph, belief_id):
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.PROMOTE,
                reason="Contradicted beliefs cannot be promoted until contradiction is reviewed.",
                failure_modes=(FailureMode.CONTRADICTION,),
            )

        if self._belief_is_stale(graph, belief_id):
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.PROMOTE,
                reason="Stale beliefs cannot be promoted until staleness is reviewed.",
                failure_modes=(FailureMode.STALE_MEMORY,),
            )

        if node.blocked_by_status:
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.PROMOTE,
                reason="Belief status blocks promotion into trusted planning.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if target_status in (BeliefStatus.SUPPORTED, BeliefStatus.VERIFIED):
            if not evidence_ids:
                return self._fail_closed(
                    graph=graph,
                    belief_id=belief_id,
                    kind=BeliefDecisionKind.PROMOTE,
                    reason="Evidence ids are required before belief promotion.",
                    failure_modes=(FailureMode.MISSING_EVIDENCE,),
                )

        if target_status == BeliefStatus.HUMAN_APPROVED:
            if human_approval_id is None or not human_approval_id.strip():
                return self._fail_closed(
                    graph=graph,
                    belief_id=belief_id,
                    kind=BeliefDecisionKind.PROMOTE,
                    reason="Human-approved belief promotion requires a human approval id.",
                    failure_modes=(FailureMode.HUMAN_REVIEW_REQUIRED,),
                )

        updated_node = replace(
            node,
            status=target_status,
            evidence_ids=evidence_ids or node.evidence_ids,
            human_approval_id=human_approval_id or node.human_approval_id,
        )
        updated_graph = self._replace_node(graph, updated_node)

        return BeliefGraphOperationResult(
            graph=updated_graph,
            decision=self._allowed(
                graph=updated_graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.PROMOTE,
                reason=reason,
            ),
        )

    def reject_belief(
        self,
        graph: BeliefGraph,
        belief_id: str,
        *,
        reason: str,
    ) -> BeliefGraphOperationResult:
        """Reject a belief while preserving an immutable graph update."""

        node = graph.node_by_id(belief_id)
        if node is None:
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.REJECT,
                reason="Cannot reject an unknown belief.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if not reason.strip():
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.REJECT,
                reason="Belief rejection reason cannot be blank.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        updated_graph = self._replace_node(graph, replace(node, status=BeliefStatus.REJECTED))
        return BeliefGraphOperationResult(
            graph=updated_graph,
            decision=self._allowed(
                graph=updated_graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.REJECT,
                reason=reason,
            ),
        )

    def quarantine_belief(
        self,
        graph: BeliefGraph,
        belief_id: str,
        *,
        reason: str,
    ) -> BeliefGraphOperationResult:
        """Quarantine a belief that needs review before use."""

        node = graph.node_by_id(belief_id)
        if node is None:
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.QUARANTINE,
                reason="Cannot quarantine an unknown belief.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if not reason.strip():
            return self._fail_closed(
                graph=graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.QUARANTINE,
                reason="Belief quarantine reason cannot be blank.",
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        updated_graph = self._replace_node(graph, replace(node, status=BeliefStatus.QUARANTINED))
        return BeliefGraphOperationResult(
            graph=updated_graph,
            decision=self._allowed(
                graph=updated_graph,
                belief_id=belief_id,
                kind=BeliefDecisionKind.QUARANTINE,
                reason=reason,
            ),
        )

    def record_contradiction(
        self,
        graph: BeliefGraph,
        contradiction: BeliefContradiction,
    ) -> BeliefGraphOperationResult:
        """Record a contradiction and mark referenced beliefs as contradicted."""

        existing_ids = {existing.contradiction_id for existing in graph.contradictions}
        if contradiction.contradiction_id in existing_ids:
            return self._fail_closed(
                graph=graph,
                belief_id=None,
                kind=BeliefDecisionKind.RECORD_CONTRADICTION,
                reason="Belief graph cannot add a duplicate contradiction id.",
                failure_modes=(FailureMode.CONTRADICTION,),
            )

        unknown_ids = [
            belief_id
            for belief_id in contradiction.belief_ids
            if graph.node_by_id(belief_id) is None
        ]
        if unknown_ids:
            return self._fail_closed(
                graph=graph,
                belief_id=unknown_ids[0],
                kind=BeliefDecisionKind.RECORD_CONTRADICTION,
                reason="Belief contradiction references an unknown belief.",
                failure_modes=(FailureMode.CONTRADICTION,),
            )

        updated_nodes = tuple(
            replace(node, status=BeliefStatus.CONTRADICTED)
            if node.belief_id in contradiction.belief_ids
            else node
            for node in graph.nodes
        )
        updated_edges = (*graph.edges, *self._contradiction_edges(graph, contradiction))
        updated_graph = replace(
            graph,
            nodes=updated_nodes,
            edges=updated_edges,
            contradictions=(*graph.contradictions, contradiction),
        )

        return BeliefGraphOperationResult(
            graph=updated_graph,
            decision=self._allowed(
                graph=updated_graph,
                belief_id=None,
                kind=BeliefDecisionKind.RECORD_CONTRADICTION,
                reason=contradiction.statement,
            ),
        )

    def mark_stale(
        self,
        graph: BeliefGraph,
        marker: StaleBeliefMarker,
    ) -> BeliefGraphOperationResult:
        """Mark a belief stale and block it from trusted use."""

        node = graph.node_by_id(marker.belief_id)
        if node is None:
            return self._fail_closed(
                graph=graph,
                belief_id=marker.belief_id,
                kind=BeliefDecisionKind.MARK_STALE,
                reason="Cannot mark an unknown belief as stale.",
                failure_modes=(FailureMode.STALE_MEMORY,),
            )

        existing_ids = {existing.marker_id for existing in graph.stale_markers}
        if marker.marker_id in existing_ids:
            return self._fail_closed(
                graph=graph,
                belief_id=marker.belief_id,
                kind=BeliefDecisionKind.MARK_STALE,
                reason="Belief graph cannot add a duplicate stale marker id.",
                failure_modes=(FailureMode.STALE_MEMORY,),
            )

        updated_node = replace(node, status=BeliefStatus.STALE)
        updated_graph = replace(
            self._replace_node(graph, updated_node),
            stale_markers=(*graph.stale_markers, marker),
        )

        return BeliefGraphOperationResult(
            graph=updated_graph,
            decision=self._allowed(
                graph=updated_graph,
                belief_id=marker.belief_id,
                kind=BeliefDecisionKind.MARK_STALE,
                reason=marker.reason,
            ),
        )

    def _replace_node(self, graph: BeliefGraph, updated_node: BeliefNode) -> BeliefGraph:
        return replace(
            graph,
            nodes=tuple(
                updated_node if node.belief_id == updated_node.belief_id else node
                for node in graph.nodes
            ),
        )

    def _belief_has_contradiction(self, graph: BeliefGraph, belief_id: str) -> bool:
        return any(belief_id in contradiction.belief_ids for contradiction in graph.contradictions)

    def _belief_is_stale(self, graph: BeliefGraph, belief_id: str) -> bool:
        return any(marker.belief_id == belief_id for marker in graph.stale_markers)

    def _contradiction_edges(
        self,
        graph: BeliefGraph,
        contradiction: BeliefContradiction,
    ) -> tuple[BeliefEdge, ...]:
        existing_edge_ids = {edge.edge_id for edge in graph.edges}
        edges: list[BeliefEdge] = []

        for source_id in contradiction.belief_ids:
            for target_id in contradiction.belief_ids:
                if source_id == target_id:
                    continue

                edge_id = f"edge:{contradiction.contradiction_id}:{source_id}->{target_id}"
                if edge_id in existing_edge_ids:
                    continue

                edges.append(
                    BeliefEdge(
                        edge_id=edge_id,
                        source_belief_id=source_id,
                        target_belief_id=target_id,
                        relation=BeliefRelationKind.CONTRADICTS,
                        statement=contradiction.statement,
                        evidence_ids=contradiction.evidence_ids,
                    )
                )

        return tuple(edges)

    def _allowed(
        self,
        graph: BeliefGraph,
        belief_id: str | None,
        kind: BeliefDecisionKind,
        reason: str,
    ) -> BeliefGraphDecision:
        return BeliefGraphDecision(
            decision_id=f"belief-decision:{graph.graph_id}:{kind.value}:{belief_id or 'graph'}",
            graph_id=graph.graph_id,
            belief_id=belief_id,
            kind=kind,
            outcome=DecisionOutcome.ALLOW,
            reason=reason,
        )

    def _fail_closed(
        self,
        graph: BeliefGraph,
        belief_id: str | None,
        kind: BeliefDecisionKind,
        reason: str,
        failure_modes: tuple[FailureMode, ...],
    ) -> BeliefGraphOperationResult:
        return BeliefGraphOperationResult(
            graph=graph,
            decision=BeliefGraphDecision(
                decision_id=f"belief-decision:{graph.graph_id}:{kind.value}:{belief_id or 'graph'}",
                graph_id=graph.graph_id,
                belief_id=belief_id,
                kind=kind,
                outcome=DecisionOutcome.FAIL_CLOSED,
                reason=reason,
                failure_modes=failure_modes,
            ),
        )


def promote_belief(
    graph: BeliefGraph,
    belief_id: str,
    target_status: BeliefStatus,
    *,
    reason: str,
    evidence_ids: tuple[str, ...] = (),
    human_approval_id: str | None = None,
) -> BeliefGraphOperationResult:
    """Promote a belief using the default immutable belief graph engine."""

    return BeliefGraphEngine().promote_belief(
        graph=graph,
        belief_id=belief_id,
        target_status=target_status,
        reason=reason,
        evidence_ids=evidence_ids,
        human_approval_id=human_approval_id,
    )
