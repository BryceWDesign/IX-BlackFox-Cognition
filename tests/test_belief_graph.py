"""Tests for belief graph models and immutable belief graph logic."""

import pytest

from ix_blackfox_cognition import (
    BeliefContradiction,
    BeliefDecisionKind,
    BeliefEdge,
    BeliefGraph,
    BeliefGraphEngine,
    BeliefKind,
    BeliefNode,
    BeliefRelationKind,
    BeliefStatus,
    CognitionInvariantError,
    DecisionOutcome,
    FailureMode,
    StaleBeliefMarker,
    promote_belief,
)


def _belief(
    belief_id: str = "belief:test",
    *,
    status: BeliefStatus = BeliefStatus.PROPOSED,
    evidence_ids: tuple[str, ...] = (),
    human_approval_id: str | None = None,
) -> BeliefNode:
    return BeliefNode(
        belief_id=belief_id,
        kind=BeliefKind.FACT,
        statement=f"Statement for {belief_id}.",
        status=status,
        evidence_ids=evidence_ids,
        human_approval_id=human_approval_id,
    )


def test_verified_belief_requires_evidence_ids() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _belief(status=BeliefStatus.VERIFIED)

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_human_approved_belief_requires_human_approval_id() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _belief(status=BeliefStatus.HUMAN_APPROVED)

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_unsupported_belief_cannot_carry_evidence_as_trust_signal() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _belief(
            status=BeliefStatus.UNSUPPORTED,
            evidence_ids=("evidence:model-confidence",),
        )

    assert exc_info.value.failure.mode == FailureMode.MODEL_CONFIDENCE_AS_EVIDENCE


def test_belief_node_reports_evidence_trust_and_blocked_status() -> None:
    proposed = _belief(status=BeliefStatus.PROPOSED)
    verified = _belief(
        belief_id="belief:verified",
        status=BeliefStatus.VERIFIED,
        evidence_ids=("evidence:test",),
    )
    stale = _belief(belief_id="belief:stale", status=BeliefStatus.STALE)

    assert proposed.requires_evidence is True
    assert proposed.trust_eligible is False
    assert verified.has_evidence is True
    assert verified.trust_eligible is True
    assert stale.blocked_by_status is True


def test_belief_edge_cannot_point_to_same_belief() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        BeliefEdge(
            edge_id="edge:self",
            source_belief_id="belief:same",
            target_belief_id="belief:same",
            relation=BeliefRelationKind.SUPPORTS,
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_belief_contradiction_requires_at_least_two_unique_beliefs() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        BeliefContradiction(
            contradiction_id="contradiction:bad",
            belief_ids=("belief:one", "belief:one"),
            statement="A contradiction cannot repeat the same belief.",
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_belief_graph_rejects_duplicate_belief_ids() -> None:
    belief = _belief("belief:duplicate")

    with pytest.raises(CognitionInvariantError) as exc_info:
        BeliefGraph(
            graph_id="graph:duplicate",
            nodes=(belief, belief),
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_belief_graph_rejects_edge_referencing_unknown_node() -> None:
    known = _belief("belief:known")
    edge = BeliefEdge(
        edge_id="edge:unknown",
        source_belief_id="belief:known",
        target_belief_id="belief:missing",
        relation=BeliefRelationKind.DEPENDS_ON,
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        BeliefGraph(
            graph_id="graph:unknown-edge",
            nodes=(known,),
            edges=(edge,),
        )

    assert exc_info.value.failure.mode == FailureMode.UNSUPPORTED_CLAIM


def test_belief_graph_rejects_stale_marker_for_unknown_node() -> None:
    marker = StaleBeliefMarker(
        marker_id="stale:unknown",
        belief_id="belief:missing",
        reason="The belief is not present in the graph.",
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        BeliefGraph(
            graph_id="graph:unknown-stale-marker",
            stale_markers=(marker,),
        )

    assert exc_info.value.failure.mode == FailureMode.STALE_MEMORY


def test_belief_graph_reports_unsupported_trust_eligible_and_blocked_nodes() -> None:
    unsupported = _belief("belief:unsupported", status=BeliefStatus.EVIDENCE_REQUIRED)
    verified = _belief(
        "belief:verified",
        status=BeliefStatus.VERIFIED,
        evidence_ids=("evidence:verified",),
    )
    rejected = _belief("belief:rejected", status=BeliefStatus.REJECTED)
    graph = BeliefGraph(
        graph_id="graph:status",
        nodes=(unsupported, verified, rejected),
    )

    assert graph.unsupported_nodes == (unsupported,)
    assert graph.trust_eligible_nodes == (verified,)
    assert graph.blocked_nodes == (rejected,)


def test_add_node_returns_new_graph_without_mutating_original() -> None:
    graph = BeliefGraph(graph_id="graph:add-node")
    node = _belief("belief:new")

    result = BeliefGraphEngine().add_node(graph, node)

    assert result.decision.allowed
    assert result.decision.kind == BeliefDecisionKind.ADD_NODE
    assert graph.nodes == ()
    assert result.graph.nodes == (node,)


def test_add_duplicate_node_fails_closed() -> None:
    node = _belief("belief:duplicate")
    graph = BeliefGraph(graph_id="graph:add-duplicate", nodes=(node,))

    result = BeliefGraphEngine().add_node(graph, node)

    assert result.decision.failed_closed
    assert result.decision.outcome == DecisionOutcome.FAIL_CLOSED
    assert result.decision.failure_modes == (FailureMode.CONTRADICTION,)
    assert result.graph == graph


def test_add_edge_between_known_nodes_returns_new_graph() -> None:
    source = _belief("belief:source")
    target = _belief("belief:target")
    graph = BeliefGraph(graph_id="graph:add-edge", nodes=(source, target))
    edge = BeliefEdge(
        edge_id="edge:source-target",
        source_belief_id="belief:source",
        target_belief_id="belief:target",
        relation=BeliefRelationKind.SUPPORTS,
    )

    result = BeliefGraphEngine().add_edge(graph, edge)

    assert result.decision.allowed
    assert graph.edges == ()
    assert result.graph.edges == (edge,)


def test_add_edge_to_unknown_target_fails_closed() -> None:
    source = _belief("belief:source")
    graph = BeliefGraph(graph_id="graph:unknown-target", nodes=(source,))
    edge = BeliefEdge(
        edge_id="edge:unknown-target",
        source_belief_id="belief:source",
        target_belief_id="belief:missing",
        relation=BeliefRelationKind.SUPPORTS,
    )

    result = BeliefGraphEngine().add_edge(graph, edge)

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.UNSUPPORTED_CLAIM,)
    assert result.graph == graph


def test_promote_belief_to_verified_requires_evidence_ids() -> None:
    node = _belief("belief:needs-evidence")
    graph = BeliefGraph(graph_id="graph:promote-missing-evidence", nodes=(node,))

    result = promote_belief(
        graph,
        "belief:needs-evidence",
        BeliefStatus.VERIFIED,
        reason="Attempt promotion without evidence.",
    )

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.MISSING_EVIDENCE,)
    assert result.graph.node_by_id("belief:needs-evidence").status == BeliefStatus.PROPOSED


def test_promote_belief_to_verified_with_evidence_returns_new_graph() -> None:
    node = _belief("belief:verified")
    graph = BeliefGraph(graph_id="graph:promote-verified", nodes=(node,))

    result = promote_belief(
        graph,
        "belief:verified",
        BeliefStatus.VERIFIED,
        reason="Verified by evidence reference.",
        evidence_ids=("evidence:test-result",),
    )

    original_node = graph.node_by_id("belief:verified")
    updated_node = result.graph.node_by_id("belief:verified")

    assert result.decision.allowed
    assert original_node.status == BeliefStatus.PROPOSED
    assert updated_node.status == BeliefStatus.VERIFIED
    assert updated_node.evidence_ids == ("evidence:test-result",)


def test_promote_belief_to_human_approved_requires_human_approval_id() -> None:
    node = _belief("belief:human-approval")
    graph = BeliefGraph(graph_id="graph:human-approval", nodes=(node,))

    result = promote_belief(
        graph,
        "belief:human-approval",
        BeliefStatus.HUMAN_APPROVED,
        reason="Attempt approval without human approval id.",
    )

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.HUMAN_REVIEW_REQUIRED,)


def test_record_contradiction_marks_beliefs_and_adds_contradiction_edges() -> None:
    first = _belief("belief:first")
    second = _belief("belief:second")
    graph = BeliefGraph(graph_id="graph:contradiction", nodes=(first, second))
    contradiction = BeliefContradiction(
        contradiction_id="contradiction:first-second",
        belief_ids=("belief:first", "belief:second"),
        statement="The beliefs cannot both be true.",
        evidence_ids=("evidence:contradiction",),
    )

    result = BeliefGraphEngine().record_contradiction(graph, contradiction)

    assert result.decision.allowed
    assert result.graph.node_by_id("belief:first").status == BeliefStatus.CONTRADICTED
    assert result.graph.node_by_id("belief:second").status == BeliefStatus.CONTRADICTED
    assert result.graph.contradictions == (contradiction,)
    assert len(result.graph.edges) == 2
    assert all(edge.relation == BeliefRelationKind.CONTRADICTS for edge in result.graph.edges)


def test_contradicted_belief_cannot_be_promoted_until_reviewed() -> None:
    first = _belief("belief:first")
    second = _belief("belief:second")
    graph = BeliefGraph(graph_id="graph:blocked-contradiction", nodes=(first, second))
    contradiction_result = BeliefGraphEngine().record_contradiction(
        graph,
        BeliefContradiction(
            contradiction_id="contradiction:block",
            belief_ids=("belief:first", "belief:second"),
            statement="Contradiction blocks promotion.",
        ),
    )

    result = promote_belief(
        contradiction_result.graph,
        "belief:first",
        BeliefStatus.VERIFIED,
        reason="Attempt promotion despite contradiction.",
        evidence_ids=("evidence:test",),
    )

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.CONTRADICTION,)


def test_mark_stale_blocks_belief_from_trusted_promotion() -> None:
    node = _belief("belief:stale-target")
    graph = BeliefGraph(graph_id="graph:stale", nodes=(node,))
    marker = StaleBeliefMarker(
        marker_id="stale:target",
        belief_id="belief:stale-target",
        reason="A newer belief superseded this one.",
        superseded_by="belief:newer",
    )

    stale_result = BeliefGraphEngine().mark_stale(graph, marker)
    promotion_result = promote_belief(
        stale_result.graph,
        "belief:stale-target",
        BeliefStatus.VERIFIED,
        reason="Attempt promotion despite staleness.",
        evidence_ids=("evidence:test",),
    )

    assert stale_result.decision.allowed
    assert stale_result.graph.node_by_id("belief:stale-target").status == BeliefStatus.STALE
    assert stale_result.graph.stale_markers == (marker,)
    assert promotion_result.decision.failed_closed
    assert promotion_result.decision.failure_modes == (FailureMode.STALE_MEMORY,)


def test_reject_and_quarantine_belief_are_immutable_operations() -> None:
    node = _belief("belief:review-target")
    graph = BeliefGraph(graph_id="graph:reject-quarantine", nodes=(node,))
    engine = BeliefGraphEngine()

    rejected = engine.reject_belief(
        graph,
        "belief:review-target",
        reason="The belief was rejected during review.",
    )
    quarantined = engine.quarantine_belief(
        graph,
        "belief:review-target",
        reason="The belief needs further evidence review.",
    )

    assert rejected.decision.allowed
    assert quarantined.decision.allowed
    assert graph.node_by_id("belief:review-target").status == BeliefStatus.PROPOSED
    assert rejected.graph.node_by_id("belief:review-target").status == BeliefStatus.REJECTED
    assert quarantined.graph.node_by_id("belief:review-target").status == BeliefStatus.QUARANTINED
