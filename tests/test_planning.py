"""Tests for proof-carrying plan graphs and conservative plan validation."""

import pytest

from ix_blackfox_cognition import (
    ActorKind,
    CognitionInvariantError,
    DecisionOutcome,
    EvidenceContract,
    EvidenceKind,
    EvidenceReference,
    EvidenceRequirement,
    EvidenceSource,
    EvidenceState,
    EvidenceStrength,
    FailureMode,
    FalsificationCondition,
    PlanDependencyKind,
    PlanEdge,
    PlanGraph,
    PlanGraphValidator,
    PlanNode,
    PlanNodeKind,
    PlanReviewRequirement,
    PlanReviewTrigger,
    PlanRollbackCondition,
    PlanValidationPolicy,
    PlanValidationSeverity,
    ProofObligation,
    RiskLevel,
    WorkState,
    validate_plan_graph,
)


def _evidence_source() -> EvidenceSource:
    return EvidenceSource(
        actor_kind=ActorKind.SYSTEM,
        actor_id="system:test-runner",
        description="Test evidence source.",
    )


def _verified_evidence(
    evidence_id: str = "evidence:test-result",
    *,
    supports_claim_ids: tuple[str, ...] = (),
    contradicts_claim_ids: tuple[str, ...] = (),
) -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        kind=EvidenceKind.TEST_RESULT,
        state=EvidenceState.VERIFIED,
        source=_evidence_source(),
        summary="Verified test evidence.",
        strength=EvidenceStrength.STRONG,
        locator=f"artifacts/{evidence_id}.json",
        supports_claim_ids=supports_claim_ids,
        contradicts_claim_ids=contradicts_claim_ids,
    )


def _requirement(requirement_id: str = "requirement:test") -> EvidenceRequirement:
    return EvidenceRequirement(
        requirement_id=requirement_id,
        statement="Verified test evidence is required.",
        acceptable_kinds=(EvidenceKind.TEST_RESULT,),
        minimum_strength=EvidenceStrength.STRONG,
    )


def _proof_obligation(
    obligation_id: str = "obligation:test",
    *,
    subject_id: str = "plan-node:test",
    requirement_id: str = "requirement:test",
) -> ProofObligation:
    return ProofObligation(
        obligation_id=obligation_id,
        subject_id=subject_id,
        statement="The plan node must carry verified evidence before trust.",
        requirement_ids=(requirement_id,),
    )


def _satisfied_contract(
    contract_id: str = "contract:test",
    *,
    subject_id: str = "plan-node:test",
    requirement_id: str = "requirement:test",
    evidence_id: str = "evidence:test-result",
) -> EvidenceContract:
    return EvidenceContract(
        contract_id=contract_id,
        subject_id=subject_id,
        requirements=(_requirement(requirement_id),),
        references=(
            _verified_evidence(
                evidence_id,
                supports_claim_ids=(subject_id,),
            ),
        ),
    )


def _review_requirement(
    requirement_id: str = "review:test",
    *,
    trigger: PlanReviewTrigger = PlanReviewTrigger.HIGH_RISK,
) -> PlanReviewRequirement:
    return PlanReviewRequirement(
        requirement_id=requirement_id,
        trigger=trigger,
        reviewer_role="human-reviewer",
        reason="Human review is required before this plan node proceeds.",
    )


def _rollback_condition(rollback_id: str = "rollback:test") -> PlanRollbackCondition:
    return PlanRollbackCondition(
        rollback_id=rollback_id,
        statement="Rollback must restore the prior reviewed state.",
        required_artifact="rollback-plan.json",
    )


def _valid_research_node(node_id: str = "plan-node:research") -> PlanNode:
    return PlanNode(
        node_id=node_id,
        kind=PlanNodeKind.RESEARCH,
        objective="Research the bounded cognition requirement.",
        scope="Review only the explicitly scoped mission and evidence contract.",
        proof_obligations=(
            _proof_obligation(
                obligation_id=f"obligation:{node_id}:evidence",
                subject_id=node_id,
                requirement_id=f"requirement:{node_id}:evidence",
            ),
        ),
        evidence_contract=_satisfied_contract(
            contract_id=f"contract:{node_id}:evidence",
            subject_id=node_id,
            requirement_id=f"requirement:{node_id}:evidence",
            evidence_id=f"evidence:{node_id}:result",
        ),
    )


def _valid_implementation_node(node_id: str = "plan-node:implementation") -> PlanNode:
    return PlanNode(
        node_id=node_id,
        kind=PlanNodeKind.IMPLEMENTATION,
        objective="Prepare a bounded implementation candidate.",
        scope="Prepare candidate work only; do not execute operational action.",
        risk_level=RiskLevel.MODERATE,
        proof_obligations=(
            _proof_obligation(
                obligation_id=f"obligation:{node_id}:evidence",
                subject_id=node_id,
                requirement_id=f"requirement:{node_id}:evidence",
            ),
        ),
        evidence_contract=_satisfied_contract(
            contract_id=f"contract:{node_id}:evidence",
            subject_id=node_id,
            requirement_id=f"requirement:{node_id}:evidence",
            evidence_id=f"evidence:{node_id}:result",
        ),
        review_requirements=(
            _review_requirement(
                requirement_id=f"review:{node_id}:human",
                trigger=PlanReviewTrigger.OPERATOR_REQUEST,
            ),
        ),
        rollback_conditions=(
            _rollback_condition(rollback_id=f"rollback:{node_id}:reviewed-state"),
        ),
    )


def test_plan_review_requirement_rejects_blank_reviewer_role() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanReviewRequirement(
            requirement_id="review:blank-role",
            trigger=PlanReviewTrigger.HIGH_RISK,
            reviewer_role=" ",
            reason="Reviewer role is required.",
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_plan_rollback_condition_rejects_blank_statement() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanRollbackCondition(
            rollback_id="rollback:blank",
            statement=" ",
        )

    assert exc_info.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_ready_plan_node_requires_proof_obligations() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanNode(
            node_id="plan-node:no-proof",
            kind=PlanNodeKind.RESEARCH,
            objective="Ready nodes cannot be proof-free.",
            status=WorkState.READY_FOR_REVIEW,
            scope="Bounded research scope.",
        )

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_high_risk_plan_node_requires_review_requirement() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanNode(
            node_id="plan-node:high-risk",
            kind=PlanNodeKind.RESEARCH,
            objective="High-risk node requires review.",
            risk_level=RiskLevel.HIGH,
            scope="Bounded high-risk scope.",
            proof_obligations=(
                _proof_obligation(subject_id="plan-node:high-risk"),
            ),
            evidence_contract=_satisfied_contract(subject_id="plan-node:high-risk"),
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_blackfox_handoff_node_requires_review_and_rollback() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanNode(
            node_id="plan-node:blackfox-handoff",
            kind=PlanNodeKind.BLACKFOX_HANDOFF,
            objective="Prepare a BlackFox-compatible handoff candidate.",
            scope="Prepare handoff only.",
            proof_obligations=(
                _proof_obligation(subject_id="plan-node:blackfox-handoff"),
            ),
            evidence_contract=_satisfied_contract(subject_id="plan-node:blackfox-handoff"),
            review_requirements=(
                _review_requirement(
                    requirement_id="review:blackfox-handoff",
                    trigger=PlanReviewTrigger.BLACKFOX_HANDOFF,
                ),
            ),
        )

    assert exc_info.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_plan_node_reports_action_candidate_and_proof_requirement_ids() -> None:
    node = _valid_implementation_node()

    assert node.is_action_candidate is True
    assert node.requires_review is True
    assert node.has_proof_obligations is True
    assert node.has_evidence_contract is True
    assert node.proof_requirement_ids == ("requirement:plan-node:implementation:evidence",)


def test_plan_edge_cannot_point_to_same_node() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanEdge(
            edge_id="edge:self",
            source_node_id="plan-node:same",
            target_node_id="plan-node:same",
            dependency=PlanDependencyKind.REQUIRES,
            reason="A plan node cannot depend on itself.",
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_plan_graph_rejects_duplicate_node_ids() -> None:
    node = _valid_research_node("plan-node:duplicate")

    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanGraph(
            graph_id="plan-graph:duplicate",
            mission_id="mission:test",
            nodes=(node, node),
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_plan_graph_rejects_edge_referencing_unknown_node() -> None:
    node = _valid_research_node("plan-node:known")
    edge = PlanEdge(
        edge_id="edge:unknown",
        source_node_id="plan-node:known",
        target_node_id="plan-node:missing",
        dependency=PlanDependencyKind.REQUIRES,
        reason="The target node does not exist.",
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        PlanGraph(
            graph_id="plan-graph:unknown-edge",
            mission_id="mission:test",
            nodes=(node,),
            edges=(edge,),
        )

    assert exc_info.value.failure.mode == FailureMode.UNSUPPORTED_CLAIM


def test_plan_graph_reports_risk_review_action_and_missing_proof_views() -> None:
    research = _valid_research_node("plan-node:research")
    implementation = _valid_implementation_node("plan-node:implementation")
    draft_without_proof = PlanNode(
        node_id="plan-node:draft-no-proof",
        kind=PlanNodeKind.REVIEW,
        objective="Draft review node without proof.",
        scope="Bounded draft review scope.",
    )
    graph = PlanGraph(
        graph_id="plan-graph:views",
        mission_id="mission:test",
        nodes=(research, implementation, draft_without_proof),
        root_node_ids=("plan-node:research",),
        terminal_node_ids=("plan-node:implementation",),
    )

    assert graph.highest_risk == RiskLevel.MODERATE
    assert graph.nodes_requiring_review == (implementation,)
    assert graph.action_candidate_nodes == (implementation,)
    assert graph.nodes_missing_proof == (draft_without_proof,)


def test_validator_fails_closed_on_empty_vague_plan_graph() -> None:
    graph = PlanGraph(
        graph_id="plan-graph:empty",
        mission_id="mission:test",
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert result.outcome == DecisionOutcome.FAIL_CLOSED
    assert len(result.blocker_findings) == 3
    assert {finding.failure_mode for finding in result.findings} == {
        FailureMode.UNKNOWN_ACTION_TYPE,
    }


def test_validator_rejects_node_missing_scope() -> None:
    node = PlanNode(
        node_id="plan-node:no-scope",
        kind=PlanNodeKind.RESEARCH,
        objective="A node with no scope must fail validation.",
        proof_obligations=(
            _proof_obligation(
                subject_id="plan-node:no-scope",
                requirement_id="requirement:no-scope",
            ),
        ),
        evidence_contract=_satisfied_contract(
            subject_id="plan-node:no-scope",
            requirement_id="requirement:no-scope",
        ),
    )
    graph = PlanGraph(
        graph_id="plan-graph:no-scope",
        mission_id="mission:test",
        nodes=(node,),
        root_node_ids=("plan-node:no-scope",),
        terminal_node_ids=("plan-node:no-scope",),
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert any(finding.failure_mode == FailureMode.SCOPE_CREEP for finding in result.findings)


def test_validator_rejects_node_missing_proof_and_evidence_contract() -> None:
    node = PlanNode(
        node_id="plan-node:no-proof",
        kind=PlanNodeKind.RESEARCH,
        objective="Draft node missing proof obligations and evidence contract.",
        scope="Bounded research scope.",
    )
    graph = PlanGraph(
        graph_id="plan-graph:no-proof",
        mission_id="mission:test",
        nodes=(node,),
        root_node_ids=("plan-node:no-proof",),
        terminal_node_ids=("plan-node:no-proof",),
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert [finding.failure_mode for finding in result.findings].count(
        FailureMode.MISSING_EVIDENCE
    ) == 2


def test_validator_rejects_unsatisfied_evidence_contract() -> None:
    requirement = _requirement("requirement:unsatisfied")
    node = PlanNode(
        node_id="plan-node:unsatisfied",
        kind=PlanNodeKind.RESEARCH,
        objective="Node with unsatisfied evidence contract.",
        scope="Bounded research scope.",
        proof_obligations=(
            _proof_obligation(
                subject_id="plan-node:unsatisfied",
                requirement_id="requirement:unsatisfied",
            ),
        ),
        evidence_contract=EvidenceContract(
            contract_id="contract:unsatisfied",
            subject_id="plan-node:unsatisfied",
            requirements=(requirement,),
        ),
    )
    graph = PlanGraph(
        graph_id="plan-graph:unsatisfied",
        mission_id="mission:test",
        nodes=(node,),
        root_node_ids=("plan-node:unsatisfied",),
        terminal_node_ids=("plan-node:unsatisfied",),
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert any(finding.failure_mode == FailureMode.MISSING_EVIDENCE for finding in result.findings)


def test_validator_rejects_falsified_evidence_contract() -> None:
    requirement = _requirement("requirement:falsified")
    condition = FalsificationCondition(
        condition_id="falsifier:plan-node",
        subject_id="plan-node:falsified",
        statement="Contradictory test evidence falsifies the plan node.",
        evidence_kinds=(EvidenceKind.TEST_RESULT,),
    )
    node = PlanNode(
        node_id="plan-node:falsified",
        kind=PlanNodeKind.RESEARCH,
        objective="Node with falsified evidence contract.",
        scope="Bounded research scope.",
        proof_obligations=(
            _proof_obligation(
                subject_id="plan-node:falsified",
                requirement_id="requirement:falsified",
            ),
        ),
        evidence_contract=EvidenceContract(
            contract_id="contract:falsified",
            subject_id="plan-node:falsified",
            requirements=(requirement,),
            falsification_conditions=(condition,),
            references=(
                _verified_evidence(
                    "evidence:contradictory",
                    contradicts_claim_ids=("plan-node:falsified",),
                ),
            ),
        ),
    )
    graph = PlanGraph(
        graph_id="plan-graph:falsified",
        mission_id="mission:test",
        nodes=(node,),
        root_node_ids=("plan-node:falsified",),
        terminal_node_ids=("plan-node:falsified",),
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert any(finding.failure_mode == FailureMode.CONTRADICTION for finding in result.findings)


def test_validator_rejects_action_candidate_without_review_or_rollback() -> None:
    node = PlanNode(
        node_id="plan-node:implementation",
        kind=PlanNodeKind.IMPLEMENTATION,
        objective="Implementation candidate missing review and rollback.",
        scope="Prepare bounded implementation candidate.",
        proof_obligations=(
            _proof_obligation(
                subject_id="plan-node:implementation",
                requirement_id="requirement:implementation",
            ),
        ),
        evidence_contract=_satisfied_contract(
            subject_id="plan-node:implementation",
            requirement_id="requirement:implementation",
        ),
    )
    graph = PlanGraph(
        graph_id="plan-graph:action-boundary",
        mission_id="mission:test",
        nodes=(node,),
        root_node_ids=("plan-node:implementation",),
        terminal_node_ids=("plan-node:implementation",),
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert {finding.failure_mode for finding in result.findings} == {
        FailureMode.HUMAN_REVIEW_REQUIRED,
        FailureMode.UNKNOWN_ACTION_TYPE,
    }


def test_validator_rejects_dependency_cycle() -> None:
    first = _valid_research_node("plan-node:first")
    second = _valid_research_node("plan-node:second")
    graph = PlanGraph(
        graph_id="plan-graph:cycle",
        mission_id="mission:test",
        nodes=(first, second),
        edges=(
            PlanEdge(
                edge_id="edge:first-second",
                source_node_id="plan-node:first",
                target_node_id="plan-node:second",
                dependency=PlanDependencyKind.REQUIRES,
                reason="First requires second.",
            ),
            PlanEdge(
                edge_id="edge:second-first",
                source_node_id="plan-node:second",
                target_node_id="plan-node:first",
                dependency=PlanDependencyKind.REQUIRES,
                reason="Second requires first.",
            ),
        ),
        root_node_ids=("plan-node:first",),
        terminal_node_ids=("plan-node:second",),
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert any(finding.failure_mode == FailureMode.CONTRADICTION for finding in result.findings)


def test_validator_rejects_unreachable_terminal_node() -> None:
    root = _valid_research_node("plan-node:root")
    terminal = _valid_research_node("plan-node:terminal")
    graph = PlanGraph(
        graph_id="plan-graph:unreachable-terminal",
        mission_id="mission:test",
        nodes=(root, terminal),
        root_node_ids=("plan-node:root",),
        terminal_node_ids=("plan-node:terminal",),
    )

    result = validate_plan_graph(graph)

    assert result.failed_closed
    assert any(finding.failure_mode == FailureMode.UNSUPPORTED_CLAIM for finding in result.findings)


def test_validator_allows_valid_proof_carrying_plan_graph() -> None:
    research = _valid_research_node("plan-node:research")
    implementation = _valid_implementation_node("plan-node:implementation")
    graph = PlanGraph(
        graph_id="plan-graph:valid",
        mission_id="mission:test",
        nodes=(research, implementation),
        edges=(
            PlanEdge(
                edge_id="edge:research-implementation",
                source_node_id="plan-node:research",
                target_node_id="plan-node:implementation",
                dependency=PlanDependencyKind.ENABLES,
                reason="Research enables bounded implementation candidate preparation.",
            ),
        ),
        root_node_ids=("plan-node:research",),
        terminal_node_ids=("plan-node:implementation",),
    )

    result = validate_plan_graph(graph)

    assert result.valid
    assert result.outcome == DecisionOutcome.ALLOW
    assert result.findings == ()


def test_validator_policy_can_relax_terminal_reachability_when_needed() -> None:
    root = _valid_research_node("plan-node:root")
    terminal = _valid_research_node("plan-node:terminal")
    graph = PlanGraph(
        graph_id="plan-graph:relaxed-reachability",
        mission_id="mission:test",
        nodes=(root, terminal),
        root_node_ids=("plan-node:root",),
        terminal_node_ids=("plan-node:terminal",),
    )
    policy = PlanValidationPolicy(require_terminal_reachability_from_root=False)

    result = PlanGraphValidator(policy=policy).validate(graph)

    assert result.valid


def test_validation_finding_defaults_to_blocker_and_human_review() -> None:
    result = validate_plan_graph(PlanGraph(graph_id="plan-graph:finding", mission_id="mission:test"))
    finding = result.findings[0]

    assert finding.severity == PlanValidationSeverity.BLOCKER
    assert finding.requires_human_review is True
