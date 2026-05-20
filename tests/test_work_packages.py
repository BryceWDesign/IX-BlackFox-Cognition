"""Tests for cognitive work package models."""

import pytest

from ix_blackfox_cognition import (
    ActorKind,
    CognitiveWorkPackage,
    CognitionInvariantError,
    EvidenceContract,
    EvidenceKind,
    EvidenceReference,
    EvidenceRequirement,
    EvidenceSource,
    EvidenceState,
    EvidenceStrength,
    FailureMode,
    PlanNode,
    PlanNodeKind,
    ProofObligation,
    RiskLevel,
    WorkPackageBatch,
    WorkPackageDependency,
    WorkPackageKind,
    WorkPackageOutputKind,
    WorkPackageReviewGate,
)


def _evidence_source() -> EvidenceSource:
    return EvidenceSource(
        actor_kind=ActorKind.SYSTEM,
        actor_id="system:test-runner",
        description="Test evidence source.",
    )


def _evidence_contract(subject_id: str = "package:test") -> EvidenceContract:
    requirement = EvidenceRequirement(
        requirement_id=f"requirement:{subject_id}:evidence",
        statement="Verified test evidence is required.",
        acceptable_kinds=(EvidenceKind.TEST_RESULT,),
        minimum_strength=EvidenceStrength.STRONG,
    )
    reference = EvidenceReference(
        evidence_id=f"evidence:{subject_id}:result",
        kind=EvidenceKind.TEST_RESULT,
        state=EvidenceState.VERIFIED,
        source=_evidence_source(),
        summary="Verified evidence for a cognitive work package.",
        strength=EvidenceStrength.STRONG,
        locator=f"artifacts/{subject_id}.json",
        supports_claim_ids=(subject_id,),
    )
    return EvidenceContract(
        contract_id=f"contract:{subject_id}:evidence",
        subject_id=subject_id,
        requirements=(requirement,),
        references=(reference,),
    )


def _review_gate(
    gate_id: str = "review:package",
    *,
    human_required: bool = False,
) -> WorkPackageReviewGate:
    return WorkPackageReviewGate(
        gate_id=gate_id,
        reviewer_role="human-reviewer" if human_required else "model-critic",
        reason="Review is required before package acceptance.",
        human_required=human_required,
    )


def _rollback_requirement() -> tuple[str, ...]:
    return ("Rollback must restore the prior reviewed state.",)


def _forbidden_actions() -> tuple[str, ...]:
    return (
        "self_approve",
        "self_authorize",
        "silently_mutate_state",
        "bypass_policy",
    )


def _basic_research_package(package_id: str = "package:research") -> CognitiveWorkPackage:
    return CognitiveWorkPackage(
        package_id=package_id,
        kind=WorkPackageKind.RESEARCH,
        objective="Research the bounded cognition requirement.",
        scope="Review only the explicitly scoped mission and evidence contract.",
        evidence_requirement_ids=(f"requirement:{package_id}:evidence",),
        evidence_contract=_evidence_contract(package_id),
        expected_outputs=(WorkPackageOutputKind.CLAIMS,),
        forbidden_actions=_forbidden_actions(),
    )


def test_review_gate_rejects_blank_reviewer_role() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        WorkPackageReviewGate(
            gate_id="review:blank-role",
            reviewer_role=" ",
            reason="Reviewer role is required.",
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_work_package_dependency_rejects_self_dependency() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        WorkPackageDependency(
            dependency_id="dependency:self",
            required_before="package:same",
            required_after="package:same",
            reason="A package cannot depend on itself.",
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_work_package_requires_objective_scope_outputs_evidence_and_forbidden_actions() -> None:
    with pytest.raises(CognitionInvariantError) as no_scope:
        CognitiveWorkPackage(
            package_id="package:no-scope",
            kind=WorkPackageKind.RESEARCH,
            objective="Research something bounded.",
            scope=" ",
            evidence_requirement_ids=("requirement:test",),
            expected_outputs=(WorkPackageOutputKind.CLAIMS,),
            forbidden_actions=_forbidden_actions(),
        )

    with pytest.raises(CognitionInvariantError) as no_outputs:
        CognitiveWorkPackage(
            package_id="package:no-outputs",
            kind=WorkPackageKind.RESEARCH,
            objective="Research something bounded.",
            scope="Bounded scope.",
            evidence_requirement_ids=("requirement:test",),
            forbidden_actions=_forbidden_actions(),
        )

    with pytest.raises(CognitionInvariantError) as no_evidence:
        CognitiveWorkPackage(
            package_id="package:no-evidence",
            kind=WorkPackageKind.RESEARCH,
            objective="Research something bounded.",
            scope="Bounded scope.",
            expected_outputs=(WorkPackageOutputKind.CLAIMS,),
            forbidden_actions=_forbidden_actions(),
        )

    with pytest.raises(CognitionInvariantError) as no_forbidden_actions:
        CognitiveWorkPackage(
            package_id="package:no-forbidden-actions",
            kind=WorkPackageKind.RESEARCH,
            objective="Research something bounded.",
            scope="Bounded scope.",
            evidence_requirement_ids=("requirement:test",),
            expected_outputs=(WorkPackageOutputKind.CLAIMS,),
        )

    assert no_scope.value.failure.mode == FailureMode.SCOPE_CREEP
    assert no_outputs.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE
    assert no_evidence.value.failure.mode == FailureMode.MISSING_EVIDENCE
    assert no_forbidden_actions.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_basic_research_package_is_bounded_but_not_action_adjacent() -> None:
    package = _basic_research_package()

    assert package.requires_review is False
    assert package.requires_human_review is False
    assert package.is_action_adjacent is False
    assert package.has_satisfied_evidence_contract is True
    assert package.expected_outputs == (WorkPackageOutputKind.CLAIMS,)


def test_high_risk_package_requires_review_gate() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        CognitiveWorkPackage(
            package_id="package:high-risk",
            kind=WorkPackageKind.RESEARCH,
            objective="Research a high-risk cognition claim.",
            scope="Bounded high-risk research scope.",
            risk_level=RiskLevel.HIGH,
            evidence_requirement_ids=("requirement:high-risk",),
            expected_outputs=(WorkPackageOutputKind.RISK_REGISTER,),
            forbidden_actions=_forbidden_actions(),
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_action_adjacent_implementation_requires_review_and_rollback() -> None:
    with pytest.raises(CognitionInvariantError) as missing_review:
        CognitiveWorkPackage(
            package_id="package:implementation:no-review",
            kind=WorkPackageKind.IMPLEMENTATION,
            objective="Prepare an implementation candidate.",
            scope="Prepare candidate only; do not execute operational action.",
            evidence_requirement_ids=("requirement:implementation",),
            expected_outputs=(WorkPackageOutputKind.PLAN_UPDATE,),
            forbidden_actions=_forbidden_actions(),
            rollback_requirements=_rollback_requirement(),
        )

    with pytest.raises(CognitionInvariantError) as missing_rollback:
        CognitiveWorkPackage(
            package_id="package:implementation:no-rollback",
            kind=WorkPackageKind.IMPLEMENTATION,
            objective="Prepare an implementation candidate.",
            scope="Prepare candidate only; do not execute operational action.",
            evidence_requirement_ids=("requirement:implementation",),
            review_gates=(_review_gate(),),
            expected_outputs=(WorkPackageOutputKind.PLAN_UPDATE,),
            forbidden_actions=_forbidden_actions(),
        )

    assert missing_review.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED
    assert missing_rollback.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_memory_update_package_requires_human_review() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        CognitiveWorkPackage(
            package_id="package:memory-update",
            kind=WorkPackageKind.MEMORY_UPDATE,
            objective="Propose a governed memory update.",
            scope="Create a proposal only; do not promote memory.",
            evidence_requirement_ids=("requirement:memory-update",),
            review_gates=(_review_gate(human_required=False),),
            rollback_requirements=_rollback_requirement(),
            expected_outputs=(WorkPackageOutputKind.MEMORY_UPDATE_PROPOSAL,),
            forbidden_actions=_forbidden_actions(),
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_self_improvement_package_requires_human_review() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        CognitiveWorkPackage(
            package_id="package:self-improvement",
            kind=WorkPackageKind.SELF_IMPROVEMENT_PROPOSAL,
            objective="Propose a change to cognition routing rules.",
            scope="Propose only; do not promote or apply the self-improvement.",
            evidence_requirement_ids=("requirement:self-improvement",),
            review_gates=(_review_gate(human_required=False),),
            rollback_requirements=_rollback_requirement(),
            expected_outputs=(WorkPackageOutputKind.SELF_IMPROVEMENT_PROPOSAL,),
            forbidden_actions=_forbidden_actions(),
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_blackfox_handoff_package_is_action_adjacent_and_human_reviewed() -> None:
    package = CognitiveWorkPackage(
        package_id="package:blackfox-handoff",
        kind=WorkPackageKind.BLACKFOX_HANDOFF,
        objective="Prepare a BlackFox-compatible action candidate.",
        scope="Prepare handoff only; IX-BlackFox governs execution.",
        evidence_requirement_ids=("requirement:blackfox-handoff",),
        review_gates=(_review_gate(human_required=True),),
        rollback_requirements=_rollback_requirement(),
        expected_outputs=(WorkPackageOutputKind.BLACKFOX_ACTION_CANDIDATE,),
        forbidden_actions=_forbidden_actions(),
    )

    assert package.is_action_adjacent is True
    assert package.requires_review is True
    assert package.requires_human_review is True


def test_work_package_from_plan_node_preserves_scope_evidence_and_risk() -> None:
    node = PlanNode(
        node_id="plan-node:implementation",
        kind=PlanNodeKind.IMPLEMENTATION,
        objective="Prepare bounded implementation work.",
        scope="Prepare a candidate patch package only.",
        risk_level=RiskLevel.MODERATE,
        proof_obligations=(
            ProofObligation(
                obligation_id="obligation:implementation",
                subject_id="plan-node:implementation",
                statement="Implementation package requires evidence.",
                requirement_ids=("requirement:implementation",),
            ),
        ),
        evidence_contract=_evidence_contract("plan-node:implementation"),
    )

    package = CognitiveWorkPackage.from_plan_node(
        package_id="package:from-plan-node",
        node=node,
        expected_outputs=(WorkPackageOutputKind.PLAN_UPDATE,),
        forbidden_actions=_forbidden_actions(),
        review_gates=(_review_gate(),),
        rollback_requirements=_rollback_requirement(),
    )

    assert package.kind == WorkPackageKind.IMPLEMENTATION
    assert package.source_plan_node_id == "plan-node:implementation"
    assert package.scope == "Prepare a candidate patch package only."
    assert package.risk_level == RiskLevel.MODERATE
    assert package.evidence_requirement_ids == ("requirement:implementation",)
    assert package.evidence_contract == node.evidence_contract


def test_work_package_from_plan_node_rejects_node_without_scope() -> None:
    node = PlanNode(
        node_id="plan-node:no-scope",
        kind=PlanNodeKind.RESEARCH,
        objective="Plan node without scope cannot become work package.",
        proof_obligations=(
            ProofObligation(
                obligation_id="obligation:no-scope",
                subject_id="plan-node:no-scope",
                statement="Evidence is required.",
                requirement_ids=("requirement:no-scope",),
            ),
        ),
        evidence_contract=_evidence_contract("plan-node:no-scope"),
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        CognitiveWorkPackage.from_plan_node(
            package_id="package:no-scope",
            node=node,
            expected_outputs=(WorkPackageOutputKind.CLAIMS,),
            forbidden_actions=_forbidden_actions(),
        )

    assert exc_info.value.failure.mode == FailureMode.SCOPE_CREEP


def test_work_package_batch_requires_at_least_one_package() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        WorkPackageBatch(
            batch_id="batch:empty",
            mission_id="mission:test",
            plan_graph_id="plan-graph:test",
            packages=(),
        )

    assert exc_info.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_work_package_batch_rejects_duplicate_package_ids() -> None:
    package = _basic_research_package("package:duplicate")

    with pytest.raises(CognitionInvariantError) as exc_info:
        WorkPackageBatch(
            batch_id="batch:duplicates",
            mission_id="mission:test",
            plan_graph_id="plan-graph:test",
            packages=(package, package),
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_work_package_batch_rejects_dependency_on_unknown_package() -> None:
    package = _basic_research_package("package:known")
    dependency = WorkPackageDependency(
        dependency_id="dependency:unknown",
        required_before="package:known",
        required_after="package:missing",
        reason="Dependency references a missing package.",
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        WorkPackageBatch(
            batch_id="batch:unknown-dependency",
            mission_id="mission:test",
            plan_graph_id="plan-graph:test",
            packages=(package,),
            dependencies=(dependency,),
        )

    assert exc_info.value.failure.mode == FailureMode.UNSUPPORTED_CLAIM


def test_work_package_batch_reports_human_review_and_action_adjacent_packages() -> None:
    research = _basic_research_package("package:research")
    handoff = CognitiveWorkPackage(
        package_id="package:blackfox-handoff",
        kind=WorkPackageKind.BLACKFOX_HANDOFF,
        objective="Prepare BlackFox-compatible action candidate.",
        scope="Prepare handoff only.",
        evidence_requirement_ids=("requirement:blackfox-handoff",),
        review_gates=(_review_gate(human_required=True),),
        rollback_requirements=_rollback_requirement(),
        expected_outputs=(WorkPackageOutputKind.BLACKFOX_ACTION_CANDIDATE,),
        forbidden_actions=_forbidden_actions(),
    )
    dependency = WorkPackageDependency(
        dependency_id="dependency:research-handoff",
        required_before="package:research",
        required_after="package:blackfox-handoff",
        reason="Research must precede handoff preparation.",
    )
    batch = WorkPackageBatch(
        batch_id="batch:test",
        mission_id="mission:test",
        plan_graph_id="plan-graph:test",
        packages=(research, handoff),
        dependencies=(dependency,),
    )

    assert batch.packages_requiring_human_review == (handoff,)
    assert batch.action_adjacent_packages == (handoff,)
