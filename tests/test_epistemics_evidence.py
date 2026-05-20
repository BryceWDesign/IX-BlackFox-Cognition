"""Tests for epistemic claim ledgers and evidence contracts."""

import pytest

from ix_blackfox_cognition import (
    ActorKind,
    ClaimKind,
    ClaimLedger,
    ClaimRecord,
    ClaimSource,
    ClaimState,
    ClaimStateTransition,
    CognitionInvariantError,
    ConfidenceBasis,
    ConfidenceSignal,
    EvidenceContract,
    EvidenceKind,
    EvidenceReference,
    EvidenceRequirement,
    EvidenceSource,
    EvidenceState,
    EvidenceStrength,
    FailureMode,
    FalsificationCondition,
    ProofObligation,
    RiskLevel,
)


def _model_source() -> ClaimSource:
    return ClaimSource(
        actor_kind=ActorKind.MODEL,
        actor_id="model:planner",
        description="Planner model used for bounded cognition.",
    )


def _human_claim_source() -> ClaimSource:
    return ClaimSource(
        actor_kind=ActorKind.HUMAN,
        actor_id="human:reviewer",
        description="Human reviewer.",
    )


def _evidence_source(actor_kind: ActorKind = ActorKind.SYSTEM) -> EvidenceSource:
    return EvidenceSource(
        actor_kind=actor_kind,
        actor_id=f"{actor_kind.value}:source",
        description="Evidence source used by tests.",
    )


def _verified_test_evidence(evidence_id: str = "evidence:test-result") -> EvidenceReference:
    return EvidenceReference(
        evidence_id=evidence_id,
        kind=EvidenceKind.TEST_RESULT,
        state=EvidenceState.VERIFIED,
        source=_evidence_source(),
        summary="Verified test result evidence.",
        strength=EvidenceStrength.STRONG,
        locator="tests/result.json",
        supports_claim_ids=("claim:verified",),
    )


def test_confidence_signal_is_never_evidence() -> None:
    confidence = ConfidenceSignal(
        basis=ConfidenceBasis.MODEL_SELF_ASSESSMENT,
        score=0.97,
        rationale="The model is confident, but confidence is not evidence.",
    )

    assert confidence.is_evidence is False


def test_confidence_signal_rejects_out_of_range_score() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ConfidenceSignal(
            basis=ConfidenceBasis.MODEL_SELF_ASSESSMENT,
            score=1.25,
        )

    assert exc_info.value.failure.mode == FailureMode.MODEL_CONFIDENCE_AS_EVIDENCE


def test_verified_claim_requires_evidence_ids() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ClaimRecord(
            claim_id="claim:no-evidence",
            kind=ClaimKind.FACT,
            statement="This cannot be verified by confidence alone.",
            source=_model_source(),
            state=ClaimState.VERIFIED,
            confidence=ConfidenceSignal(
                basis=ConfidenceBasis.MODEL_SELF_ASSESSMENT,
                score=0.99,
            ),
        )

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_claim_with_model_confidence_remains_unverified_without_evidence() -> None:
    claim = ClaimRecord(
        claim_id="claim:model-output",
        kind=ClaimKind.MODEL_OUTPUT,
        statement="The model proposes that this plan is likely correct.",
        source=_model_source(),
        state=ClaimState.UNVERIFIED,
        confidence=ConfidenceSignal(
            basis=ConfidenceBasis.MODEL_SELF_ASSESSMENT,
            score=0.91,
        ),
    )

    assert claim.has_evidence is False
    assert claim.requires_evidence is True
    assert claim.is_trust_eligible is False


def test_human_approved_claim_requires_human_approval_id() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ClaimRecord(
            claim_id="claim:no-human-approval-id",
            kind=ClaimKind.HUMAN_INPUT,
            statement="This cannot be human-approved without an approval id.",
            source=_human_claim_source(),
            state=ClaimState.HUMAN_APPROVED,
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_contradicted_claim_requires_conflicting_claim_reference() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ClaimRecord(
            claim_id="claim:contradicted",
            kind=ClaimKind.FACT,
            statement="This claim is contradicted but names no conflicting claim.",
            source=_model_source(),
            state=ClaimState.CONTRADICTED,
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_claim_record_can_transition_to_verified_with_evidence_ids() -> None:
    claim = ClaimRecord(
        claim_id="claim:verified",
        kind=ClaimKind.FACT,
        statement="A verified claim needs evidence.",
        source=_model_source(),
    )

    verified_claim = claim.with_state(
        ClaimState.VERIFIED,
        evidence_ids=("evidence:test-result",),
    )

    assert verified_claim.state == ClaimState.VERIFIED
    assert verified_claim.has_evidence is True
    assert verified_claim.is_trust_eligible is True


def test_claim_ledger_rejects_duplicate_claim_ids() -> None:
    claim = ClaimRecord(
        claim_id="claim:duplicate",
        kind=ClaimKind.FACT,
        statement="Duplicate claim ids must fail.",
        source=_model_source(),
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        ClaimLedger(
            ledger_id="ledger:duplicate",
            claims=(claim, claim),
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_claim_ledger_adds_unique_claims_without_silent_mutation() -> None:
    ledger = ClaimLedger(ledger_id="ledger:test")
    claim = ClaimRecord(
        claim_id="claim:unique",
        kind=ClaimKind.ASSUMPTION,
        statement="This assumption needs evidence before trust.",
        source=_model_source(),
        state=ClaimState.ASSUMED,
    )

    updated = ledger.add_claim(claim)

    assert ledger.claims == ()
    assert updated.claim_by_id("claim:unique") == claim
    assert updated.unverified_claims == (claim,)
    assert updated.trust_eligible_claims == ()


def test_claim_state_transition_to_verified_requires_evidence_ids() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ClaimStateTransition(
            transition_id="transition:no-evidence",
            claim_id="claim:needs-evidence",
            from_state=ClaimState.UNVERIFIED,
            to_state=ClaimState.VERIFIED,
            reason="Transition should fail without evidence.",
            actor=_model_source(),
        )

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_claim_ledger_records_transition_without_rewriting_claim() -> None:
    claim = ClaimRecord(
        claim_id="claim:transition",
        kind=ClaimKind.FACT,
        statement="Transition events should not silently rewrite claims.",
        source=_model_source(),
        state=ClaimState.UNVERIFIED,
    )
    ledger = ClaimLedger(ledger_id="ledger:transition", claims=(claim,))
    transition = ClaimStateTransition(
        transition_id="transition:verified",
        claim_id="claim:transition",
        from_state=ClaimState.UNVERIFIED,
        to_state=ClaimState.VERIFIED,
        reason="Evidence was attached for verification.",
        actor=_human_claim_source(),
        evidence_ids=("evidence:test-result",),
    )

    updated = ledger.record_transition(transition)

    assert updated.transitions == (transition,)
    assert updated.claim_by_id("claim:transition") == claim
    assert updated.claim_by_id("claim:transition").state == ClaimState.UNVERIFIED


def test_verified_evidence_reference_requires_locator_or_digest() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        EvidenceReference(
            evidence_id="evidence:no-locator",
            kind=EvidenceKind.TEST_RESULT,
            state=EvidenceState.VERIFIED,
            source=_evidence_source(),
            summary="Verified evidence without locator or digest should fail.",
        )

    assert exc_info.value.failure.mode == FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE


def test_human_approval_evidence_requires_human_source() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        EvidenceReference(
            evidence_id="evidence:bad-human-approval",
            kind=EvidenceKind.HUMAN_APPROVAL,
            state=EvidenceState.VERIFIED,
            source=_evidence_source(actor_kind=ActorKind.MODEL),
            summary="A model cannot be the source of human approval evidence.",
            locator="approval.json",
        )

    assert exc_info.value.failure.mode == FailureMode.HUMAN_REVIEW_REQUIRED


def test_evidence_requirement_accepts_matching_verified_evidence() -> None:
    requirement = EvidenceRequirement(
        requirement_id="requirement:test",
        statement="A strong verified test result is required.",
        acceptable_kinds=(EvidenceKind.TEST_RESULT,),
        minimum_strength=EvidenceStrength.STRONG,
        risk_level=RiskLevel.MODERATE,
    )

    assert requirement.satisfied_by((_verified_test_evidence(),)) is True


def test_evidence_requirement_rejects_unverified_or_weak_evidence() -> None:
    requirement = EvidenceRequirement(
        requirement_id="requirement:strong-static-analysis",
        statement="Strong verified static-analysis evidence is required.",
        acceptable_kinds=(EvidenceKind.STATIC_ANALYSIS,),
        minimum_strength=EvidenceStrength.STRONG,
    )
    weak_reference = EvidenceReference(
        evidence_id="evidence:weak-static-analysis",
        kind=EvidenceKind.STATIC_ANALYSIS,
        state=EvidenceState.PRESENT,
        source=_evidence_source(),
        summary="Weak and unverified static-analysis reference.",
        strength=EvidenceStrength.WEAK,
        locator="analysis.txt",
    )

    assert requirement.satisfied_by((weak_reference,)) is False


def test_proof_obligation_requires_requirement_ids() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ProofObligation(
            obligation_id="obligation:no-requirements",
            subject_id="plan-node:test",
            statement="Proof obligations must identify evidence requirements.",
            requirement_ids=(),
        )

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_falsification_condition_matches_contradictory_evidence() -> None:
    condition = FalsificationCondition(
        condition_id="falsifier:test",
        subject_id="claim:verified",
        statement="Contradictory test evidence falsifies this claim.",
        evidence_kinds=(EvidenceKind.TEST_RESULT,),
    )
    contradictory_evidence = EvidenceReference(
        evidence_id="evidence:contradiction",
        kind=EvidenceKind.TEST_RESULT,
        state=EvidenceState.VERIFIED,
        source=_evidence_source(),
        summary="Verified evidence contradicting a claim.",
        strength=EvidenceStrength.STRONG,
        locator="tests/contradiction.json",
        contradicts_claim_ids=("claim:verified",),
    )

    assert condition.matches(contradictory_evidence) is True


def test_evidence_contract_reports_unsatisfied_requirements() -> None:
    requirement = EvidenceRequirement(
        requirement_id="requirement:test",
        statement="A verified test result is required.",
        acceptable_kinds=(EvidenceKind.TEST_RESULT,),
    )
    contract = EvidenceContract(
        contract_id="contract:test",
        subject_id="claim:verified",
        requirements=(requirement,),
    )

    assert contract.satisfied is False
    assert contract.trust_eligible is False
    assert contract.unsatisfied_requirements == (requirement,)


def test_evidence_contract_becomes_trust_eligible_when_satisfied_and_not_falsified() -> None:
    requirement = EvidenceRequirement(
        requirement_id="requirement:test",
        statement="A verified test result is required.",
        acceptable_kinds=(EvidenceKind.TEST_RESULT,),
        minimum_strength=EvidenceStrength.STRONG,
    )
    contract = EvidenceContract(
        contract_id="contract:trusted",
        subject_id="claim:verified",
        requirements=(requirement,),
    ).add_reference(_verified_test_evidence())

    assert contract.satisfied is True
    assert contract.falsified is False
    assert contract.trust_eligible is True


def test_evidence_contract_is_not_trust_eligible_when_falsified() -> None:
    requirement = EvidenceRequirement(
        requirement_id="requirement:test",
        statement="A verified test result is required.",
        acceptable_kinds=(EvidenceKind.TEST_RESULT,),
        minimum_strength=EvidenceStrength.STRONG,
    )
    condition = FalsificationCondition(
        condition_id="falsifier:test",
        subject_id="claim:verified",
        statement="Contradictory test result falsifies the subject.",
        evidence_kinds=(EvidenceKind.TEST_RESULT,),
    )
    contradictory_evidence = EvidenceReference(
        evidence_id="evidence:contradictory-test",
        kind=EvidenceKind.TEST_RESULT,
        state=EvidenceState.VERIFIED,
        source=_evidence_source(),
        summary="Contradictory verified test result.",
        strength=EvidenceStrength.STRONG,
        locator="tests/contradictory.json",
        contradicts_claim_ids=("claim:verified",),
    )
    contract = EvidenceContract(
        contract_id="contract:falsified",
        subject_id="claim:verified",
        requirements=(requirement,),
        falsification_conditions=(condition,),
        references=(contradictory_evidence,),
    )

    assert contract.satisfied is True
    assert contract.falsified is True
    assert contract.trust_eligible is False


def test_evidence_contract_rejects_duplicate_evidence_reference_ids() -> None:
    requirement = EvidenceRequirement(
        requirement_id="requirement:test",
        statement="A verified test result is required.",
        acceptable_kinds=(EvidenceKind.TEST_RESULT,),
    )
    reference = _verified_test_evidence("evidence:duplicate")
    contract = EvidenceContract(
        contract_id="contract:duplicates",
        subject_id="claim:verified",
        requirements=(requirement,),
        references=(reference,),
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        contract.add_reference(reference)

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION
