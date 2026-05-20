"""Tests for shared IX-BlackFox-Cognition core types."""

import pytest

from ix_blackfox_cognition import (
    ActorKind,
    ClaimState,
    CognitionInvariantError,
    DecisionOutcome,
    EvidenceState,
    FailureMode,
    FailureRecord,
    MemoryState,
    RiskLevel,
    WorkState,
    fail_closed,
    require_invariant,
)


def test_actor_kinds_cover_human_model_system_policy_and_blackfox() -> None:
    assert ActorKind.HUMAN.value == "human"
    assert ActorKind.MODEL.value == "model"
    assert ActorKind.SYSTEM.value == "system"
    assert ActorKind.POLICY.value == "policy"
    assert ActorKind.BLACKFOX.value == "blackfox"


def test_decision_outcomes_include_fail_closed_and_review_required() -> None:
    assert DecisionOutcome.ALLOW.value == "allow"
    assert DecisionOutcome.REVIEW_REQUIRED.value == "review_required"
    assert DecisionOutcome.FAIL_CLOSED.value == "fail_closed"


def test_common_states_preserve_evidence_claim_memory_and_work_lifecycles() -> None:
    assert EvidenceState.MISSING.value == "missing"
    assert EvidenceState.VERIFIED.value == "verified"
    assert ClaimState.EVIDENCE_REQUIRED.value == "evidence_required"
    assert ClaimState.HUMAN_APPROVED.value == "human_approved"
    assert MemoryState.QUARANTINED.value == "quarantined"
    assert WorkState.READY_FOR_REVIEW.value == "ready_for_review"


def test_risk_level_ranks_are_stable_and_ordered() -> None:
    assert RiskLevel.LOW.rank() < RiskLevel.MODERATE.rank()
    assert RiskLevel.MODERATE.rank() < RiskLevel.HIGH.rank()
    assert RiskLevel.HIGH.rank() < RiskLevel.CRITICAL.rank()


def test_failure_record_is_fail_closed_by_default() -> None:
    failure = FailureRecord(
        mode=FailureMode.MISSING_EVIDENCE,
        message="Evidence is required before trust can be granted.",
    )

    assert failure.fail_closed is True
    assert failure.mode == FailureMode.MISSING_EVIDENCE
    assert "Evidence is required" in failure.message


def test_fail_closed_raises_invariant_error_with_failure_record() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        fail_closed(
            mode=FailureMode.UNCLEAR_AUTHORITY,
            message="Human authority is unclear.",
        )

    assert exc_info.value.failure.mode == FailureMode.UNCLEAR_AUTHORITY
    assert exc_info.value.failure.fail_closed is True
    assert "unclear_authority" in str(exc_info.value)


def test_require_invariant_passes_when_condition_is_true() -> None:
    require_invariant(
        condition=True,
        mode=FailureMode.MISSING_POLICY,
        message="This should not fail.",
    )


def test_require_invariant_fails_closed_when_condition_is_false() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        require_invariant(
            condition=False,
            mode=FailureMode.MODEL_CONFIDENCE_AS_EVIDENCE,
            message="Model confidence cannot be treated as evidence.",
        )

    assert exc_info.value.failure.mode == FailureMode.MODEL_CONFIDENCE_AS_EVIDENCE
    assert "Model confidence cannot be treated as evidence" in str(exc_info.value)
