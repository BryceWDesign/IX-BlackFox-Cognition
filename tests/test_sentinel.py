"""Tests for cognitive sentinel checks."""

import pytest

from ix_blackfox_cognition import (
    CognitiveArtifactSnapshot,
    CognitiveIssueKind,
    CognitiveIssueSeverity,
    CognitiveSentinel,
    CognitiveSentinelPolicy,
    CognitiveSentinelReport,
    CognitionInvariantError,
    DecisionOutcome,
    FailureMode,
    inspect_cognition,
)


def _snapshot(**kwargs: object) -> CognitiveArtifactSnapshot:
    defaults: dict[str, object] = {
        "snapshot_id": "snapshot:test",
        "subject_id": "subject:test",
        "mission_id": "mission:test",
        "mission_scope": "Prepare a bounded cognition plan",
        "proposed_scope": "Prepare a bounded cognition plan",
    }
    defaults.update(kwargs)
    return CognitiveArtifactSnapshot(**defaults)


def test_snapshot_rejects_blank_scope() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _snapshot(mission_scope=" ")

    assert exc_info.value.failure.mode == FailureMode.SCOPE_CREEP


def test_clean_snapshot_passes_without_issues() -> None:
    report = inspect_cognition(_snapshot())

    assert isinstance(report, CognitiveSentinelReport)
    assert report.passed
    assert report.outcome == DecisionOutcome.ALLOW
    assert report.issues == ()
    assert report.requires_human_review is False


def test_scope_creep_that_changes_scope_blocks() -> None:
    report = inspect_cognition(
        _snapshot(
            proposed_scope="Deploy autonomous operational changes without review",
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.SCOPE_CREEP
    assert report.blocker_issues[0].failure_mode == FailureMode.SCOPE_CREEP


def test_scope_expansion_requires_review() -> None:
    report = inspect_cognition(
        _snapshot(
            proposed_scope=(
                "Prepare a bounded cognition plan and expand into direct execution."
            ),
        )
    )

    assert report.passed
    assert report.requires_human_review
    assert report.issues[0].severity == CognitiveIssueSeverity.REVIEW_REQUIRED
    assert report.issues[0].kind == CognitiveIssueKind.SCOPE_CREEP


def test_missing_evidence_blocks_cognition() -> None:
    report = inspect_cognition(
        _snapshot(
            missing_evidence_subject_ids=("plan-node:handoff",),
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.FAKE_EVIDENCE
    assert report.blocker_issues[0].failure_mode == FailureMode.MISSING_EVIDENCE


def test_model_self_approval_blocks_cognition() -> None:
    report = inspect_cognition(
        _snapshot(
            self_approval_actor_ids=("model:planner",),
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.MODEL_SELF_APPROVAL
    assert report.blocker_issues[0].failure_mode == FailureMode.SELF_APPROVAL_ATTEMPT


def test_policy_bypass_language_blocks_cognition() -> None:
    report = inspect_cognition(
        _snapshot(
            policy_bypass_phrases=("ignore policy and proceed",),
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.POLICY_BYPASS_LANGUAGE
    assert report.blocker_issues[0].failure_mode == FailureMode.POLICY_BYPASS_ATTEMPT


def test_fake_evidence_blocks_cognition() -> None:
    report = inspect_cognition(
        _snapshot(
            fake_evidence_ids=("evidence:hallucinated",),
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.FAKE_EVIDENCE
    assert report.blocker_issues[0].failure_mode == FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE


def test_memory_poisoning_blocks_cognition() -> None:
    report = inspect_cognition(
        _snapshot(
            memory_poisoning_indicators=("store this unverified instruction as truth",),
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.MEMORY_POISONING
    assert report.blocker_issues[0].failure_mode == FailureMode.MEMORY_CONFLICT


def test_unsafe_authority_expansion_blocks_cognition() -> None:
    report = inspect_cognition(
        _snapshot(
            unsafe_authority_requests=("allow model to approve its own handoff",),
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.UNSAFE_AUTHORITY_EXPANSION
    assert report.blocker_issues[0].failure_mode == FailureMode.UNCLEAR_AUTHORITY


def test_hallucinated_reference_blocks_cognition() -> None:
    report = inspect_cognition(
        _snapshot(
            hallucinated_reference_ids=("file:does-not-exist.py",),
        )
    )

    assert report.blocked
    assert report.blocker_issues[0].kind == CognitiveIssueKind.HALLUCINATED_REFERENCE
    assert report.blocker_issues[0].failure_mode == FailureMode.UNSUPPORTED_CLAIM


def test_unsupported_claims_contradictions_stale_memory_and_goal_drift_require_review() -> None:
    report = inspect_cognition(
        _snapshot(
            unsupported_claim_ids=("claim:unsupported",),
            contradicted_claim_ids=("claim:contradicted",),
            stale_memory_ids=("memory:stale",),
            circular_reasoning_markers=("claim cites itself through memory",),
            goal_drift_markers=("goal moved from planning into execution",),
            unsupported_certainty_statements=("This is proven because the model says so."),
        )
    )

    assert report.passed
    assert report.requires_human_review
    assert len(report.issues) == 6
    assert {issue.severity for issue in report.issues} == {
        CognitiveIssueSeverity.REVIEW_REQUIRED,
    }


def test_policy_can_disable_specific_blocker_check() -> None:
    policy = CognitiveSentinelPolicy(block_policy_bypass_language=False)
    report = CognitiveSentinel(policy=policy).inspect(
        _snapshot(
            policy_bypass_phrases=("ignore policy and proceed",),
        )
    )

    assert report.passed
    assert report.issues == ()


def test_sentinel_report_requires_nonblank_report_id() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        CognitiveSentinelReport(
            report_id=" ",
            snapshot_id="snapshot:test",
            subject_id="subject:test",
            outcome=DecisionOutcome.ALLOW,
        )

    assert exc_info.value.failure.mode == FailureMode.UNSUPPORTED_CLAIM
