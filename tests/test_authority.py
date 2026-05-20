"""Tests for IX-BlackFox-Cognition authority models and firewall decisions."""

import pytest

from ix_blackfox_cognition import (
    ActorKind,
    AuthorityBoundary,
    AuthorityFirewall,
    AuthorityLevel,
    AuthorityRequest,
    AuthoritySnapshot,
    CognitionInvariantError,
    CognitionPermission,
    DecisionOutcome,
    FailureMode,
    ForbiddenAction,
    HumanAuthorityRequirement,
    RiskLevel,
)


def _model_boundary(
    *,
    allowed_permissions: tuple[CognitionPermission, ...],
    authority_level: AuthorityLevel = AuthorityLevel.PROPOSE_ONLY,
    human_requirements: tuple[HumanAuthorityRequirement, ...] = (),
) -> AuthorityBoundary:
    return AuthorityBoundary(
        actor_kind=ActorKind.MODEL,
        actor_id="model:planner",
        authority_level=authority_level,
        allowed_permissions=allowed_permissions,
        forbidden_actions=(
            ForbiddenAction.SELF_AUTHORIZE,
            ForbiddenAction.SELF_APPROVE,
            ForbiddenAction.BYPASS_POLICY,
            ForbiddenAction.SILENTLY_MUTATE_STATE,
        ),
        human_requirements=human_requirements,
    )


def _snapshot(boundary: AuthorityBoundary) -> AuthoritySnapshot:
    return AuthoritySnapshot(
        snapshot_id="authority-snapshot:test",
        boundaries=(boundary,),
    )


def _request(
    *,
    requested_permission: CognitionPermission,
    risk_level: RiskLevel = RiskLevel.LOW,
    evidence_ids: tuple[str, ...] = (),
    human_approval_id: str | None = None,
    actor_kind: ActorKind = ActorKind.MODEL,
    actor_id: str = "model:planner",
) -> AuthorityRequest:
    return AuthorityRequest(
        request_id=f"authority-request:{requested_permission.value}",
        actor_kind=actor_kind,
        actor_id=actor_id,
        requested_permission=requested_permission,
        justification="The request is part of a bounded cognition test.",
        risk_level=risk_level,
        target="test-target",
        evidence_ids=evidence_ids,
        human_approval_id=human_approval_id,
    )


def test_human_authority_requirement_rejects_blank_identifier() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        HumanAuthorityRequirement(
            requirement_id=" ",
            reason="Human review is required.",
            required_for=(CognitionPermission.PROPOSE_PLAN,),
        )

    assert exc_info.value.failure.mode == FailureMode.UNCLEAR_AUTHORITY
    assert exc_info.value.failure.fail_closed is True


def test_only_human_actor_may_hold_human_approved_authority_level() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        AuthorityBoundary(
            actor_kind=ActorKind.MODEL,
            actor_id="model:planner",
            authority_level=AuthorityLevel.HUMAN_APPROVED,
        )

    assert exc_info.value.failure.mode == FailureMode.SELF_APPROVAL_ATTEMPT


def test_authority_boundary_reports_allowed_permissions_and_forbidden_actions() -> None:
    boundary = _model_boundary(
        allowed_permissions=(
            CognitionPermission.THINK,
            CognitionPermission.PROPOSE_PLAN,
        )
    )

    assert boundary.allows_permission(CognitionPermission.THINK)
    assert boundary.allows_permission(CognitionPermission.PROPOSE_PLAN)
    assert not boundary.allows_permission(CognitionPermission.PREPARE_BLACKFOX_HANDOFF)
    assert boundary.forbids_action(ForbiddenAction.SELF_APPROVE)


def test_authority_snapshot_finds_matching_actor_boundary() -> None:
    boundary = _model_boundary(allowed_permissions=(CognitionPermission.PROPOSE_PLAN,))
    snapshot = _snapshot(boundary)

    assert snapshot.boundary_for(ActorKind.MODEL, "model:planner") == boundary
    assert snapshot.boundary_for(ActorKind.MODEL, "model:critic") is None


def test_firewall_allows_explicit_low_risk_permission() -> None:
    boundary = _model_boundary(allowed_permissions=(CognitionPermission.PROPOSE_PLAN,))
    request = _request(requested_permission=CognitionPermission.PROPOSE_PLAN)

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.allowed
    assert decision.outcome == DecisionOutcome.ALLOW
    assert decision.failure_modes == ()
    assert decision.required_human_review is False


def test_firewall_fails_closed_when_actor_boundary_is_missing() -> None:
    request = _request(requested_permission=CognitionPermission.PROPOSE_PLAN)
    snapshot = AuthoritySnapshot(
        snapshot_id="authority-snapshot:empty",
        boundaries=(),
    )

    decision = AuthorityFirewall().evaluate(request=request, snapshot=snapshot)

    assert decision.failed_closed
    assert decision.outcome == DecisionOutcome.FAIL_CLOSED
    assert decision.failure_modes == (FailureMode.UNCLEAR_AUTHORITY,)
    assert decision.required_human_review is True


def test_firewall_fails_closed_when_permission_is_not_explicitly_allowed() -> None:
    boundary = _model_boundary(allowed_permissions=(CognitionPermission.THINK,))
    request = _request(requested_permission=CognitionPermission.PROPOSE_PLAN)

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.failed_closed
    assert decision.failure_modes == (FailureMode.UNKNOWN_ACTION_TYPE,)
    assert "not explicitly allowed" in decision.reason


def test_firewall_fails_closed_when_required_evidence_is_missing() -> None:
    boundary = _model_boundary(
        allowed_permissions=(CognitionPermission.PREPARE_BLACKFOX_HANDOFF,)
    )
    request = _request(
        requested_permission=CognitionPermission.PREPARE_BLACKFOX_HANDOFF,
        risk_level=RiskLevel.LOW,
    )

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.failed_closed
    assert decision.failure_modes == (FailureMode.MISSING_EVIDENCE,)
    assert decision.required_human_review is True


def test_firewall_requires_human_review_for_high_risk_request() -> None:
    boundary = _model_boundary(allowed_permissions=(CognitionPermission.PROPOSE_PLAN,))
    request = _request(
        requested_permission=CognitionPermission.PROPOSE_PLAN,
        risk_level=RiskLevel.HIGH,
    )

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.outcome == DecisionOutcome.REVIEW_REQUIRED
    assert decision.failure_modes == (FailureMode.HUMAN_REVIEW_REQUIRED,)
    assert decision.required_human_review is True


def test_firewall_allows_high_risk_request_only_with_human_approval() -> None:
    boundary = _model_boundary(allowed_permissions=(CognitionPermission.PROPOSE_PLAN,))
    request = _request(
        requested_permission=CognitionPermission.PROPOSE_PLAN,
        risk_level=RiskLevel.HIGH,
        human_approval_id="human-approval:001",
    )

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.outcome == DecisionOutcome.ALLOW
    assert decision.required_human_review is True


def test_boundary_specific_human_requirement_forces_review() -> None:
    requirement = HumanAuthorityRequirement(
        requirement_id="human-requirement:moderate-plan",
        reason="Moderate-risk planning requires human review.",
        required_for=(CognitionPermission.PROPOSE_PLAN,),
        minimum_risk=RiskLevel.MODERATE,
    )
    boundary = _model_boundary(
        allowed_permissions=(CognitionPermission.PROPOSE_PLAN,),
        human_requirements=(requirement,),
    )
    request = _request(
        requested_permission=CognitionPermission.PROPOSE_PLAN,
        risk_level=RiskLevel.MODERATE,
    )

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.outcome == DecisionOutcome.REVIEW_REQUIRED
    assert decision.failure_modes == (FailureMode.HUMAN_REVIEW_REQUIRED,)


def test_boundary_specific_human_requirement_passes_after_approval() -> None:
    requirement = HumanAuthorityRequirement(
        requirement_id="human-requirement:moderate-plan",
        reason="Moderate-risk planning requires human review.",
        required_for=(CognitionPermission.PROPOSE_PLAN,),
        minimum_risk=RiskLevel.MODERATE,
    )
    boundary = _model_boundary(
        allowed_permissions=(CognitionPermission.PROPOSE_PLAN,),
        human_requirements=(requirement,),
    )
    request = _request(
        requested_permission=CognitionPermission.PROPOSE_PLAN,
        risk_level=RiskLevel.MODERATE,
        human_approval_id="human-approval:002",
    )

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.outcome == DecisionOutcome.ALLOW
    assert decision.required_human_review is True


def test_model_actor_cannot_convert_human_review_request_into_self_authority() -> None:
    boundary = _model_boundary(
        allowed_permissions=(CognitionPermission.REQUEST_HUMAN_REVIEW,)
    )
    request = _request(
        requested_permission=CognitionPermission.REQUEST_HUMAN_REVIEW,
        human_approval_id="model-supplied-human-approval",
    )

    decision = AuthorityFirewall().evaluate(
        request=request,
        snapshot=_snapshot(boundary),
    )

    assert decision.failed_closed
    assert decision.failure_modes == (FailureMode.SELF_APPROVAL_ATTEMPT,)
