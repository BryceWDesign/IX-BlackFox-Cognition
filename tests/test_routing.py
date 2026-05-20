"""Tests for model-role tribunal routing and separation logic."""

import pytest

from ix_blackfox_cognition import (
    CognitionInvariantError,
    DecisionOutcome,
    FailureMode,
    ModelAssignment,
    ModelCapability,
    ModelIdentity,
    ModelProviderKind,
    ModelRoleKind,
    ModelRoleSpec,
    ModelRouteDecision,
    ModelRouteRequest,
    ModelRouter,
    ModelTribunal,
    RiskLevel,
    RoutePurpose,
    RouteSeparationPolicy,
    WorkPackageKind,
    WorkPackageOutputKind,
    route_model_request,
)


def _model_identity(
    model_id: str = "model:planner",
    *,
    provider_kind: ModelProviderKind = ModelProviderKind.FRONTIER_REMOTE,
    provider_name: str = "replaceable-provider",
    model_name: str = "replaceable-model",
) -> ModelIdentity:
    return ModelIdentity(
        model_id=model_id,
        provider_kind=provider_kind,
        provider_name=provider_name,
        model_name=model_name,
    )


def _planner_role(role_id: str = "role:planner") -> ModelRoleSpec:
    return ModelRoleSpec(
        role_id=role_id,
        role_kind=ModelRoleKind.PLANNER,
        description="Generates bounded plans and work package proposals.",
        allowed_package_kinds=(WorkPackageKind.RESEARCH, WorkPackageKind.DESIGN),
        required_capabilities=(ModelCapability.PLANNING,),
        allowed_output_kinds=(WorkPackageOutputKind.CLAIMS, WorkPackageOutputKind.PLAN_UPDATE),
        may_generate=True,
        may_review=False,
    )


def _critic_role(role_id: str = "role:critic") -> ModelRoleSpec:
    return ModelRoleSpec(
        role_id=role_id,
        role_kind=ModelRoleKind.CRITIC,
        description="Reviews and critiques generated work from other roles.",
        allowed_package_kinds=(WorkPackageKind.RESEARCH, WorkPackageKind.DESIGN),
        required_capabilities=(ModelCapability.CRITIQUE,),
        allowed_output_kinds=(WorkPackageOutputKind.REVIEW_RECORD,),
        may_generate=False,
        may_review=True,
    )


def _evidence_reviewer_role(role_id: str = "role:evidence-reviewer") -> ModelRoleSpec:
    return ModelRoleSpec(
        role_id=role_id,
        role_kind=ModelRoleKind.EVIDENCE_REVIEWER,
        description="Reviews evidence references and evidence contracts.",
        allowed_package_kinds=(WorkPackageKind.EVIDENCE_REVIEW,),
        required_capabilities=(ModelCapability.EVIDENCE_REVIEW,),
        allowed_output_kinds=(WorkPackageOutputKind.REVIEW_RECORD,),
        may_generate=False,
        may_review=True,
    )


def _human_authority_role(role_id: str = "role:human-authority") -> ModelRoleSpec:
    return ModelRoleSpec(
        role_id=role_id,
        role_kind=ModelRoleKind.HUMAN_REVIEW_COORDINATOR,
        description="Coordinates explicit human review and authority.",
        allowed_package_kinds=(
            WorkPackageKind.RESEARCH,
            WorkPackageKind.DESIGN,
            WorkPackageKind.EVIDENCE_REVIEW,
            WorkPackageKind.BLACKFOX_HANDOFF,
        ),
        required_capabilities=(ModelCapability.POLICY_REVIEW,),
        allowed_output_kinds=(WorkPackageOutputKind.REVIEW_RECORD,),
        may_generate=False,
        may_review=True,
        may_approve=True,
        human_authority_role=True,
    )


def _assignment(
    assignment_id: str,
    role: ModelRoleSpec,
    model: ModelIdentity,
    *,
    maximum_risk: RiskLevel = RiskLevel.MODERATE,
    enabled: bool = True,
) -> ModelAssignment:
    return ModelAssignment(
        assignment_id=assignment_id,
        role=role,
        model=model,
        maximum_risk=maximum_risk,
        enabled=enabled,
    )


def _tribunal(assignments: tuple[ModelAssignment, ...]) -> ModelTribunal:
    return ModelTribunal(
        tribunal_id="tribunal:test",
        assignments=assignments,
    )


def _route_request(
    *,
    route_request_id: str = "route-request:test",
    package_kind: WorkPackageKind = WorkPackageKind.RESEARCH,
    expected_outputs: tuple[WorkPackageOutputKind, ...] = (WorkPackageOutputKind.CLAIMS,),
    purpose: RoutePurpose = RoutePurpose.GENERATE,
    risk_level: RiskLevel = RiskLevel.LOW,
    originating_role_id: str | None = None,
) -> ModelRouteRequest:
    return ModelRouteRequest(
        route_request_id=route_request_id,
        package_id="package:test",
        package_kind=package_kind,
        expected_outputs=expected_outputs,
        purpose=purpose,
        risk_level=risk_level,
        originating_role_id=originating_role_id,
    )


def test_model_identity_rejects_blank_model_id() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _model_identity(model_id=" ")

    assert exc_info.value.failure.mode == FailureMode.UNCLEAR_AUTHORITY


def test_model_role_requires_nonempty_package_capability_and_output_sets() -> None:
    with pytest.raises(CognitionInvariantError) as no_packages:
        ModelRoleSpec(
            role_id="role:bad",
            role_kind=ModelRoleKind.PLANNER,
            description="Bad role.",
            allowed_package_kinds=(),
            required_capabilities=(ModelCapability.PLANNING,),
            allowed_output_kinds=(WorkPackageOutputKind.CLAIMS,),
        )

    with pytest.raises(CognitionInvariantError) as no_capabilities:
        ModelRoleSpec(
            role_id="role:bad",
            role_kind=ModelRoleKind.PLANNER,
            description="Bad role.",
            allowed_package_kinds=(WorkPackageKind.RESEARCH,),
            required_capabilities=(),
            allowed_output_kinds=(WorkPackageOutputKind.CLAIMS,),
        )

    with pytest.raises(CognitionInvariantError) as no_outputs:
        ModelRoleSpec(
            role_id="role:bad",
            role_kind=ModelRoleKind.PLANNER,
            description="Bad role.",
            allowed_package_kinds=(WorkPackageKind.RESEARCH,),
            required_capabilities=(ModelCapability.PLANNING,),
            allowed_output_kinds=(),
        )

    assert no_packages.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE
    assert no_capabilities.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE
    assert no_outputs.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_nonhuman_role_cannot_approve() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ModelRoleSpec(
            role_id="role:model-approver",
            role_kind=ModelRoleKind.CRITIC,
            description="Model roles cannot approve.",
            allowed_package_kinds=(WorkPackageKind.RESEARCH,),
            required_capabilities=(ModelCapability.CRITIQUE,),
            allowed_output_kinds=(WorkPackageOutputKind.REVIEW_RECORD,),
            may_generate=False,
            may_review=True,
            may_approve=True,
            human_authority_role=False,
        )

    assert exc_info.value.failure.mode == FailureMode.SELF_APPROVAL_ATTEMPT


def test_human_authority_role_cannot_also_generate() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ModelRoleSpec(
            role_id="role:human-generator",
            role_kind=ModelRoleKind.HUMAN_REVIEW_COORDINATOR,
            description="Human authority cannot also be a generation role.",
            allowed_package_kinds=(WorkPackageKind.RESEARCH,),
            required_capabilities=(ModelCapability.POLICY_REVIEW,),
            allowed_output_kinds=(WorkPackageOutputKind.REVIEW_RECORD,),
            may_generate=True,
            may_review=True,
            may_approve=True,
            human_authority_role=True,
        )

    assert exc_info.value.failure.mode == FailureMode.SELF_APPROVAL_ATTEMPT


def test_human_authority_assignment_requires_human_operator_identity() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _assignment(
            "assignment:bad-human",
            _human_authority_role(),
            _model_identity("model:not-human"),
        )

    assert exc_info.value.failure.mode == FailureMode.SELF_APPROVAL_ATTEMPT


def test_human_operator_identity_must_use_human_authority_role() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _assignment(
            "assignment:bad-operator-role",
            _planner_role(),
            _model_identity(
                "human:operator",
                provider_kind=ModelProviderKind.HUMAN_OPERATOR,
                provider_name="human",
                model_name="human-operator",
            ),
        )

    assert exc_info.value.failure.mode == FailureMode.UNCLEAR_AUTHORITY


def test_model_assignment_respects_enabled_state_and_maximum_risk() -> None:
    assignment = _assignment(
        "assignment:planner",
        _planner_role(),
        _model_identity(),
        maximum_risk=RiskLevel.MODERATE,
    )
    disabled = _assignment(
        "assignment:disabled-planner",
        _planner_role("role:disabled-planner"),
        _model_identity("model:disabled-planner"),
        enabled=False,
    )

    assert assignment.can_accept_risk(RiskLevel.LOW) is True
    assert assignment.can_accept_risk(RiskLevel.HIGH) is False
    assert disabled.can_accept_risk(RiskLevel.LOW) is False


def test_model_tribunal_rejects_duplicate_assignment_ids() -> None:
    assignment = _assignment(
        "assignment:duplicate",
        _planner_role(),
        _model_identity(),
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        ModelTribunal(
            tribunal_id="tribunal:duplicates",
            assignments=(assignment, assignment),
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_model_tribunal_rejects_duplicate_role_ids() -> None:
    first = _assignment(
        "assignment:first",
        _planner_role("role:duplicate"),
        _model_identity("model:first"),
    )
    second = _assignment(
        "assignment:second",
        _planner_role("role:duplicate"),
        _model_identity("model:second"),
    )

    with pytest.raises(CognitionInvariantError) as exc_info:
        ModelTribunal(
            tribunal_id="tribunal:duplicate-roles",
            assignments=(first, second),
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_model_tribunal_filters_assignments_by_package_and_risk() -> None:
    low_risk_planner = _assignment(
        "assignment:planner-low",
        _planner_role(),
        _model_identity(),
        maximum_risk=RiskLevel.LOW,
    )
    high_risk_planner = _assignment(
        "assignment:planner-high",
        _planner_role("role:planner-high"),
        _model_identity("model:planner-high"),
        maximum_risk=RiskLevel.HIGH,
    )
    critic = _assignment(
        "assignment:critic",
        _critic_role(),
        _model_identity("model:critic"),
        maximum_risk=RiskLevel.HIGH,
    )
    tribunal = _tribunal((low_risk_planner, high_risk_planner, critic))

    matches = tribunal.assignments_for_package(
        package_kind=WorkPackageKind.RESEARCH,
        risk_level=RiskLevel.HIGH,
    )

    assert matches == (high_risk_planner, critic)


def test_route_decision_requires_selected_ids_when_allowed() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        ModelRouteDecision(
            decision_id="decision:bad",
            route_request_id="route:test",
            outcome=DecisionOutcome.ALLOW,
            reason="Allowed decisions need selected ids.",
        )

    assert exc_info.value.failure.mode == FailureMode.UNCLEAR_AUTHORITY


def test_router_allows_generation_to_planner_role() -> None:
    planner = _assignment(
        "assignment:planner",
        _planner_role(),
        _model_identity(),
    )
    request = _route_request()
    decision = route_model_request(request, _tribunal((planner,)))

    assert decision.allowed
    assert decision.selected_assignment_id == "assignment:planner"
    assert decision.selected_role_id == "role:planner"
    assert decision.selected_model_id == "model:planner"


def test_router_fails_closed_when_no_assignment_can_handle_package_kind() -> None:
    planner = _assignment(
        "assignment:planner",
        _planner_role(),
        _model_identity(),
    )
    request = _route_request(
        package_kind=WorkPackageKind.EVIDENCE_REVIEW,
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
    )

    decision = route_model_request(request, _tribunal((planner,)))

    assert decision.failed_closed
    assert decision.failure_modes == (FailureMode.UNCLEAR_AUTHORITY,)


def test_router_fails_closed_when_no_assignment_supports_expected_outputs() -> None:
    planner = _assignment(
        "assignment:planner",
        _planner_role(),
        _model_identity(),
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.BLACKFOX_ACTION_CANDIDATE,),
    )

    decision = route_model_request(request, _tribunal((planner,)))

    assert decision.failed_closed
    assert decision.failure_modes == (FailureMode.UNKNOWN_ACTION_TYPE,)


def test_router_fails_closed_when_purpose_is_not_supported() -> None:
    planner = _assignment(
        "assignment:planner",
        _planner_role(),
        _model_identity(),
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.CLAIMS,),
        purpose=RoutePurpose.REVIEW,
    )

    decision = route_model_request(request, _tribunal((planner,)))

    assert decision.failed_closed
    assert decision.failure_modes == (FailureMode.UNKNOWN_ACTION_TYPE,)


def test_router_blocks_originating_role_from_reviewing_its_own_work() -> None:
    critic_same_role = _assignment(
        "assignment:critic",
        _critic_role("role:originating"),
        _model_identity("model:critic"),
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.REVIEW,
        originating_role_id="role:originating",
    )

    decision = route_model_request(request, _tribunal((critic_same_role,)))

    assert decision.failed_closed
    assert decision.failure_modes == (FailureMode.SELF_APPROVAL_ATTEMPT,)


def test_router_routes_review_to_different_critic_role() -> None:
    planner = _assignment(
        "assignment:planner",
        _planner_role("role:planner"),
        _model_identity("model:planner"),
    )
    critic = _assignment(
        "assignment:critic",
        _critic_role("role:critic"),
        _model_identity("model:critic"),
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.REVIEW,
        originating_role_id="role:planner",
    )

    decision = route_model_request(request, _tribunal((planner, critic)))

    assert decision.allowed
    assert decision.selected_assignment_id == "assignment:critic"
    assert decision.selected_role_id == "role:critic"


def test_router_requires_human_for_human_escalation() -> None:
    critic = _assignment(
        "assignment:critic",
        _critic_role("role:critic"),
        _model_identity("model:critic"),
        maximum_risk=RiskLevel.HIGH,
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.HUMAN_ESCALATION,
        risk_level=RiskLevel.LOW,
        originating_role_id="role:planner",
    )

    decision = route_model_request(request, _tribunal((critic,)))

    assert decision.outcome == DecisionOutcome.REVIEW_REQUIRED
    assert decision.failure_modes == (FailureMode.HUMAN_REVIEW_REQUIRED,)
    assert decision.requires_human_review is True


def test_router_routes_human_escalation_to_human_authority_assignment() -> None:
    critic = _assignment(
        "assignment:critic",
        _critic_role("role:critic"),
        _model_identity("model:critic"),
        maximum_risk=RiskLevel.HIGH,
    )
    human = _assignment(
        "assignment:human-authority",
        _human_authority_role(),
        _model_identity(
            "human:reviewer",
            provider_kind=ModelProviderKind.HUMAN_OPERATOR,
            provider_name="human",
            model_name="human-reviewer",
        ),
        maximum_risk=RiskLevel.CRITICAL,
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.HUMAN_ESCALATION,
        risk_level=RiskLevel.HIGH,
        originating_role_id="role:planner",
    )

    decision = route_model_request(request, _tribunal((critic, human)))

    assert decision.allowed
    assert decision.selected_assignment_id == "assignment:human-authority"
    assert decision.requires_human_review is True


def test_router_requires_human_review_for_high_risk_non_generation_route() -> None:
    critic = _assignment(
        "assignment:critic",
        _critic_role("role:critic"),
        _model_identity("model:critic"),
        maximum_risk=RiskLevel.CRITICAL,
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.REVIEW,
        risk_level=RiskLevel.HIGH,
        originating_role_id="role:planner",
    )

    decision = route_model_request(request, _tribunal((critic,)))

    assert decision.outcome == DecisionOutcome.REVIEW_REQUIRED
    assert decision.failure_modes == (FailureMode.HUMAN_REVIEW_REQUIRED,)


def test_router_can_relax_high_risk_human_review_by_policy() -> None:
    critic = _assignment(
        "assignment:critic",
        _critic_role("role:critic"),
        _model_identity("model:critic"),
        maximum_risk=RiskLevel.CRITICAL,
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.REVIEW,
        risk_level=RiskLevel.HIGH,
        originating_role_id="role:planner",
    )
    policy = RouteSeparationPolicy(high_risk_requires_human_review=False)

    decision = ModelRouter(policy=policy).route(request, _tribunal((critic,)))

    assert decision.allowed
    assert decision.selected_assignment_id == "assignment:critic"


def test_router_selects_deterministic_assignment_by_assignment_id() -> None:
    second = _assignment(
        "assignment:z-critic",
        _critic_role("role:z-critic"),
        _model_identity("model:z-critic"),
    )
    first = _assignment(
        "assignment:a-critic",
        _critic_role("role:a-critic"),
        _model_identity("model:a-critic"),
    )
    request = _route_request(
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.REVIEW,
        originating_role_id="role:planner",
    )

    decision = route_model_request(request, _tribunal((second, first)))

    assert decision.allowed
    assert decision.selected_assignment_id == "assignment:a-critic"


def test_evidence_reviewer_route_supports_evidence_review_package() -> None:
    evidence_reviewer = _assignment(
        "assignment:evidence-reviewer",
        _evidence_reviewer_role(),
        _model_identity("model:evidence-reviewer"),
    )
    request = _route_request(
        package_kind=WorkPackageKind.EVIDENCE_REVIEW,
        expected_outputs=(WorkPackageOutputKind.REVIEW_RECORD,),
        purpose=RoutePurpose.EVIDENCE_CHECK,
        risk_level=RiskLevel.LOW,
        originating_role_id="role:planner",
    )

    decision = route_model_request(request, _tribunal((evidence_reviewer,)))

    assert decision.allowed
    assert decision.selected_assignment_id == "assignment:evidence-reviewer"
