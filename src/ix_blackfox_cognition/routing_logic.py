"""Model-role separation and routing logic for IX-BlackFox-Cognition.

The routing layer enforces the tribunal rule:

No model or role may approve its own work.

Routing is deterministic and conservative. If a route cannot satisfy package
kind, output kind, risk, purpose, role separation, and human-authority
requirements, it fails closed instead of selecting a weak route.
"""

from __future__ import annotations

from dataclasses import dataclass

from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, RiskLevel
from ix_blackfox_cognition.routing import (
    ModelAssignment,
    ModelProviderKind,
    ModelRouteDecision,
    ModelRouteRequest,
    ModelTribunal,
    RoutePurpose,
)


@dataclass(frozen=True, slots=True)
class RouteSeparationPolicy:
    """Policy controlling role separation and self-approval prevention."""

    policy_id: str = "default-route-separation-policy"
    require_different_role_for_review: bool = True
    require_human_for_approval: bool = True
    require_human_for_human_escalation: bool = True
    block_originating_role_for_review: bool = True
    block_disabled_assignments: bool = True
    high_risk_requires_human_review: bool = True
    high_risk_threshold: RiskLevel = RiskLevel.HIGH


class ModelRouter:
    """Conservative router for model-role tribunal assignments."""

    def __init__(self, policy: RouteSeparationPolicy | None = None) -> None:
        self.policy = policy or RouteSeparationPolicy()

    def route(
        self,
        request: ModelRouteRequest,
        tribunal: ModelTribunal,
    ) -> ModelRouteDecision:
        """Route a work package request to a separated role, or fail closed."""

        candidates = tribunal.assignments_for_package(
            package_kind=request.package_kind,
            risk_level=request.risk_level,
        )

        if self.policy.block_disabled_assignments:
            candidates = tuple(candidate for candidate in candidates if candidate.enabled)

        if not candidates:
            return self._fail_closed(
                request=request,
                reason="No enabled tribunal assignment can handle this package kind and risk level.",
                failure_modes=(FailureMode.UNCLEAR_AUTHORITY,),
            )

        candidates = tuple(
            candidate for candidate in candidates if self._supports_expected_outputs(candidate, request)
        )
        if not candidates:
            return self._fail_closed(
                request=request,
                reason="No tribunal assignment supports all expected output kinds.",
                failure_modes=(FailureMode.UNKNOWN_ACTION_TYPE,),
            )

        candidates = tuple(
            candidate for candidate in candidates if self._supports_purpose(candidate, request)
        )
        if not candidates:
            return self._fail_closed(
                request=request,
                reason="No tribunal assignment supports the requested route purpose.",
                failure_modes=(FailureMode.UNKNOWN_ACTION_TYPE,),
            )

        candidates = tuple(
            candidate for candidate in candidates if self._passes_role_separation(candidate, request)
        )
        if not candidates:
            return self._fail_closed(
                request=request,
                reason="Route would violate model-role separation or self-review rules.",
                failure_modes=(FailureMode.SELF_APPROVAL_ATTEMPT,),
            )

        if self._requires_human_escalation(request):
            human_candidates = tuple(candidate for candidate in candidates if self._is_human(candidate))
            if not human_candidates:
                return self._review_required(
                    request=request,
                    reason="Requested route requires explicit human authority.",
                )
            candidates = human_candidates

        if self._high_risk_requires_human_review(request):
            human_candidates = tuple(candidate for candidate in candidates if self._is_human(candidate))
            if not human_candidates:
                return self._review_required(
                    request=request,
                    reason="High-risk route requires human review before model routing can proceed.",
                )
            candidates = human_candidates

        selected = self._select_candidate(candidates)

        return ModelRouteDecision(
            decision_id=self._decision_id(request),
            route_request_id=request.route_request_id,
            outcome=DecisionOutcome.ALLOW,
            reason="Route selected under model-role separation policy.",
            selected_assignment_id=selected.assignment_id,
            selected_role_id=selected.role.role_id,
            selected_model_id=selected.model.model_id,
            requires_human_review=selected.role.human_authority_role,
        )

    def _supports_expected_outputs(
        self,
        candidate: ModelAssignment,
        request: ModelRouteRequest,
    ) -> bool:
        return all(
            candidate.role.can_emit_output_kind(output_kind)
            for output_kind in request.expected_outputs
        )

    def _supports_purpose(
        self,
        candidate: ModelAssignment,
        request: ModelRouteRequest,
    ) -> bool:
        if request.purpose == RoutePurpose.GENERATE:
            return candidate.role.may_generate

        if request.purpose in (
            RoutePurpose.REVIEW,
            RoutePurpose.CRITIQUE,
            RoutePurpose.RED_TEAM,
            RoutePurpose.EVIDENCE_CHECK,
            RoutePurpose.POLICY_CHECK,
            RoutePurpose.MEMORY_CHECK,
        ):
            return candidate.role.may_review

        if request.purpose == RoutePurpose.HUMAN_ESCALATION:
            return candidate.role.may_review or candidate.role.human_authority_role

        return False

    def _passes_role_separation(
        self,
        candidate: ModelAssignment,
        request: ModelRouteRequest,
    ) -> bool:
        if request.originating_role_id is None:
            return True

        if not self.policy.require_different_role_for_review:
            return True

        if request.purpose == RoutePurpose.GENERATE:
            return True

        if self.policy.block_originating_role_for_review:
            return candidate.role.role_id != request.originating_role_id

        return True

    def _requires_human_escalation(self, request: ModelRouteRequest) -> bool:
        return (
            self.policy.require_human_for_human_escalation
            and request.purpose == RoutePurpose.HUMAN_ESCALATION
        )

    def _high_risk_requires_human_review(self, request: ModelRouteRequest) -> bool:
        return (
            self.policy.high_risk_requires_human_review
            and request.risk_level.rank() >= self.policy.high_risk_threshold.rank()
            and request.purpose != RoutePurpose.GENERATE
        )

    def _is_human(self, candidate: ModelAssignment) -> bool:
        return (
            candidate.role.human_authority_role
            and candidate.model.provider_kind == ModelProviderKind.HUMAN_OPERATOR
        )

    def _select_candidate(self, candidates: tuple[ModelAssignment, ...]) -> ModelAssignment:
        return sorted(candidates, key=lambda candidate: candidate.assignment_id)[0]

    def _review_required(self, request: ModelRouteRequest, reason: str) -> ModelRouteDecision:
        return ModelRouteDecision(
            decision_id=self._decision_id(request),
            route_request_id=request.route_request_id,
            outcome=DecisionOutcome.REVIEW_REQUIRED,
            reason=reason,
            failure_modes=(FailureMode.HUMAN_REVIEW_REQUIRED,),
            requires_human_review=True,
        )

    def _fail_closed(
        self,
        request: ModelRouteRequest,
        reason: str,
        failure_modes: tuple[FailureMode, ...],
    ) -> ModelRouteDecision:
        return ModelRouteDecision(
            decision_id=self._decision_id(request),
            route_request_id=request.route_request_id,
            outcome=DecisionOutcome.FAIL_CLOSED,
            reason=reason,
            failure_modes=failure_modes,
            requires_human_review=True,
        )

    def _decision_id(self, request: ModelRouteRequest) -> str:
        return f"model-route-decision:{request.route_request_id}"


def route_model_request(
    request: ModelRouteRequest,
    tribunal: ModelTribunal,
    policy: RouteSeparationPolicy | None = None,
) -> ModelRouteDecision:
    """Route a request through the default conservative model router."""

    return ModelRouter(policy=policy).route(request=request, tribunal=tribunal)
