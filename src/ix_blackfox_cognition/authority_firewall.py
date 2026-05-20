"""Fail-closed authority firewall for IX-BlackFox-Cognition.

The authority firewall turns authority requests and authority snapshots into
reviewable decisions. It does not execute work. It decides whether cognition may
continue, must request human review, or must fail closed.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ix_blackfox_cognition.authority import (
    AuthorityBoundary,
    AuthorityDecisionRecord,
    AuthorityRequest,
    AuthoritySnapshot,
    CognitionPermission,
)
from ix_blackfox_cognition.core import (
    ActorKind,
    DecisionOutcome,
    FailureMode,
    RiskLevel,
    require_invariant,
)


@dataclass(frozen=True, slots=True)
class AuthorityFirewallPolicy:
    """Policy knobs for authority firewall decisions."""

    policy_id: str = "default-authority-firewall-policy"
    high_risk_requires_human_review: bool = True
    minimum_human_review_risk: RiskLevel = RiskLevel.HIGH
    evidence_required_permissions: tuple[CognitionPermission, ...] = field(
        default_factory=lambda: (
            CognitionPermission.RECORD_EVIDENCE_REFERENCE,
            CognitionPermission.PREPARE_BLACKFOX_HANDOFF,
        )
    )
    review_required_permissions: tuple[CognitionPermission, ...] = field(
        default_factory=lambda: (
            CognitionPermission.PROPOSE_SELF_IMPROVEMENT,
            CognitionPermission.PREPARE_BLACKFOX_HANDOFF,
        )
    )

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.policy_id.strip()),
            FailureMode.MISSING_POLICY,
            "Authority firewall policy id cannot be blank.",
        )


class AuthorityFirewall:
    """Fail-closed authority firewall.

    The firewall preserves the IX-BlackFox-Cognition doctrine:

    Model thinks → Cognition structures → BlackFox governs → humans authorize
    → evidence decides trust.
    """

    def __init__(self, policy: AuthorityFirewallPolicy | None = None) -> None:
        self.policy = policy or AuthorityFirewallPolicy()

    def evaluate(
        self,
        request: AuthorityRequest,
        snapshot: AuthoritySnapshot,
    ) -> AuthorityDecisionRecord:
        """Evaluate an authority request against a frozen authority snapshot."""

        boundary = snapshot.boundary_for(
            actor_kind=request.actor_kind,
            actor_id=request.actor_id,
        )

        if boundary is None:
            return self._fail_closed(
                request=request,
                reason="No authority boundary exists for this actor.",
                failure_modes=(FailureMode.UNCLEAR_AUTHORITY,),
            )

        if boundary.authority_level.value == "blocked":
            return self._fail_closed(
                request=request,
                reason="Actor authority level is blocked.",
                failure_modes=(FailureMode.UNCLEAR_AUTHORITY,),
            )

        if not boundary.allows_permission(request.requested_permission):
            return self._fail_closed(
                request=request,
                reason="Requested permission is not explicitly allowed.",
                failure_modes=(FailureMode.UNKNOWN_ACTION_TYPE,),
            )

        if self._requires_evidence(request) and not request.evidence_ids:
            return self._fail_closed(
                request=request,
                reason="Requested permission requires evidence references.",
                failure_modes=(FailureMode.MISSING_EVIDENCE,),
            )

        if self._boundary_requirement_requires_review(boundary, request):
            if not request.has_human_approval:
                return self._review_required(
                    request=request,
                    reason="Authority boundary requires explicit human review.",
                )

        if self._policy_requires_review(request):
            if not request.has_human_approval:
                return self._review_required(
                    request=request,
                    reason="Firewall policy requires explicit human review.",
                )

        if self._model_attempts_human_authority(request):
            return self._fail_closed(
                request=request,
                reason="A model actor cannot be treated as human authority.",
                failure_modes=(FailureMode.SELF_APPROVAL_ATTEMPT,),
            )

        return AuthorityDecisionRecord(
            decision_id=self._decision_id(request),
            request_id=request.request_id,
            outcome=DecisionOutcome.ALLOW,
            actor_kind=request.actor_kind,
            reason="Requested cognition permission is explicitly allowed.",
            risk_level=request.risk_level,
            required_human_review=self._human_review_was_required(request, boundary),
        )

    def _policy_requires_review(self, request: AuthorityRequest) -> bool:
        if request.requested_permission in self.policy.review_required_permissions:
            return True

        if not self.policy.high_risk_requires_human_review:
            return False

        return request.risk_level.rank() >= self.policy.minimum_human_review_risk.rank()

    def _requires_evidence(self, request: AuthorityRequest) -> bool:
        return request.requested_permission in self.policy.evidence_required_permissions

    def _boundary_requirement_requires_review(
        self,
        boundary: AuthorityBoundary,
        request: AuthorityRequest,
    ) -> bool:
        for requirement in boundary.human_requirements:
            if request.risk_level.rank() < requirement.minimum_risk.rank():
                continue

            covered_values = {
                covered.value if hasattr(covered, "value") else str(covered)
                for covered in requirement.required_for
            }
            if request.requested_permission.value in covered_values:
                return requirement.explicit_approval_required

        return False

    def _human_review_was_required(
        self,
        request: AuthorityRequest,
        boundary: AuthorityBoundary,
    ) -> bool:
        return self._policy_requires_review(request) or self._boundary_requirement_requires_review(
            boundary=boundary,
            request=request,
        )

    def _model_attempts_human_authority(self, request: AuthorityRequest) -> bool:
        return (
            request.actor_kind == ActorKind.MODEL
            and request.requested_permission == CognitionPermission.REQUEST_HUMAN_REVIEW
            and request.has_human_approval
        )

    def _review_required(self, request: AuthorityRequest, reason: str) -> AuthorityDecisionRecord:
        return AuthorityDecisionRecord(
            decision_id=self._decision_id(request),
            request_id=request.request_id,
            outcome=DecisionOutcome.REVIEW_REQUIRED,
            actor_kind=request.actor_kind,
            reason=reason,
            risk_level=request.risk_level,
            failure_modes=(FailureMode.HUMAN_REVIEW_REQUIRED,),
            required_human_review=True,
        )

    def _fail_closed(
        self,
        request: AuthorityRequest,
        reason: str,
        failure_modes: tuple[FailureMode, ...],
    ) -> AuthorityDecisionRecord:
        return AuthorityDecisionRecord(
            decision_id=self._decision_id(request),
            request_id=request.request_id,
            outcome=DecisionOutcome.FAIL_CLOSED,
            actor_kind=request.actor_kind,
            reason=reason,
            risk_level=request.risk_level,
            failure_modes=failure_modes,
            required_human_review=True,
        )

    def _decision_id(self, request: AuthorityRequest) -> str:
        return f"authority-decision:{request.request_id}"


def evaluate_authority_request(
    request: AuthorityRequest,
    snapshot: AuthoritySnapshot,
    policy: AuthorityFirewallPolicy | None = None,
) -> AuthorityDecisionRecord:
    """Evaluate an authority request with a one-shot authority firewall."""

    return AuthorityFirewall(policy=policy).evaluate(request=request, snapshot=snapshot)
