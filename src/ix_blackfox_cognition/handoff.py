"""BlackFox-compatible handoff protocol for IX-BlackFox-Cognition.

IX-BlackFox-Cognition does not directly execute risky action. It prepares
reviewable action candidates for IX-BlackFox-style governance.

The handoff protocol preserves the core doctrine:

Model thinks → Cognition structures → BlackFox governs → humans authorize
→ evidence decides trust.

A handoff candidate must carry scope, risk, evidence contracts, required tests,
rollback plans, sentinel review, and human authority requirements before it can
be considered ready for governed execution review.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, RiskLevel, WorkState, require_invariant
from ix_blackfox_cognition.evidence import EvidenceContract
from ix_blackfox_cognition.sentinel import CognitiveSentinelReport


class BlackFoxActionKind(StrEnum):
    """Kinds of BlackFox-compatible action candidates."""

    PATCH_CANDIDATE = "patch_candidate"
    TEST_REQUEST = "test_request"
    VALIDATION_REQUEST = "validation_request"
    EVIDENCE_REVIEW = "evidence_review"
    MEMORY_REVIEW = "memory_review"
    SELF_IMPROVEMENT_REVIEW = "self_improvement_review"
    DOCUMENTATION_REVIEW = "documentation_review"
    POLICY_REVIEW = "policy_review"


class HandoffReadiness(StrEnum):
    """Readiness states for BlackFox handoff envelopes."""

    DRAFT = "draft"
    BLOCKED = "blocked"
    READY_FOR_REVIEW = "ready_for_review"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    APPROVED_FOR_BLACKFOX_REVIEW = "approved_for_blackfox_review"


@dataclass(frozen=True, slots=True)
class RollbackPlan:
    """Rollback plan required before action-adjacent handoff."""

    rollback_id: str
    statement: str
    required_artifacts: tuple[str, ...]
    human_review_required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.rollback_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Rollback plan id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Rollback plan statement cannot be blank.",
        )
        require_invariant(
            bool(self.required_artifacts),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Rollback plan must identify required artifacts.",
        )


@dataclass(frozen=True, slots=True)
class RequiredTest:
    """Allowlisted test or validation command expected before trust."""

    test_id: str
    command: str
    reason: str
    allowlisted: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.test_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Required test id cannot be blank.",
        )
        require_invariant(
            bool(self.command.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Required test command cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Required test reason cannot be blank.",
        )
        require_invariant(
            self.allowlisted,
            FailureMode.POLICY_BYPASS_ATTEMPT,
            "BlackFox handoff required tests must be allowlisted.",
        )


@dataclass(frozen=True, slots=True)
class BlackFoxActionCandidate:
    """Action candidate prepared by cognition for BlackFox-style governance."""

    candidate_id: str
    action_kind: BlackFoxActionKind
    objective: str
    scope: str
    risk_level: RiskLevel
    source_package_id: str
    target_paths: tuple[str, ...] = field(default_factory=tuple)
    required_tests: tuple[RequiredTest, ...] = field(default_factory=tuple)
    evidence_contract: EvidenceContract | None = None
    rollback_plan: RollbackPlan | None = None
    forbidden_actions: tuple[str, ...] = field(default_factory=tuple)
    human_review_required: bool = True
    expected_receipt_chain: tuple[str, ...] = field(default_factory=tuple)
    notes: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.candidate_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox action candidate id cannot be blank.",
        )
        require_invariant(
            bool(self.objective.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox action candidate objective cannot be blank.",
        )
        require_invariant(
            bool(self.scope.strip()),
            FailureMode.SCOPE_CREEP,
            "BlackFox action candidate scope cannot be blank.",
        )
        require_invariant(
            bool(self.source_package_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox action candidate source package id cannot be blank.",
        )
        require_invariant(
            bool(self.forbidden_actions),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox action candidate must preserve forbidden actions.",
        )
        require_invariant(
            self.evidence_contract is not None,
            FailureMode.MISSING_EVIDENCE,
            "BlackFox action candidate requires an evidence contract.",
        )
        require_invariant(
            self.rollback_plan is not None,
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox action candidate requires a rollback plan.",
        )

        if self.action_kind in (
            BlackFoxActionKind.PATCH_CANDIDATE,
            BlackFoxActionKind.TEST_REQUEST,
            BlackFoxActionKind.VALIDATION_REQUEST,
        ):
            require_invariant(
                bool(self.required_tests),
                FailureMode.UNKNOWN_ACTION_TYPE,
                "Patch, test, and validation candidates require allowlisted tests.",
            )

        if self.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            require_invariant(
                self.human_review_required,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "High-risk BlackFox action candidates require human review.",
            )

    @property
    def evidence_satisfied(self) -> bool:
        """Return whether the candidate evidence contract is satisfied."""

        return self.evidence_contract is not None and self.evidence_contract.trust_eligible

    @property
    def action_adjacent(self) -> bool:
        """Return whether this candidate is close to executable action."""

        return self.action_kind in (
            BlackFoxActionKind.PATCH_CANDIDATE,
            BlackFoxActionKind.TEST_REQUEST,
            BlackFoxActionKind.VALIDATION_REQUEST,
            BlackFoxActionKind.SELF_IMPROVEMENT_REVIEW,
        )


@dataclass(frozen=True, slots=True)
class BlackFoxHandoffEnvelope:
    """Reviewable handoff envelope sent toward IX-BlackFox-style governance."""

    envelope_id: str
    mission_id: str
    plan_graph_id: str
    candidate: BlackFoxActionCandidate
    sentinel_report: CognitiveSentinelReport
    status: HandoffReadiness = HandoffReadiness.DRAFT
    human_approval_id: str | None = None
    receipt_expectations: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.envelope_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox handoff envelope id cannot be blank.",
        )
        require_invariant(
            bool(self.mission_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox handoff envelope mission id cannot be blank.",
        )
        require_invariant(
            bool(self.plan_graph_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox handoff envelope plan graph id cannot be blank.",
        )
        require_invariant(
            bool(self.receipt_expectations),
            FailureMode.MISSING_EVIDENCE,
            "BlackFox handoff envelope must define receipt expectations.",
        )

        if self.status == HandoffReadiness.APPROVED_FOR_BLACKFOX_REVIEW:
            require_invariant(
                self.has_human_approval,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "Approved BlackFox handoff envelopes require human approval.",
            )

    @property
    def has_human_approval(self) -> bool:
        """Return whether the envelope has explicit human approval."""

        return self.human_approval_id is not None and bool(self.human_approval_id.strip())

    @property
    def blocked(self) -> bool:
        """Return whether this envelope is blocked from BlackFox review."""

        return self.status == HandoffReadiness.BLOCKED

    @property
    def ready_for_review(self) -> bool:
        """Return whether this envelope is ready for human or BlackFox review."""

        return self.status in (
            HandoffReadiness.READY_FOR_REVIEW,
            HandoffReadiness.HUMAN_REVIEW_REQUIRED,
            HandoffReadiness.APPROVED_FOR_BLACKFOX_REVIEW,
        )


@dataclass(frozen=True, slots=True)
class BlackFoxHandoffDecision:
    """Decision produced by handoff readiness evaluation."""

    decision_id: str
    envelope_id: str
    outcome: DecisionOutcome
    readiness: HandoffReadiness
    reason: str
    failure_modes: tuple[FailureMode, ...] = field(default_factory=tuple)
    requires_human_review: bool = True

    @property
    def allowed(self) -> bool:
        """Return whether the handoff envelope is allowed to proceed to review."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def failed_closed(self) -> bool:
        """Return whether the handoff envelope failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED


@dataclass(frozen=True, slots=True)
class BlackFoxHandoffResult:
    """Result of evaluating a BlackFox handoff envelope."""

    envelope: BlackFoxHandoffEnvelope
    decision: BlackFoxHandoffDecision


class BlackFoxHandoffGate:
    """Conservative gate for BlackFox-compatible handoff envelopes."""

    def evaluate(self, envelope: BlackFoxHandoffEnvelope) -> BlackFoxHandoffResult:
        """Evaluate whether a handoff envelope is ready for governed review."""

        if envelope.sentinel_report.blocked:
            return self._fail_closed(
                envelope=envelope,
                reason="Cognitive sentinel report blocks this handoff.",
                failure_modes=tuple(issue.failure_mode for issue in envelope.sentinel_report.blocker_issues),
            )

        if not envelope.candidate.evidence_satisfied:
            return self._fail_closed(
                envelope=envelope,
                reason="BlackFox action candidate evidence contract is not satisfied.",
                failure_modes=(FailureMode.MISSING_EVIDENCE,),
            )

        if envelope.candidate.human_review_required and not envelope.has_human_approval:
            updated = replace(envelope, status=HandoffReadiness.HUMAN_REVIEW_REQUIRED)
            return BlackFoxHandoffResult(
                envelope=updated,
                decision=BlackFoxHandoffDecision(
                    decision_id=self._decision_id(envelope),
                    envelope_id=envelope.envelope_id,
                    outcome=DecisionOutcome.REVIEW_REQUIRED,
                    readiness=HandoffReadiness.HUMAN_REVIEW_REQUIRED,
                    reason="BlackFox handoff requires explicit human approval.",
                    failure_modes=(FailureMode.HUMAN_REVIEW_REQUIRED,),
                    requires_human_review=True,
                ),
            )

        updated = replace(
            envelope,
            status=HandoffReadiness.APPROVED_FOR_BLACKFOX_REVIEW
            if envelope.has_human_approval
            else HandoffReadiness.READY_FOR_REVIEW,
        )
        return BlackFoxHandoffResult(
            envelope=updated,
            decision=BlackFoxHandoffDecision(
                decision_id=self._decision_id(envelope),
                envelope_id=envelope.envelope_id,
                outcome=DecisionOutcome.ALLOW,
                readiness=updated.status,
                reason="BlackFox handoff envelope is ready for governed BlackFox review.",
                requires_human_review=envelope.candidate.human_review_required,
            ),
        )

    def _fail_closed(
        self,
        envelope: BlackFoxHandoffEnvelope,
        reason: str,
        failure_modes: tuple[FailureMode, ...],
    ) -> BlackFoxHandoffResult:
        updated = replace(envelope, status=HandoffReadiness.BLOCKED)
        return BlackFoxHandoffResult(
            envelope=updated,
            decision=BlackFoxHandoffDecision(
                decision_id=self._decision_id(envelope),
                envelope_id=envelope.envelope_id,
                outcome=DecisionOutcome.FAIL_CLOSED,
                readiness=HandoffReadiness.BLOCKED,
                reason=reason,
                failure_modes=self._dedupe_failure_modes(failure_modes),
                requires_human_review=True,
            ),
        )

    def _decision_id(self, envelope: BlackFoxHandoffEnvelope) -> str:
        return f"blackfox-handoff-decision:{envelope.envelope_id}"

    def _dedupe_failure_modes(self, modes: tuple[FailureMode, ...]) -> tuple[FailureMode, ...]:
        deduped: list[FailureMode] = []
        for mode in modes:
            if mode not in deduped:
                deduped.append(mode)
        return tuple(deduped)


def evaluate_blackfox_handoff(envelope: BlackFoxHandoffEnvelope) -> BlackFoxHandoffResult:
    """Evaluate a BlackFox handoff envelope with the default handoff gate."""

    return BlackFoxHandoffGate().evaluate(envelope)


@dataclass(frozen=True, slots=True)
class BlackFoxExecutionRequest:
    """Explicit request object for a future BlackFox execution control plane.

    This is still only a request. It is not execution authority.
    """

    request_id: str
    envelope: BlackFoxHandoffEnvelope
    requested_state: WorkState = WorkState.READY_FOR_REVIEW

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.request_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "BlackFox execution request id cannot be blank.",
        )
        require_invariant(
            self.envelope.status == HandoffReadiness.APPROVED_FOR_BLACKFOX_REVIEW,
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "BlackFox execution requests require an approved handoff envelope.",
        )
        require_invariant(
            self.envelope.has_human_approval,
            FailureMode.HUMAN_REVIEW_REQUIRED,
            "BlackFox execution requests require explicit human approval.",
        )
