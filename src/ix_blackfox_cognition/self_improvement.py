"""Self-improvement airlock for IX-BlackFox-Cognition.

IX-BlackFox-Cognition may propose improvements to its own planning rules,
memory schemas, routing rules, sentinel checks, evidence requirements, policy
checks, work package templates, and BlackFox handoff templates.

It may not promote or apply those improvements by itself.

Required chain:

proposal → risk classification → test plan → adversarial review → evidence
bundle → human approval → versioned promotion → rollback path

The airlock preserves the core rule:

The system may propose self-improvement.
It may not self-promote.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, RiskLevel, WorkState
from ix_blackfox_cognition.evidence import EvidenceContract
from ix_blackfox_cognition.handoff import RollbackPlan


class SelfImprovementTargetKind(StrEnum):
    """Kinds of cognition substrate components that may receive improvement proposals."""

    AUTHORITY_RULE = "authority_rule"
    MISSION_INTAKE_RULE = "mission_intake_rule"
    EPISTEMIC_SCHEMA = "epistemic_schema"
    BELIEF_GRAPH_RULE = "belief_graph_rule"
    PLAN_GRAPH_RULE = "plan_graph_rule"
    WORK_PACKAGE_TEMPLATE = "work_package_template"
    MODEL_ROUTING_RULE = "model_routing_rule"
    MEMORY_SCHEMA = "memory_schema"
    MEMORY_PROMOTION_RULE = "memory_promotion_rule"
    SENTINEL_CHECK = "sentinel_check"
    EVIDENCE_REQUIREMENT = "evidence_requirement"
    BLACKFOX_HANDOFF_TEMPLATE = "blackfox_handoff_template"
    POLICY_FILE = "policy_file"
    TEST_SUITE = "test_suite"
    DOCUMENTATION = "documentation"


class SelfImprovementProposalState(StrEnum):
    """Lifecycle states for self-improvement proposals."""

    DRAFT = "draft"
    PROPOSED = "proposed"
    QUARANTINED = "quarantined"
    READY_FOR_REVIEW = "ready_for_review"
    HUMAN_REVIEW_REQUIRED = "human_review_required"
    APPROVED_FOR_HANDOFF = "approved_for_handoff"
    REJECTED = "rejected"
    BLOCKED = "blocked"


@dataclass(frozen=True, slots=True)
class SelfImprovementRisk:
    """Risk classification for a self-improvement proposal."""

    risk_id: str
    level: RiskLevel
    statement: str
    affected_invariants: tuple[str, ...]
    mitigation: str

    def __post_init__(self) -> None:
        if not self.risk_id.strip():
            raise ValueError("Self-improvement risk id cannot be blank.")
        if not self.statement.strip():
            raise ValueError("Self-improvement risk statement cannot be blank.")
        if not self.affected_invariants:
            raise ValueError("Self-improvement risk must identify affected invariants.")
        if not self.mitigation.strip():
            raise ValueError("Self-improvement risk mitigation cannot be blank.")


@dataclass(frozen=True, slots=True)
class SelfImprovementTestPlan:
    """Required test plan for a self-improvement proposal."""

    test_plan_id: str
    required_tests: tuple[str, ...]
    adversarial_scenarios: tuple[str, ...]
    acceptance_criteria: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.test_plan_id.strip():
            raise ValueError("Self-improvement test plan id cannot be blank.")
        if not self.required_tests:
            raise ValueError("Self-improvement test plan must define required tests.")
        if not self.adversarial_scenarios:
            raise ValueError("Self-improvement test plan must define adversarial scenarios.")
        if not self.acceptance_criteria:
            raise ValueError("Self-improvement test plan must define acceptance criteria.")


@dataclass(frozen=True, slots=True)
class SelfImprovementProposal:
    """Proposal to improve the cognition substrate.

    This object is only a proposal. It is not authority to change code, policy,
    memory, routing, sentinel behavior, or evidence requirements.
    """

    proposal_id: str
    target_kind: SelfImprovementTargetKind
    title: str
    summary: str
    rationale: str
    proposed_change: str
    proposer_role_id: str
    state: SelfImprovementProposalState = SelfImprovementProposalState.DRAFT
    risk: SelfImprovementRisk | None = None
    test_plan: SelfImprovementTestPlan | None = None
    evidence_contract: EvidenceContract | None = None
    rollback_plan: RollbackPlan | None = None
    human_approval_id: str | None = None
    blocked_failure_modes: tuple[FailureMode, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not self.proposal_id.strip():
            raise ValueError("Self-improvement proposal id cannot be blank.")
        if not self.title.strip():
            raise ValueError("Self-improvement proposal title cannot be blank.")
        if not self.summary.strip():
            raise ValueError("Self-improvement proposal summary cannot be blank.")
        if not self.rationale.strip():
            raise ValueError("Self-improvement proposal rationale cannot be blank.")
        if not self.proposed_change.strip():
            raise ValueError("Self-improvement proposed change cannot be blank.")
        if not self.proposer_role_id.strip():
            raise ValueError("Self-improvement proposer role id cannot be blank.")

        if self.state == SelfImprovementProposalState.APPROVED_FOR_HANDOFF:
            if not self.has_human_approval:
                raise ValueError("Approved self-improvement proposals require human approval.")
            if self.evidence_contract is None or not self.evidence_contract.trust_eligible:
                raise ValueError("Approved self-improvement proposals require satisfied evidence.")
            if self.rollback_plan is None:
                raise ValueError("Approved self-improvement proposals require rollback plan.")

    @property
    def has_human_approval(self) -> bool:
        """Return whether the proposal has explicit human approval."""

        return self.human_approval_id is not None and bool(self.human_approval_id.strip())

    @property
    def has_required_airlock_inputs(self) -> bool:
        """Return whether the proposal has the minimum airlock review inputs."""

        return (
            self.risk is not None
            and self.test_plan is not None
            and self.evidence_contract is not None
            and self.rollback_plan is not None
        )

    @property
    def risk_level(self) -> RiskLevel:
        """Return the proposal risk level, defaulting to critical if unclassified."""

        if self.risk is None:
            return RiskLevel.CRITICAL
        return self.risk.level


@dataclass(frozen=True, slots=True)
class SelfImprovementAirlockPolicy:
    """Policy for self-improvement airlock evaluation."""

    policy_id: str = "default-self-improvement-airlock-policy"
    require_risk_classification: bool = True
    require_test_plan: bool = True
    require_adversarial_scenarios: bool = True
    require_evidence_contract: bool = True
    require_satisfied_evidence: bool = True
    require_rollback_plan: bool = True
    require_human_approval: bool = True
    block_policy_file_self_promotion: bool = True

    def __post_init__(self) -> None:
        if not self.policy_id.strip():
            raise ValueError("Self-improvement airlock policy id cannot be blank.")


@dataclass(frozen=True, slots=True)
class SelfImprovementDecision:
    """Reviewable decision from the self-improvement airlock."""

    decision_id: str
    proposal_id: str
    outcome: DecisionOutcome
    state: SelfImprovementProposalState
    reason: str
    failure_modes: tuple[FailureMode, ...] = field(default_factory=tuple)
    requires_human_review: bool = True

    @property
    def allowed(self) -> bool:
        """Return whether the self-improvement proposal may move forward."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def failed_closed(self) -> bool:
        """Return whether the self-improvement proposal failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED


@dataclass(frozen=True, slots=True)
class SelfImprovementAirlockResult:
    """Result of evaluating a self-improvement proposal."""

    proposal: SelfImprovementProposal
    decision: SelfImprovementDecision


class SelfImprovementAirlock:
    """Conservative self-improvement airlock.

    The airlock can approve a proposal for later BlackFox-compatible review, but
    it cannot apply the improvement and cannot promote itself.
    """

    def __init__(self, policy: SelfImprovementAirlockPolicy | None = None) -> None:
        self.policy = policy or SelfImprovementAirlockPolicy()

    def evaluate(self, proposal: SelfImprovementProposal) -> SelfImprovementAirlockResult:
        """Evaluate a self-improvement proposal against the airlock policy."""

        failures = self._failure_modes(proposal)
        if failures:
            blocked = replace(
                proposal,
                state=SelfImprovementProposalState.BLOCKED,
                blocked_failure_modes=failures,
            )
            return SelfImprovementAirlockResult(
                proposal=blocked,
                decision=SelfImprovementDecision(
                    decision_id=self._decision_id(proposal),
                    proposal_id=proposal.proposal_id,
                    outcome=DecisionOutcome.FAIL_CLOSED,
                    state=SelfImprovementProposalState.BLOCKED,
                    reason="Self-improvement proposal failed the airlock.",
                    failure_modes=failures,
                    requires_human_review=True,
                ),
            )

        if self.policy.require_human_approval and not proposal.has_human_approval:
            review_required = replace(
                proposal,
                state=SelfImprovementProposalState.HUMAN_REVIEW_REQUIRED,
            )
            return SelfImprovementAirlockResult(
                proposal=review_required,
                decision=SelfImprovementDecision(
                    decision_id=self._decision_id(proposal),
                    proposal_id=proposal.proposal_id,
                    outcome=DecisionOutcome.REVIEW_REQUIRED,
                    state=SelfImprovementProposalState.HUMAN_REVIEW_REQUIRED,
                    reason="Self-improvement proposal requires explicit human approval.",
                    failure_modes=(FailureMode.HUMAN_REVIEW_REQUIRED,),
                    requires_human_review=True,
                ),
            )

        approved = replace(
            proposal,
            state=SelfImprovementProposalState.APPROVED_FOR_HANDOFF,
        )
        return SelfImprovementAirlockResult(
            proposal=approved,
            decision=SelfImprovementDecision(
                decision_id=self._decision_id(proposal),
                proposal_id=proposal.proposal_id,
                outcome=DecisionOutcome.ALLOW,
                state=SelfImprovementProposalState.APPROVED_FOR_HANDOFF,
                reason=(
                    "Self-improvement proposal passed the airlock for later "
                    "BlackFox-compatible review."
                ),
                requires_human_review=True,
            ),
        )

    def _failure_modes(self, proposal: SelfImprovementProposal) -> tuple[FailureMode, ...]:
        failures: list[FailureMode] = []

        if self.policy.require_risk_classification and proposal.risk is None:
            failures.append(FailureMode.UNSAFE_SELF_IMPROVEMENT)

        if self.policy.require_test_plan and proposal.test_plan is None:
            failures.append(FailureMode.MISSING_EVIDENCE)

        if (
            self.policy.require_adversarial_scenarios
            and proposal.test_plan is not None
            and not proposal.test_plan.adversarial_scenarios
        ):
            failures.append(FailureMode.UNSAFE_SELF_IMPROVEMENT)

        if self.policy.require_evidence_contract and proposal.evidence_contract is None:
            failures.append(FailureMode.MISSING_EVIDENCE)

        if (
            self.policy.require_satisfied_evidence
            and proposal.evidence_contract is not None
            and not proposal.evidence_contract.trust_eligible
        ):
            failures.append(FailureMode.MISSING_EVIDENCE)

        if self.policy.require_rollback_plan and proposal.rollback_plan is None:
            failures.append(FailureMode.UNKNOWN_ACTION_TYPE)

        if (
            self.policy.block_policy_file_self_promotion
            and proposal.target_kind == SelfImprovementTargetKind.POLICY_FILE
            and not proposal.has_human_approval
        ):
            failures.append(FailureMode.POLICY_BYPASS_ATTEMPT)

        return self._dedupe(failures)

    def _dedupe(self, failures: list[FailureMode]) -> tuple[FailureMode, ...]:
        deduped: list[FailureMode] = []
        for failure in failures:
            if failure not in deduped:
                deduped.append(failure)
        return tuple(deduped)

    def _decision_id(self, proposal: SelfImprovementProposal) -> str:
        return f"self-improvement-airlock-decision:{proposal.proposal_id}"


def evaluate_self_improvement(
    proposal: SelfImprovementProposal,
    policy: SelfImprovementAirlockPolicy | None = None,
) -> SelfImprovementAirlockResult:
    """Evaluate a self-improvement proposal through the default airlock."""

    return SelfImprovementAirlock(policy=policy).evaluate(proposal)
