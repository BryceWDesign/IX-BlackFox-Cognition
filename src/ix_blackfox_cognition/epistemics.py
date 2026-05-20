"""Epistemic claim ledger models for IX-BlackFox-Cognition.

The epistemic ledger makes cognition inspectable before action. It separates
claims, assumptions, model-confidence signals, evidence identifiers,
contradictions, and human approval state.

The core invariant is strict:

Model confidence is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from ix_blackfox_cognition.core import (
    ActorKind,
    ClaimState,
    FailureMode,
    require_invariant,
)


class ClaimKind(StrEnum):
    """Kinds of claims that can appear in the epistemic ledger."""

    FACT = "fact"
    ASSUMPTION = "assumption"
    INFERENCE = "inference"
    REQUIREMENT = "requirement"
    RISK = "risk"
    DECISION = "decision"
    OBSERVATION = "observation"
    MODEL_OUTPUT = "model_output"
    HUMAN_INPUT = "human_input"
    POLICY_ASSERTION = "policy_assertion"


class ConfidenceBasis(StrEnum):
    """Basis for confidence signals.

    Confidence can explain why a model or actor believed something. It cannot
    replace evidence.
    """

    MODEL_LOGPROB = "model_logprob"
    MODEL_SELF_ASSESSMENT = "model_self_assessment"
    HUMAN_JUDGMENT = "human_judgment"
    TEST_HISTORY = "test_history"
    PRIOR_LEDGER_PATTERN = "prior_ledger_pattern"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ClaimSource:
    """Source actor for a claim."""

    actor_kind: ActorKind
    actor_id: str
    description: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.actor_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim source actor id cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class ConfidenceSignal:
    """Confidence signal attached to a claim.

    This is intentionally not evidence. Evidence must be represented by separate
    evidence identifiers and later evidence records.
    """

    basis: ConfidenceBasis
    score: float | None = None
    rationale: str | None = None

    def __post_init__(self) -> None:
        if self.score is not None:
            require_invariant(
                0.0 <= self.score <= 1.0,
                FailureMode.MODEL_CONFIDENCE_AS_EVIDENCE,
                "Confidence score must be between 0.0 and 1.0.",
            )

    @property
    def is_evidence(self) -> bool:
        """Return False because confidence is never evidence."""

        return False


@dataclass(frozen=True, slots=True)
class ClaimRecord:
    """Immutable epistemic claim record."""

    claim_id: str
    kind: ClaimKind
    statement: str
    source: ClaimSource
    state: ClaimState = ClaimState.UNVERIFIED
    confidence: ConfidenceSignal | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    contradicts: tuple[str, ...] = field(default_factory=tuple)
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    human_approval_id: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.claim_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim statement cannot be blank.",
        )

        if self.state == ClaimState.VERIFIED:
            require_invariant(
                bool(self.evidence_ids),
                FailureMode.MISSING_EVIDENCE,
                "Verified claims require at least one evidence id.",
            )

        if self.state == ClaimState.HUMAN_APPROVED:
            require_invariant(
                self.has_human_approval,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "Human-approved claims require a human approval id.",
            )

        if self.state == ClaimState.CONTRADICTED:
            require_invariant(
                bool(self.contradicts),
                FailureMode.CONTRADICTION,
                "Contradicted claims must identify at least one conflicting claim.",
            )

        if self.confidence is not None and self.state == ClaimState.VERIFIED:
            require_invariant(
                bool(self.evidence_ids),
                FailureMode.MODEL_CONFIDENCE_AS_EVIDENCE,
                "Confidence cannot verify a claim without evidence.",
            )

    @property
    def has_evidence(self) -> bool:
        """Return whether the claim has evidence identifiers attached."""

        return bool(self.evidence_ids)

    @property
    def has_human_approval(self) -> bool:
        """Return whether the claim has an explicit human approval id."""

        return self.human_approval_id is not None and bool(self.human_approval_id.strip())

    @property
    def requires_evidence(self) -> bool:
        """Return whether the claim still requires evidence before trust."""

        return self.state in (
            ClaimState.UNVERIFIED,
            ClaimState.ASSUMED,
            ClaimState.EVIDENCE_REQUIRED,
        )

    @property
    def is_trust_eligible(self) -> bool:
        """Return whether the claim has reached a trust-eligible state."""

        return self.state in (ClaimState.VERIFIED, ClaimState.HUMAN_APPROVED)

    def with_state(
        self,
        state: ClaimState,
        *,
        evidence_ids: tuple[str, ...] | None = None,
        human_approval_id: str | None = None,
        contradicts: tuple[str, ...] | None = None,
    ) -> ClaimRecord:
        """Return a copy of the claim with a new state and optional metadata."""

        return replace(
            self,
            state=state,
            evidence_ids=self.evidence_ids if evidence_ids is None else evidence_ids,
            human_approval_id=(
                self.human_approval_id if human_approval_id is None else human_approval_id
            ),
            contradicts=self.contradicts if contradicts is None else contradicts,
        )


@dataclass(frozen=True, slots=True)
class ClaimStateTransition:
    """Recorded transition for a claim state change."""

    transition_id: str
    claim_id: str
    from_state: ClaimState
    to_state: ClaimState
    reason: str
    actor: ClaimSource
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    human_approval_id: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.transition_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim transition id cannot be blank.",
        )
        require_invariant(
            bool(self.claim_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim transition claim id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim transition reason cannot be blank.",
        )

        if self.to_state == ClaimState.VERIFIED:
            require_invariant(
                bool(self.evidence_ids),
                FailureMode.MISSING_EVIDENCE,
                "Transition to verified requires evidence ids.",
            )

        if self.to_state == ClaimState.HUMAN_APPROVED:
            require_invariant(
                self.human_approval_id is not None and bool(self.human_approval_id.strip()),
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "Transition to human approved requires a human approval id.",
            )


@dataclass(frozen=True, slots=True)
class ClaimLedger:
    """Immutable ledger of epistemic claims and claim transitions."""

    ledger_id: str
    claims: tuple[ClaimRecord, ...] = field(default_factory=tuple)
    transitions: tuple[ClaimStateTransition, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.ledger_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim ledger id cannot be blank.",
        )
        claim_ids = [claim.claim_id for claim in self.claims]
        require_invariant(
            len(claim_ids) == len(set(claim_ids)),
            FailureMode.CONTRADICTION,
            "Claim ledger cannot contain duplicate claim ids.",
        )

    def claim_by_id(self, claim_id: str) -> ClaimRecord | None:
        """Return a claim by id, if present."""

        for claim in self.claims:
            if claim.claim_id == claim_id:
                return claim
        return None

    def add_claim(self, claim: ClaimRecord) -> ClaimLedger:
        """Return a new ledger containing an additional unique claim."""

        require_invariant(
            self.claim_by_id(claim.claim_id) is None,
            FailureMode.CONTRADICTION,
            "Claim ledger cannot add a duplicate claim id.",
        )
        return replace(self, claims=(*self.claims, claim))

    def record_transition(self, transition: ClaimStateTransition) -> ClaimLedger:
        """Return a new ledger with a recorded claim transition.

        This method records the transition event. It does not rewrite the claim;
        claim replacement is intentionally explicit so silent state mutation does
        not occur inside the ledger.
        """

        require_invariant(
            self.claim_by_id(transition.claim_id) is not None,
            FailureMode.UNSUPPORTED_CLAIM,
            "Claim transition must reference an existing claim.",
        )
        return replace(self, transitions=(*self.transitions, transition))

    @property
    def unverified_claims(self) -> tuple[ClaimRecord, ...]:
        """Return claims that still need evidence or review."""

        return tuple(claim for claim in self.claims if claim.requires_evidence)

    @property
    def trust_eligible_claims(self) -> tuple[ClaimRecord, ...]:
        """Return claims that are eligible for bounded trust."""

        return tuple(claim for claim in self.claims if claim.is_trust_eligible)
