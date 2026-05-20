"""Evidence requirement system for IX-BlackFox-Cognition.

Evidence is the trust boundary between model-generated cognition and reviewable
engineering claims. This module defines evidence references, evidence
requirements, proof obligations, falsification conditions, and evidence
contracts.

Core invariant:

Model confidence is not evidence.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum

from ix_blackfox_cognition.core import (
    ActorKind,
    EvidenceState,
    FailureMode,
    RiskLevel,
    require_invariant,
)


class EvidenceKind(StrEnum):
    """Kinds of evidence accepted by the governed cognition substrate."""

    TEST_RESULT = "test_result"
    STATIC_ANALYSIS = "static_analysis"
    HUMAN_APPROVAL = "human_approval"
    REVIEW_RECORD = "review_record"
    POLICY_DECISION = "policy_decision"
    RECEIPT = "receipt"
    FILE_HASH = "file_hash"
    LEDGER_ENTRY = "ledger_entry"
    RUNTIME_OBSERVATION = "runtime_observation"
    EXTERNAL_REFERENCE = "external_reference"
    BLACKFOX_RUN_BUNDLE = "blackfox_run_bundle"
    BLACKFOX_VERIFICATION_SUMMARY = "blackfox_verification_summary"


class EvidenceStrength(StrEnum):
    """Relative strength of evidence for bounded trust decisions."""

    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    AUTHORITATIVE = "authoritative"

    def rank(self) -> int:
        """Return a stable rank for evidence-strength comparison."""

        ranks = {
            EvidenceStrength.WEAK: 1,
            EvidenceStrength.MODERATE: 2,
            EvidenceStrength.STRONG: 3,
            EvidenceStrength.AUTHORITATIVE: 4,
        }
        return ranks[self]


@dataclass(frozen=True, slots=True)
class EvidenceSource:
    """Source actor or system for an evidence reference."""

    actor_kind: ActorKind
    actor_id: str
    description: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.actor_id.strip()),
            FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE,
            "Evidence source actor id cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class EvidenceReference:
    """Reference to evidence used by claims, plans, work packages, or handoffs."""

    evidence_id: str
    kind: EvidenceKind
    state: EvidenceState
    source: EvidenceSource
    summary: str
    strength: EvidenceStrength = EvidenceStrength.MODERATE
    locator: str | None = None
    digest: str | None = None
    supports_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    contradicts_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.evidence_id.strip()),
            FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE,
            "Evidence id cannot be blank.",
        )
        require_invariant(
            bool(self.summary.strip()),
            FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE,
            "Evidence summary cannot be blank.",
        )

        if self.state == EvidenceState.VERIFIED:
            require_invariant(
                self.has_locator or self.has_digest,
                FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE,
                "Verified evidence requires a locator or digest.",
            )

        if self.kind == EvidenceKind.HUMAN_APPROVAL:
            require_invariant(
                self.source.actor_kind == ActorKind.HUMAN,
                FailureMode.HUMAN_REVIEW_REQUIRED,
                "Human approval evidence must come from a human source.",
            )

    @property
    def has_locator(self) -> bool:
        """Return whether the evidence has a nonblank locator."""

        return self.locator is not None and bool(self.locator.strip())

    @property
    def has_digest(self) -> bool:
        """Return whether the evidence has a nonblank digest."""

        return self.digest is not None and bool(self.digest.strip())

    @property
    def is_verified(self) -> bool:
        """Return whether this evidence reference is verified."""

        return self.state == EvidenceState.VERIFIED

    @property
    def is_contradictory(self) -> bool:
        """Return whether this evidence contradicts at least one claim."""

        return bool(self.contradicts_claim_ids)

    def verified(self, *, locator: str | None = None, digest: str | None = None) -> EvidenceReference:
        """Return a verified copy of the evidence reference."""

        return replace(
            self,
            state=EvidenceState.VERIFIED,
            locator=self.locator if locator is None else locator,
            digest=self.digest if digest is None else digest,
        )


@dataclass(frozen=True, slots=True)
class EvidenceRequirement:
    """Evidence requirement that must be satisfied before bounded trust."""

    requirement_id: str
    statement: str
    acceptable_kinds: tuple[EvidenceKind, ...]
    minimum_count: int = 1
    minimum_strength: EvidenceStrength = EvidenceStrength.MODERATE
    required: bool = True
    must_be_verified: bool = True
    risk_level: RiskLevel = RiskLevel.LOW

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.requirement_id.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Evidence requirement id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Evidence requirement statement cannot be blank.",
        )
        require_invariant(
            bool(self.acceptable_kinds),
            FailureMode.MISSING_EVIDENCE,
            "Evidence requirement must define at least one acceptable evidence kind.",
        )
        require_invariant(
            self.minimum_count >= 1,
            FailureMode.MISSING_EVIDENCE,
            "Evidence requirement minimum count must be at least 1.",
        )

    def accepts(self, evidence: EvidenceReference) -> bool:
        """Return whether an evidence reference can satisfy this requirement."""

        if evidence.kind not in self.acceptable_kinds:
            return False

        if self.must_be_verified and not evidence.is_verified:
            return False

        if evidence.strength.rank() < self.minimum_strength.rank():
            return False

        return True

    def satisfied_by(self, references: tuple[EvidenceReference, ...]) -> bool:
        """Return whether this requirement is satisfied by evidence references."""

        if not self.required:
            return True

        accepted = [reference for reference in references if self.accepts(reference)]
        return len(accepted) >= self.minimum_count


@dataclass(frozen=True, slots=True)
class ProofObligation:
    """Proof obligation attached to a claim, plan node, package, or handoff."""

    obligation_id: str
    subject_id: str
    statement: str
    requirement_ids: tuple[str, ...]
    required_for_trust: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.obligation_id.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Proof obligation id cannot be blank.",
        )
        require_invariant(
            bool(self.subject_id.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Proof obligation subject id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Proof obligation statement cannot be blank.",
        )
        require_invariant(
            bool(self.requirement_ids),
            FailureMode.MISSING_EVIDENCE,
            "Proof obligation must reference at least one evidence requirement.",
        )


@dataclass(frozen=True, slots=True)
class FalsificationCondition:
    """Condition that would weaken, reject, or falsify a claim or plan."""

    condition_id: str
    subject_id: str
    statement: str
    evidence_kinds: tuple[EvidenceKind, ...]
    requires_review: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.condition_id.strip()),
            FailureMode.CONTRADICTION,
            "Falsification condition id cannot be blank.",
        )
        require_invariant(
            bool(self.subject_id.strip()),
            FailureMode.CONTRADICTION,
            "Falsification condition subject id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.CONTRADICTION,
            "Falsification condition statement cannot be blank.",
        )
        require_invariant(
            bool(self.evidence_kinds),
            FailureMode.CONTRADICTION,
            "Falsification condition must define relevant evidence kinds.",
        )

    def matches(self, evidence: EvidenceReference) -> bool:
        """Return whether evidence matches this falsification condition."""

        return evidence.kind in self.evidence_kinds and evidence.is_contradictory


@dataclass(frozen=True, slots=True)
class EvidenceContract:
    """Contract joining requirements, obligations, falsifiers, and references."""

    contract_id: str
    subject_id: str
    requirements: tuple[EvidenceRequirement, ...]
    proof_obligations: tuple[ProofObligation, ...] = field(default_factory=tuple)
    falsification_conditions: tuple[FalsificationCondition, ...] = field(default_factory=tuple)
    references: tuple[EvidenceReference, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.contract_id.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Evidence contract id cannot be blank.",
        )
        require_invariant(
            bool(self.subject_id.strip()),
            FailureMode.MISSING_EVIDENCE,
            "Evidence contract subject id cannot be blank.",
        )
        require_invariant(
            bool(self.requirements),
            FailureMode.MISSING_EVIDENCE,
            "Evidence contract must contain at least one requirement.",
        )

        requirement_ids = [requirement.requirement_id for requirement in self.requirements]
        require_invariant(
            len(requirement_ids) == len(set(requirement_ids)),
            FailureMode.CONTRADICTION,
            "Evidence contract cannot contain duplicate requirement ids.",
        )

    @property
    def required_requirements(self) -> tuple[EvidenceRequirement, ...]:
        """Return requirements that must be satisfied."""

        return tuple(requirement for requirement in self.requirements if requirement.required)

    @property
    def unsatisfied_requirements(self) -> tuple[EvidenceRequirement, ...]:
        """Return required evidence requirements that are not currently satisfied."""

        return tuple(
            requirement
            for requirement in self.required_requirements
            if not requirement.satisfied_by(self.references)
        )

    @property
    def satisfied(self) -> bool:
        """Return whether all required evidence requirements are satisfied."""

        return not self.unsatisfied_requirements

    @property
    def falsified(self) -> bool:
        """Return whether any falsification condition matches current evidence."""

        return any(
            condition.matches(reference)
            for condition in self.falsification_conditions
            for reference in self.references
        )

    @property
    def trust_eligible(self) -> bool:
        """Return whether the contract supports bounded trust."""

        return self.satisfied and not self.falsified

    def add_reference(self, reference: EvidenceReference) -> EvidenceContract:
        """Return a copy of the contract with an additional evidence reference."""

        existing_ids = {item.evidence_id for item in self.references}
        require_invariant(
            reference.evidence_id not in existing_ids,
            FailureMode.CONTRADICTION,
            "Evidence contract cannot contain duplicate evidence ids.",
        )
        return replace(self, references=(*self.references, reference))
