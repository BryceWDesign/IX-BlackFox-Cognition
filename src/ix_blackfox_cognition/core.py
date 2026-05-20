"""Shared core types for IX-BlackFox-Cognition.

This module holds dependency-light enums, status values, failure modes, and
exceptions used across the governed cognition substrate. It is intentionally
kept free of imports from authority, planning, memory, routing, sentinel, or
handoff modules so those later layers can depend on it without circular imports.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import NoReturn


class ActorKind(StrEnum):
    """Kinds of actors that may appear in cognition records."""

    HUMAN = "human"
    MODEL = "model"
    SYSTEM = "system"
    POLICY = "policy"
    BLACKFOX = "blackfox"


class DecisionOutcome(StrEnum):
    """Common decision outcomes across governance and cognition layers."""

    ALLOW = "allow"
    DENY = "deny"
    ESCALATE = "escalate"
    QUARANTINE = "quarantine"
    REVIEW_REQUIRED = "review_required"
    FAIL_CLOSED = "fail_closed"


class EvidenceState(StrEnum):
    """Lifecycle states for evidence references and evidence obligations."""

    MISSING = "missing"
    PROPOSED = "proposed"
    PRESENT = "present"
    VERIFIED = "verified"
    REJECTED = "rejected"
    CONTRADICTED = "contradicted"
    STALE = "stale"


class ClaimState(StrEnum):
    """Lifecycle states for epistemic claims."""

    UNVERIFIED = "unverified"
    ASSUMED = "assumed"
    EVIDENCE_REQUIRED = "evidence_required"
    VERIFIED = "verified"
    CONTRADICTED = "contradicted"
    REJECTED = "rejected"
    STALE = "stale"
    HUMAN_APPROVED = "human_approved"


class MemoryState(StrEnum):
    """Lifecycle states for governed memory records and update proposals."""

    PROPOSED = "proposed"
    QUARANTINED = "quarantined"
    PROMOTED = "promoted"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"
    EXPIRED = "expired"


class WorkState(StrEnum):
    """Common lifecycle states for plans, packages, and handoff candidates."""

    DRAFT = "draft"
    READY_FOR_REVIEW = "ready_for_review"
    REVIEW_REQUIRED = "review_required"
    APPROVED = "approved"
    REJECTED = "rejected"
    BLOCKED = "blocked"
    COMPLETED = "completed"


class RiskLevel(StrEnum):
    """Risk levels used by plans, work packages, memory updates, and handoffs."""

    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    CRITICAL = "critical"

    def rank(self) -> int:
        """Return a stable rank for risk comparison."""

        ranks = {
            RiskLevel.LOW: 1,
            RiskLevel.MODERATE: 2,
            RiskLevel.HIGH: 3,
            RiskLevel.CRITICAL: 4,
        }
        return ranks[self]


class FailureMode(StrEnum):
    """Canonical fail-closed reasons for governed cognition behavior."""

    MISSING_POLICY = "missing_policy"
    MISSING_EVIDENCE = "missing_evidence"
    UNCLEAR_AUTHORITY = "unclear_authority"
    SELF_APPROVAL_ATTEMPT = "self_approval_attempt"
    UNKNOWN_ACTION_TYPE = "unknown_action_type"
    MEMORY_CONFLICT = "memory_conflict"
    POLICY_BYPASS_ATTEMPT = "policy_bypass_attempt"
    FAKE_OR_UNVERIFIABLE_EVIDENCE = "fake_or_unverifiable_evidence"
    SILENT_STATE_MUTATION = "silent_state_mutation"
    MODEL_CONFIDENCE_AS_EVIDENCE = "model_confidence_as_evidence"
    SCOPE_CREEP = "scope_creep"
    UNSUPPORTED_CLAIM = "unsupported_claim"
    CONTRADICTION = "contradiction"
    STALE_MEMORY = "stale_memory"
    UNSAFE_SELF_IMPROVEMENT = "unsafe_self_improvement"
    HUMAN_REVIEW_REQUIRED = "human_review_required"


@dataclass(frozen=True, slots=True)
class FailureRecord:
    """Immutable description of a governed cognition failure.

    A failure record should describe why a proposed cognition action, memory
    update, evidence claim, plan, or handoff failed closed.
    """

    mode: FailureMode
    message: str
    fail_closed: bool = True


class CognitionError(Exception):
    """Base exception for IX-BlackFox-Cognition."""


class CognitionInvariantError(CognitionError):
    """Raised when a core governance invariant is violated."""

    def __init__(self, failure: FailureRecord) -> None:
        self.failure = failure
        super().__init__(f"{failure.mode.value}: {failure.message}")


class AuthorityError(CognitionInvariantError):
    """Raised when authority boundaries are unclear or violated."""


class EvidenceError(CognitionInvariantError):
    """Raised when evidence is missing, fake, stale, or unverifiable."""


class PolicyError(CognitionInvariantError):
    """Raised when policy is missing or bypassed."""


class CognitionMemoryError(CognitionInvariantError):
    """Raised when governed memory rules are violated."""


class SelfImprovementError(CognitionInvariantError):
    """Raised when self-improvement safety rules are violated."""


def fail_closed(mode: FailureMode, message: str) -> NoReturn:
    """Raise a fail-closed invariant error with a canonical failure mode."""

    raise CognitionInvariantError(FailureRecord(mode=mode, message=message))


def require_invariant(condition: bool, mode: FailureMode, message: str) -> None:
    """Require a condition or fail closed with a canonical failure record."""

    if not condition:
        fail_closed(mode=mode, message=message)
