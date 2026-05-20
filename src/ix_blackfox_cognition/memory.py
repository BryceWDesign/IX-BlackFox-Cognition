"""Memory immune system models for IX-BlackFox-Cognition.

Memory is not truth. Memory is a governed substrate that can preserve useful
operational context only after provenance, evidence, contradiction, staleness,
and authority boundaries remain visible.

This module defines memory records and memory update proposals only. Quarantine,
promotion, and conflict-checking logic is introduced in the next commit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import (
    ActorKind,
    FailureMode,
    MemoryState,
    RiskLevel,
    require_invariant,
)


class MemoryRecordKind(StrEnum):
    """Kinds of governed operational memory."""

    DECISION = "decision"
    REJECTED_DECISION = "rejected_decision"
    FAILURE_PATTERN = "failure_pattern"
    SUCCESS_PATTERN = "success_pattern"
    POLICY_BLOCK = "policy_block"
    HUMAN_APPROVAL = "human_approval"
    EVIDENCE_REFERENCE = "evidence_reference"
    REPO_FACT = "repo_fact"
    BELIEF_SUMMARY = "belief_summary"
    PLAN_PATTERN = "plan_pattern"
    ROUTING_PATTERN = "routing_pattern"
    RISK_PATTERN = "risk_pattern"
    SELF_IMPROVEMENT_HISTORY = "self_improvement_history"


class MemoryUpdateKind(StrEnum):
    """Kinds of governed memory update proposals."""

    ADD = "add"
    REVISE = "revise"
    SUPERSEDE = "supersede"
    EXPIRE = "expire"
    REJECT = "reject"
    PROMOTE = "promote"


class MemoryConflictKind(StrEnum):
    """Kinds of conflicts that can block memory promotion."""

    CONTRADICTS_EXISTING_MEMORY = "contradicts_existing_memory"
    CONTRADICTS_BELIEF_GRAPH = "contradicts_belief_graph"
    CONTRADICTS_CLAIM_LEDGER = "contradicts_claim_ledger"
    STALE_SOURCE = "stale_source"
    MISSING_PROVENANCE = "missing_provenance"
    MISSING_EVIDENCE = "missing_evidence"
    CIRCULAR_MEMORY_REFERENCE = "circular_memory_reference"
    EVIDENCE_LAUNDERING = "evidence_laundering"


@dataclass(frozen=True, slots=True)
class MemorySource:
    """Source actor or system that proposed or produced a memory record."""

    actor_kind: ActorKind
    actor_id: str
    description: str | None = None

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.actor_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Memory source actor id cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MemoryRecord:
    """Governed memory record.

    A promoted memory record must be evidence-backed or explicitly
    human-approved. This prevents model output from becoming durable operational
    memory just because it was repeated or summarized.
    """

    memory_id: str
    kind: MemoryRecordKind
    statement: str
    source: MemorySource
    state: MemoryState = MemoryState.PROPOSED
    risk_level: RiskLevel = RiskLevel.LOW
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    claim_ids: tuple[str, ...] = field(default_factory=tuple)
    belief_ids: tuple[str, ...] = field(default_factory=tuple)
    plan_node_ids: tuple[str, ...] = field(default_factory=tuple)
    work_package_ids: tuple[str, ...] = field(default_factory=tuple)
    supersedes_memory_ids: tuple[str, ...] = field(default_factory=tuple)
    superseded_by_memory_id: str | None = None
    human_approval_id: str | None = None
    expires_after: str | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.memory_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Memory record id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Memory record statement cannot be blank.",
        )

        if self.state == MemoryState.PROMOTED:
            require_invariant(
                bool(self.evidence_ids) or self.has_human_approval,
                FailureMode.MISSING_EVIDENCE,
                "Promoted memory requires evidence or explicit human approval.",
            )

        if self.state == MemoryState.SUPERSEDED:
            require_invariant(
                self.superseded_by_memory_id is not None
                and bool(self.superseded_by_memory_id.strip()),
                FailureMode.STALE_MEMORY,
                "Superseded memory requires a superseding memory id.",
            )

        if self.state == MemoryState.EXPIRED:
            require_invariant(
                self.expires_after is not None and bool(self.expires_after.strip()),
                FailureMode.STALE_MEMORY,
                "Expired memory requires an expiration marker.",
            )

    @property
    def has_human_approval(self) -> bool:
        """Return whether the memory has explicit human approval."""

        return self.human_approval_id is not None and bool(self.human_approval_id.strip())

    @property
    def has_evidence(self) -> bool:
        """Return whether the memory references evidence."""

        return bool(self.evidence_ids)

    @property
    def trust_eligible(self) -> bool:
        """Return whether the memory can be considered for bounded trust."""

        return self.state == MemoryState.PROMOTED and (
            self.has_evidence or self.has_human_approval
        )

    @property
    def blocked_from_use(self) -> bool:
        """Return whether the memory is blocked from trusted use."""

        return self.state in (
            MemoryState.PROPOSED,
            MemoryState.QUARANTINED,
            MemoryState.REJECTED,
            MemoryState.SUPERSEDED,
            MemoryState.EXPIRED,
        )


@dataclass(frozen=True, slots=True)
class MemoryConflict:
    """Inspectable conflict that can block memory promotion."""

    conflict_id: str
    kind: MemoryConflictKind
    memory_id: str
    statement: str
    conflicting_reference_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    requires_human_review: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.conflict_id.strip()),
            FailureMode.MEMORY_CONFLICT,
            "Memory conflict id cannot be blank.",
        )
        require_invariant(
            bool(self.memory_id.strip()),
            FailureMode.MEMORY_CONFLICT,
            "Memory conflict memory id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.MEMORY_CONFLICT,
            "Memory conflict statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MemoryQuarantineRecord:
    """Record explaining why memory remains quarantined."""

    quarantine_id: str
    memory_id: str
    reason: str
    conflict_ids: tuple[str, ...] = field(default_factory=tuple)
    evidence_requirement_ids: tuple[str, ...] = field(default_factory=tuple)
    human_review_required: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.quarantine_id.strip()),
            FailureMode.MEMORY_CONFLICT,
            "Memory quarantine id cannot be blank.",
        )
        require_invariant(
            bool(self.memory_id.strip()),
            FailureMode.MEMORY_CONFLICT,
            "Memory quarantine memory id cannot be blank.",
        )
        require_invariant(
            bool(self.reason.strip()),
            FailureMode.MEMORY_CONFLICT,
            "Memory quarantine reason cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class MemoryUpdateProposal:
    """Proposal to change governed memory.

    A proposal is not a memory update. It must remain reviewable until the
    memory immune system later decides whether it should be quarantined,
    rejected, promoted, superseded, or expired.
    """

    proposal_id: str
    update_kind: MemoryUpdateKind
    proposed_record: MemoryRecord
    proposer: MemorySource
    rationale: str
    target_memory_id: str | None = None
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    conflict_ids: tuple[str, ...] = field(default_factory=tuple)
    requires_human_review: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.proposal_id.strip()),
            FailureMode.SILENT_STATE_MUTATION,
            "Memory update proposal id cannot be blank.",
        )
        require_invariant(
            bool(self.rationale.strip()),
            FailureMode.SILENT_STATE_MUTATION,
            "Memory update proposal rationale cannot be blank.",
        )

        if self.update_kind in (
            MemoryUpdateKind.REVISE,
            MemoryUpdateKind.SUPERSEDE,
            MemoryUpdateKind.EXPIRE,
            MemoryUpdateKind.REJECT,
            MemoryUpdateKind.PROMOTE,
        ):
            require_invariant(
                self.target_memory_id is not None and bool(self.target_memory_id.strip()),
                FailureMode.SILENT_STATE_MUTATION,
                "Non-add memory update proposals require a target memory id.",
            )

        if self.update_kind == MemoryUpdateKind.PROMOTE:
            require_invariant(
                bool(self.evidence_ids) or self.proposed_record.has_human_approval,
                FailureMode.MISSING_EVIDENCE,
                "Memory promotion proposals require evidence or human approval.",
            )

    @property
    def has_conflicts(self) -> bool:
        """Return whether the proposal already references known conflicts."""

        return bool(self.conflict_ids)

    @property
    def promotion_candidate(self) -> bool:
        """Return whether the proposal requests memory promotion."""

        return self.update_kind == MemoryUpdateKind.PROMOTE


@dataclass(frozen=True, slots=True)
class MemoryStore:
    """Immutable collection of governed memory records and proposal artifacts."""

    store_id: str
    records: tuple[MemoryRecord, ...] = field(default_factory=tuple)
    proposals: tuple[MemoryUpdateProposal, ...] = field(default_factory=tuple)
    conflicts: tuple[MemoryConflict, ...] = field(default_factory=tuple)
    quarantines: tuple[MemoryQuarantineRecord, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.store_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Memory store id cannot be blank.",
        )

        memory_ids = [record.memory_id for record in self.records]
        require_invariant(
            len(memory_ids) == len(set(memory_ids)),
            FailureMode.CONTRADICTION,
            "Memory store cannot contain duplicate memory ids.",
        )

        proposal_ids = [proposal.proposal_id for proposal in self.proposals]
        require_invariant(
            len(proposal_ids) == len(set(proposal_ids)),
            FailureMode.CONTRADICTION,
            "Memory store cannot contain duplicate proposal ids.",
        )

        conflict_ids = [conflict.conflict_id for conflict in self.conflicts]
        require_invariant(
            len(conflict_ids) == len(set(conflict_ids)),
            FailureMode.CONTRADICTION,
            "Memory store cannot contain duplicate conflict ids.",
        )

        quarantine_ids = [quarantine.quarantine_id for quarantine in self.quarantines]
        require_invariant(
            len(quarantine_ids) == len(set(quarantine_ids)),
            FailureMode.CONTRADICTION,
            "Memory store cannot contain duplicate quarantine ids.",
        )

    def record_by_id(self, memory_id: str) -> MemoryRecord | None:
        """Return a memory record by id, if present."""

        for record in self.records:
            if record.memory_id == memory_id:
                return record
        return None

    @property
    def promoted_records(self) -> tuple[MemoryRecord, ...]:
        """Return promoted memory records."""

        return tuple(record for record in self.records if record.state == MemoryState.PROMOTED)

    @property
    def blocked_records(self) -> tuple[MemoryRecord, ...]:
        """Return memory records blocked from trusted use."""

        return tuple(record for record in self.records if record.blocked_from_use)
