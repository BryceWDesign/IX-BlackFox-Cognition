"""Memory quarantine and promotion logic for IX-BlackFox-Cognition.

The memory immune system prevents model output, stale context, unsupported
claims, circular memory references, and evidence laundering from becoming
durable operational memory.

Memory changes are immutable. Every operation returns a new store plus a
reviewable decision record. The original store is never silently mutated.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from enum import StrEnum

from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, MemoryState, RiskLevel
from ix_blackfox_cognition.memory import (
    MemoryConflict,
    MemoryConflictKind,
    MemoryQuarantineRecord,
    MemoryRecord,
    MemoryStore,
    MemoryUpdateKind,
    MemoryUpdateProposal,
)


class MemoryDecisionKind(StrEnum):
    """Kinds of decisions made by the memory immune system."""

    SUBMIT_PROPOSAL = "submit_proposal"
    QUARANTINE = "quarantine"
    PROMOTE = "promote"
    REJECT = "reject"
    SUPERSEDE = "supersede"
    EXPIRE = "expire"


@dataclass(frozen=True, slots=True)
class MemoryImmunePolicy:
    """Policy knobs for memory quarantine and promotion."""

    policy_id: str = "default-memory-immune-policy"
    require_provenance: bool = True
    require_evidence_for_promotion: bool = True
    require_human_for_high_risk: bool = True
    high_risk_threshold: RiskLevel = RiskLevel.HIGH
    quarantine_missing_evidence: bool = True
    quarantine_existing_conflicts: bool = True
    quarantine_circular_memory_reference: bool = True
    quarantine_evidence_laundering: bool = True


@dataclass(frozen=True, slots=True)
class MemoryDecision:
    """Reviewable decision record for a memory immune-system operation."""

    decision_id: str
    store_id: str
    kind: MemoryDecisionKind
    outcome: DecisionOutcome
    reason: str
    memory_id: str | None = None
    proposal_id: str | None = None
    failure_modes: tuple[FailureMode, ...] = ()
    conflict_ids: tuple[str, ...] = ()
    quarantine_id: str | None = None
    requires_human_review: bool = False

    @property
    def allowed(self) -> bool:
        """Return whether the memory operation was allowed."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def failed_closed(self) -> bool:
        """Return whether the memory operation failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED

    @property
    def quarantined(self) -> bool:
        """Return whether the memory operation quarantined the proposal."""

        return self.outcome == DecisionOutcome.QUARANTINE


@dataclass(frozen=True, slots=True)
class MemoryOperationResult:
    """Result of a memory immune-system operation."""

    store: MemoryStore
    decision: MemoryDecision


class MemoryImmuneSystem:
    """Conservative memory immune system.

    The memory immune system permits adding proposals, quarantining unsafe
    records, and promoting memory only when provenance, evidence, conflict,
    staleness, and authority requirements are satisfied.
    """

    def __init__(self, policy: MemoryImmunePolicy | None = None) -> None:
        self.policy = policy or MemoryImmunePolicy()

    def submit_proposal(
        self,
        store: MemoryStore,
        proposal: MemoryUpdateProposal,
    ) -> MemoryOperationResult:
        """Submit a memory update proposal for quarantine-aware review."""

        if self._proposal_exists(store, proposal.proposal_id):
            return self._fail_closed(
                store=store,
                kind=MemoryDecisionKind.SUBMIT_PROPOSAL,
                reason="Memory store cannot add a duplicate proposal id.",
                proposal_id=proposal.proposal_id,
                memory_id=proposal.proposed_record.memory_id,
                failure_modes=(FailureMode.CONTRADICTION,),
            )

        if self._record_exists(store, proposal.proposed_record.memory_id):
            if proposal.update_kind == MemoryUpdateKind.ADD:
                return self._fail_closed(
                    store=store,
                    kind=MemoryDecisionKind.SUBMIT_PROPOSAL,
                    reason="Add proposal cannot reuse an existing memory id.",
                    proposal_id=proposal.proposal_id,
                    memory_id=proposal.proposed_record.memory_id,
                    failure_modes=(FailureMode.CONTRADICTION,),
                )

        conflicts = self._detect_conflicts(store, proposal)
        updated_store = replace(
            store,
            proposals=(*store.proposals, proposal),
        )

        if conflicts:
            quarantine = self._quarantine_record(
                proposal=proposal,
                reason="Memory proposal requires quarantine before promotion.",
                conflicts=conflicts,
            )
            quarantined_record = replace(
                proposal.proposed_record,
                state=MemoryState.QUARANTINED,
            )
            updated_store = replace(
                updated_store,
                records=(*updated_store.records, quarantined_record),
                conflicts=(*updated_store.conflicts, *conflicts),
                quarantines=(*updated_store.quarantines, quarantine),
            )
            return MemoryOperationResult(
                store=updated_store,
                decision=MemoryDecision(
                    decision_id=self._decision_id(
                        store,
                        MemoryDecisionKind.QUARANTINE,
                        proposal.proposal_id,
                    ),
                    store_id=store.store_id,
                    kind=MemoryDecisionKind.QUARANTINE,
                    outcome=DecisionOutcome.QUARANTINE,
                    reason=quarantine.reason,
                    memory_id=proposal.proposed_record.memory_id,
                    proposal_id=proposal.proposal_id,
                    failure_modes=self._failure_modes_for_conflicts(conflicts),
                    conflict_ids=tuple(conflict.conflict_id for conflict in conflicts),
                    quarantine_id=quarantine.quarantine_id,
                    requires_human_review=True,
                ),
            )

        return MemoryOperationResult(
            store=updated_store,
            decision=MemoryDecision(
                decision_id=self._decision_id(
                    store,
                    MemoryDecisionKind.SUBMIT_PROPOSAL,
                    proposal.proposal_id,
                ),
                store_id=store.store_id,
                kind=MemoryDecisionKind.SUBMIT_PROPOSAL,
                outcome=DecisionOutcome.ALLOW,
                reason="Memory proposal was accepted for review without quarantine.",
                memory_id=proposal.proposed_record.memory_id,
                proposal_id=proposal.proposal_id,
                requires_human_review=proposal.requires_human_review,
            ),
        )

    def promote_proposal(
        self,
        store: MemoryStore,
        proposal_id: str,
        *,
        human_approval_id: str | None = None,
    ) -> MemoryOperationResult:
        """Promote a submitted memory proposal only when all gates pass."""

        proposal = self._proposal_by_id(store, proposal_id)
        if proposal is None:
            return self._fail_closed(
                store=store,
                kind=MemoryDecisionKind.PROMOTE,
                reason="Cannot promote an unknown memory proposal.",
                proposal_id=proposal_id,
                memory_id=None,
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if self._proposal_has_store_conflicts(store, proposal):
            return self._fail_closed(
                store=store,
                kind=MemoryDecisionKind.PROMOTE,
                reason="Cannot promote a memory proposal with unresolved conflicts.",
                proposal_id=proposal.proposal_id,
                memory_id=proposal.proposed_record.memory_id,
                failure_modes=(FailureMode.MEMORY_CONFLICT,),
            )

        if self.policy.require_evidence_for_promotion:
            if not proposal.evidence_ids and not proposal.proposed_record.has_human_approval:
                return self._fail_closed(
                    store=store,
                    kind=MemoryDecisionKind.PROMOTE,
                    reason="Memory promotion requires evidence or explicit human approval.",
                    proposal_id=proposal.proposal_id,
                    memory_id=proposal.proposed_record.memory_id,
                    failure_modes=(FailureMode.MISSING_EVIDENCE,),
                )

        if self._requires_human_review(proposal):
            if human_approval_id is None or not human_approval_id.strip():
                return MemoryOperationResult(
                    store=store,
                    decision=MemoryDecision(
                        decision_id=self._decision_id(
                            store,
                            MemoryDecisionKind.PROMOTE,
                            proposal.proposal_id,
                        ),
                        store_id=store.store_id,
                        kind=MemoryDecisionKind.PROMOTE,
                        outcome=DecisionOutcome.REVIEW_REQUIRED,
                        reason="Memory promotion requires explicit human review.",
                        memory_id=proposal.proposed_record.memory_id,
                        proposal_id=proposal.proposal_id,
                        failure_modes=(FailureMode.HUMAN_REVIEW_REQUIRED,),
                        requires_human_review=True,
                    ),
                )

        promoted_record = replace(
            proposal.proposed_record,
            state=MemoryState.PROMOTED,
            evidence_ids=proposal.evidence_ids or proposal.proposed_record.evidence_ids,
            human_approval_id=human_approval_id or proposal.proposed_record.human_approval_id,
        )

        updated_store = self._upsert_record(store, promoted_record)

        return MemoryOperationResult(
            store=updated_store,
            decision=MemoryDecision(
                decision_id=self._decision_id(
                    store,
                    MemoryDecisionKind.PROMOTE,
                    proposal.proposal_id,
                ),
                store_id=store.store_id,
                kind=MemoryDecisionKind.PROMOTE,
                outcome=DecisionOutcome.ALLOW,
                reason="Memory proposal was promoted with evidence and authority gates satisfied.",
                memory_id=promoted_record.memory_id,
                proposal_id=proposal.proposal_id,
                requires_human_review=self._requires_human_review(proposal),
            ),
        )

    def reject_proposal(
        self,
        store: MemoryStore,
        proposal_id: str,
        *,
        reason: str,
    ) -> MemoryOperationResult:
        """Reject a submitted memory proposal without deleting its audit trail."""

        proposal = self._proposal_by_id(store, proposal_id)
        if proposal is None:
            return self._fail_closed(
                store=store,
                kind=MemoryDecisionKind.REJECT,
                reason="Cannot reject an unknown memory proposal.",
                proposal_id=proposal_id,
                memory_id=None,
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        if not reason.strip():
            return self._fail_closed(
                store=store,
                kind=MemoryDecisionKind.REJECT,
                reason="Memory proposal rejection reason cannot be blank.",
                proposal_id=proposal.proposal_id,
                memory_id=proposal.proposed_record.memory_id,
                failure_modes=(FailureMode.UNSUPPORTED_CLAIM,),
            )

        rejected_record = replace(proposal.proposed_record, state=MemoryState.REJECTED)
        updated_store = self._upsert_record(store, rejected_record)

        return MemoryOperationResult(
            store=updated_store,
            decision=MemoryDecision(
                decision_id=self._decision_id(
                    store,
                    MemoryDecisionKind.REJECT,
                    proposal.proposal_id,
                ),
                store_id=store.store_id,
                kind=MemoryDecisionKind.REJECT,
                outcome=DecisionOutcome.ALLOW,
                reason=reason,
                memory_id=rejected_record.memory_id,
                proposal_id=proposal.proposal_id,
                requires_human_review=False,
            ),
        )

    def _detect_conflicts(
        self,
        store: MemoryStore,
        proposal: MemoryUpdateProposal,
    ) -> tuple[MemoryConflict, ...]:
        conflicts: list[MemoryConflict] = []
        record = proposal.proposed_record

        if self.policy.require_provenance and not record.source.actor_id.strip():
            conflicts.append(
                self._conflict(
                    proposal=proposal,
                    kind=MemoryConflictKind.MISSING_PROVENANCE,
                    statement="Memory proposal is missing provenance.",
                )
            )

        if self.policy.quarantine_missing_evidence and proposal.promotion_candidate:
            if not proposal.evidence_ids and not record.has_human_approval:
                conflicts.append(
                    self._conflict(
                        proposal=proposal,
                        kind=MemoryConflictKind.MISSING_EVIDENCE,
                        statement="Memory promotion proposal is missing evidence and human approval.",
                    )
                )

        if self.policy.quarantine_existing_conflicts and proposal.conflict_ids:
            conflicts.append(
                self._conflict(
                    proposal=proposal,
                    kind=MemoryConflictKind.CONTRADICTS_EXISTING_MEMORY,
                    statement="Memory proposal references known unresolved conflicts.",
                    conflicting_reference_ids=proposal.conflict_ids,
                )
            )

        if self.policy.quarantine_circular_memory_reference:
            circular_ids = self._circular_memory_references(proposal)
            if circular_ids:
                conflicts.append(
                    self._conflict(
                        proposal=proposal,
                        kind=MemoryConflictKind.CIRCULAR_MEMORY_REFERENCE,
                        statement="Memory proposal attempts to use its own memory id as support.",
                        conflicting_reference_ids=circular_ids,
                    )
                )

        if self.policy.quarantine_evidence_laundering:
            if self._looks_like_evidence_laundering(store, proposal):
                conflicts.append(
                    self._conflict(
                        proposal=proposal,
                        kind=MemoryConflictKind.EVIDENCE_LAUNDERING,
                        statement=(
                            "Memory proposal appears to rely on memory references "
                            "without independent evidence."
                        ),
                    )
                )

        if self._target_memory_is_stale_or_missing(store, proposal):
            conflicts.append(
                self._conflict(
                    proposal=proposal,
                    kind=MemoryConflictKind.STALE_SOURCE,
                    statement="Memory update target is missing, expired, rejected, or superseded.",
                    conflicting_reference_ids=(
                        (proposal.target_memory_id,) if proposal.target_memory_id else ()
                    ),
                )
            )

        return tuple(conflicts)

    def _conflict(
        self,
        proposal: MemoryUpdateProposal,
        kind: MemoryConflictKind,
        statement: str,
        conflicting_reference_ids: tuple[str, ...] = (),
    ) -> MemoryConflict:
        return MemoryConflict(
            conflict_id=f"memory-conflict:{proposal.proposal_id}:{kind.value}",
            kind=kind,
            memory_id=proposal.proposed_record.memory_id,
            statement=statement,
            conflicting_reference_ids=conflicting_reference_ids,
            evidence_ids=proposal.evidence_ids,
        )

    def _quarantine_record(
        self,
        proposal: MemoryUpdateProposal,
        reason: str,
        conflicts: tuple[MemoryConflict, ...],
    ) -> MemoryQuarantineRecord:
        return MemoryQuarantineRecord(
            quarantine_id=f"memory-quarantine:{proposal.proposal_id}",
            memory_id=proposal.proposed_record.memory_id,
            reason=reason,
            conflict_ids=tuple(conflict.conflict_id for conflict in conflicts),
            evidence_requirement_ids=proposal.evidence_ids,
            human_review_required=True,
        )

    def _target_memory_is_stale_or_missing(
        self,
        store: MemoryStore,
        proposal: MemoryUpdateProposal,
    ) -> bool:
        if proposal.update_kind == MemoryUpdateKind.ADD:
            return False

        if proposal.target_memory_id is None:
            return True

        target = store.record_by_id(proposal.target_memory_id)
        if target is None:
            return True

        return target.state in (
            MemoryState.REJECTED,
            MemoryState.SUPERSEDED,
            MemoryState.EXPIRED,
        )

    def _circular_memory_references(
        self,
        proposal: MemoryUpdateProposal,
    ) -> tuple[str, ...]:
        memory_id = proposal.proposed_record.memory_id
        referenced_ids = (
            *proposal.proposed_record.supersedes_memory_ids,
            *proposal.proposed_record.claim_ids,
            *proposal.proposed_record.belief_ids,
            *proposal.proposed_record.plan_node_ids,
            *proposal.proposed_record.work_package_ids,
        )
        return tuple(reference_id for reference_id in referenced_ids if reference_id == memory_id)

    def _looks_like_evidence_laundering(
        self,
        store: MemoryStore,
        proposal: MemoryUpdateProposal,
    ) -> bool:
        if proposal.evidence_ids:
            return False

        referenced_memory_ids = set(proposal.proposed_record.supersedes_memory_ids)
        if not referenced_memory_ids:
            return False

        referenced_records = [
            record for record in store.records if record.memory_id in referenced_memory_ids
        ]
        if not referenced_records:
            return False

        return all(not record.has_evidence for record in referenced_records)

    def _requires_human_review(self, proposal: MemoryUpdateProposal) -> bool:
        return (
            proposal.requires_human_review
            or (
                self.policy.require_human_for_high_risk
                and proposal.proposed_record.risk_level.rank()
                >= self.policy.high_risk_threshold.rank()
            )
            or proposal.update_kind
            in (
                MemoryUpdateKind.PROMOTE,
                MemoryUpdateKind.SUPERSEDE,
                MemoryUpdateKind.EXPIRE,
            )
        )

    def _proposal_has_store_conflicts(
        self,
        store: MemoryStore,
        proposal: MemoryUpdateProposal,
    ) -> bool:
        known_conflict_memory_ids = {conflict.memory_id for conflict in store.conflicts}
        return proposal.proposed_record.memory_id in known_conflict_memory_ids or bool(
            proposal.conflict_ids
        )

    def _proposal_exists(self, store: MemoryStore, proposal_id: str) -> bool:
        return any(proposal.proposal_id == proposal_id for proposal in store.proposals)

    def _record_exists(self, store: MemoryStore, memory_id: str) -> bool:
        return store.record_by_id(memory_id) is not None

    def _proposal_by_id(
        self,
        store: MemoryStore,
        proposal_id: str,
    ) -> MemoryUpdateProposal | None:
        for proposal in store.proposals:
            if proposal.proposal_id == proposal_id:
                return proposal
        return None

    def _upsert_record(self, store: MemoryStore, record: MemoryRecord) -> MemoryStore:
        if store.record_by_id(record.memory_id) is None:
            return replace(store, records=(*store.records, record))

        return replace(
            store,
            records=tuple(
                record if existing.memory_id == record.memory_id else existing
                for existing in store.records
            ),
        )

    def _failure_modes_for_conflicts(
        self,
        conflicts: tuple[MemoryConflict, ...],
    ) -> tuple[FailureMode, ...]:
        modes: list[FailureMode] = []

        for conflict in conflicts:
            if conflict.kind == MemoryConflictKind.MISSING_EVIDENCE:
                modes.append(FailureMode.MISSING_EVIDENCE)
            elif conflict.kind == MemoryConflictKind.STALE_SOURCE:
                modes.append(FailureMode.STALE_MEMORY)
            elif conflict.kind == MemoryConflictKind.EVIDENCE_LAUNDERING:
                modes.append(FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE)
            else:
                modes.append(FailureMode.MEMORY_CONFLICT)

        deduped: list[FailureMode] = []
        for mode in modes:
            if mode not in deduped:
                deduped.append(mode)

        return tuple(deduped)

    def _fail_closed(
        self,
        store: MemoryStore,
        kind: MemoryDecisionKind,
        reason: str,
        proposal_id: str | None,
        memory_id: str | None,
        failure_modes: tuple[FailureMode, ...],
    ) -> MemoryOperationResult:
        return MemoryOperationResult(
            store=store,
            decision=MemoryDecision(
                decision_id=self._decision_id(store, kind, proposal_id or memory_id or "unknown"),
                store_id=store.store_id,
                kind=kind,
                outcome=DecisionOutcome.FAIL_CLOSED,
                reason=reason,
                memory_id=memory_id,
                proposal_id=proposal_id,
                failure_modes=failure_modes,
                requires_human_review=True,
            ),
        )

    def _decision_id(
        self,
        store: MemoryStore,
        kind: MemoryDecisionKind,
        suffix: str,
    ) -> str:
        return f"memory-decision:{store.store_id}:{kind.value}:{suffix}"


def submit_memory_proposal(
    store: MemoryStore,
    proposal: MemoryUpdateProposal,
    policy: MemoryImmunePolicy | None = None,
) -> MemoryOperationResult:
    """Submit a memory update proposal through the default memory immune system."""

    return MemoryImmuneSystem(policy=policy).submit_proposal(store=store, proposal=proposal)


def promote_memory_proposal(
    store: MemoryStore,
    proposal_id: str,
    *,
    human_approval_id: str | None = None,
    policy: MemoryImmunePolicy | None = None,
) -> MemoryOperationResult:
    """Promote a submitted memory proposal through the default memory immune system."""

    return MemoryImmuneSystem(policy=policy).promote_proposal(
        store=store,
        proposal_id=proposal_id,
        human_approval_id=human_approval_id,
    )
