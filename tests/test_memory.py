"""Tests for memory immune-system models and quarantine/promotion logic."""

import pytest

from ix_blackfox_cognition import (
    ActorKind,
    CognitionInvariantError,
    DecisionOutcome,
    FailureMode,
    MemoryConflict,
    MemoryConflictKind,
    MemoryDecisionKind,
    MemoryImmunePolicy,
    MemoryImmuneSystem,
    MemoryRecord,
    MemoryRecordKind,
    MemorySource,
    MemoryState,
    MemoryStore,
    MemoryUpdateKind,
    MemoryUpdateProposal,
    RiskLevel,
    promote_memory_proposal,
    submit_memory_proposal,
)


def _source(actor_kind: ActorKind = ActorKind.MODEL, actor_id: str = "model:memory-reviewer") -> MemorySource:
    return MemorySource(
        actor_kind=actor_kind,
        actor_id=actor_id,
        description="Memory source used by tests.",
    )


def _record(
    memory_id: str = "memory:test",
    *,
    state: MemoryState = MemoryState.PROPOSED,
    evidence_ids: tuple[str, ...] = (),
    human_approval_id: str | None = None,
    risk_level: RiskLevel = RiskLevel.LOW,
    supersedes_memory_ids: tuple[str, ...] = (),
    claim_ids: tuple[str, ...] = (),
    belief_ids: tuple[str, ...] = (),
    plan_node_ids: tuple[str, ...] = (),
    work_package_ids: tuple[str, ...] = (),
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        kind=MemoryRecordKind.BELIEF_SUMMARY,
        statement=f"Governed memory statement for {memory_id}.",
        source=_source(),
        state=state,
        risk_level=risk_level,
        evidence_ids=evidence_ids,
        human_approval_id=human_approval_id,
        supersedes_memory_ids=supersedes_memory_ids,
        claim_ids=claim_ids,
        belief_ids=belief_ids,
        plan_node_ids=plan_node_ids,
        work_package_ids=work_package_ids,
    )


def _proposal(
    proposal_id: str = "proposal:test",
    *,
    update_kind: MemoryUpdateKind = MemoryUpdateKind.ADD,
    proposed_record: MemoryRecord | None = None,
    target_memory_id: str | None = None,
    evidence_ids: tuple[str, ...] = (),
    conflict_ids: tuple[str, ...] = (),
    requires_human_review: bool = True,
) -> MemoryUpdateProposal:
    return MemoryUpdateProposal(
        proposal_id=proposal_id,
        update_kind=update_kind,
        proposed_record=proposed_record or _record(),
        proposer=_source(actor_id="model:proposer"),
        rationale="The memory update is proposed for bounded cognition review.",
        target_memory_id=target_memory_id,
        evidence_ids=evidence_ids,
        conflict_ids=conflict_ids,
        requires_human_review=requires_human_review,
    )


def test_memory_source_rejects_blank_actor_id() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _source(actor_id=" ")

    assert exc_info.value.failure.mode == FailureMode.UNSUPPORTED_CLAIM


def test_promoted_memory_requires_evidence_or_human_approval() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _record(state=MemoryState.PROMOTED)

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_promoted_memory_with_evidence_is_trust_eligible() -> None:
    record = _record(
        state=MemoryState.PROMOTED,
        evidence_ids=("evidence:verified-memory",),
    )

    assert record.has_evidence is True
    assert record.has_human_approval is False
    assert record.trust_eligible is True
    assert record.blocked_from_use is False


def test_promoted_memory_with_human_approval_is_trust_eligible() -> None:
    record = _record(
        state=MemoryState.PROMOTED,
        human_approval_id="human-approval:memory",
    )

    assert record.has_evidence is False
    assert record.has_human_approval is True
    assert record.trust_eligible is True


def test_superseded_memory_requires_superseding_memory_id() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _record(state=MemoryState.SUPERSEDED)

    assert exc_info.value.failure.mode == FailureMode.STALE_MEMORY


def test_expired_memory_requires_expiration_marker() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _record(state=MemoryState.EXPIRED)

    assert exc_info.value.failure.mode == FailureMode.STALE_MEMORY


def test_memory_update_proposal_requires_target_for_non_add_updates() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _proposal(update_kind=MemoryUpdateKind.REVISE)

    assert exc_info.value.failure.mode == FailureMode.SILENT_STATE_MUTATION


def test_memory_promotion_proposal_requires_evidence_or_human_approval() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        _proposal(
            update_kind=MemoryUpdateKind.PROMOTE,
            target_memory_id="memory:target",
        )

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_memory_store_rejects_duplicate_record_ids() -> None:
    record = _record("memory:duplicate")

    with pytest.raises(CognitionInvariantError) as exc_info:
        MemoryStore(
            store_id="memory-store:duplicate-records",
            records=(record, record),
        )

    assert exc_info.value.failure.mode == FailureMode.CONTRADICTION


def test_memory_store_reports_promoted_and_blocked_records() -> None:
    promoted = _record(
        "memory:promoted",
        state=MemoryState.PROMOTED,
        evidence_ids=("evidence:promoted",),
    )
    quarantined = _record("memory:quarantined", state=MemoryState.QUARANTINED)
    rejected = _record("memory:rejected", state=MemoryState.REJECTED)
    store = MemoryStore(
        store_id="memory-store:views",
        records=(promoted, quarantined, rejected),
    )

    assert store.promoted_records == (promoted,)
    assert store.blocked_records == (quarantined, rejected)


def test_submit_add_proposal_is_immutable_and_does_not_silently_promote_memory() -> None:
    store = MemoryStore(store_id="memory-store:submit")
    proposal = _proposal(
        proposal_id="proposal:add",
        proposed_record=_record("memory:add"),
        evidence_ids=("evidence:add",),
    )

    result = submit_memory_proposal(store, proposal)

    assert result.decision.allowed
    assert result.decision.kind == MemoryDecisionKind.SUBMIT_PROPOSAL
    assert result.decision.outcome == DecisionOutcome.ALLOW
    assert store.proposals == ()
    assert store.records == ()
    assert result.store.proposals == (proposal,)
    assert result.store.records == ()


def test_submit_duplicate_proposal_fails_closed() -> None:
    proposal = _proposal(proposal_id="proposal:duplicate")
    store = MemoryStore(
        store_id="memory-store:duplicate-proposal",
        proposals=(proposal,),
    )

    result = submit_memory_proposal(store, proposal)

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.CONTRADICTION,)
    assert result.store == store


def test_submit_add_proposal_rejects_reused_existing_memory_id() -> None:
    existing = _record("memory:existing")
    proposal = _proposal(
        proposal_id="proposal:reused-memory-id",
        proposed_record=_record("memory:existing"),
    )
    store = MemoryStore(
        store_id="memory-store:reused-memory-id",
        records=(existing,),
    )

    result = submit_memory_proposal(store, proposal)

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.CONTRADICTION,)
    assert result.store == store


def test_submit_proposal_quarantines_known_conflicts() -> None:
    proposal = _proposal(
        proposal_id="proposal:known-conflict",
        proposed_record=_record("memory:known-conflict"),
        conflict_ids=("conflict:existing",),
    )
    store = MemoryStore(store_id="memory-store:known-conflict")

    result = submit_memory_proposal(store, proposal)

    assert result.decision.quarantined
    assert result.decision.kind == MemoryDecisionKind.QUARANTINE
    assert result.decision.failure_modes == (FailureMode.MEMORY_CONFLICT,)
    assert result.decision.requires_human_review is True
    assert len(result.store.conflicts) == 1
    assert result.store.conflicts[0].kind == MemoryConflictKind.CONTRADICTS_EXISTING_MEMORY
    assert len(result.store.quarantines) == 1
    assert result.store.records[0].state == MemoryState.QUARANTINED


def test_submit_proposal_quarantines_circular_memory_reference() -> None:
    record = _record(
        "memory:circular",
        supersedes_memory_ids=("memory:circular",),
    )
    proposal = _proposal(
        proposal_id="proposal:circular",
        proposed_record=record,
    )
    store = MemoryStore(store_id="memory-store:circular")

    result = submit_memory_proposal(store, proposal)

    assert result.decision.quarantined
    assert result.decision.failure_modes == (FailureMode.MEMORY_CONFLICT,)
    assert result.store.conflicts[0].kind == MemoryConflictKind.CIRCULAR_MEMORY_REFERENCE


def test_submit_proposal_quarantines_evidence_laundering() -> None:
    old_memory = _record("memory:old", state=MemoryState.PROPOSED)
    proposed = _record(
        "memory:new",
        supersedes_memory_ids=("memory:old",),
    )
    proposal = _proposal(
        proposal_id="proposal:evidence-laundering",
        proposed_record=proposed,
    )
    store = MemoryStore(
        store_id="memory-store:evidence-laundering",
        records=(old_memory,),
    )

    result = submit_memory_proposal(store, proposal)

    assert result.decision.quarantined
    assert result.decision.failure_modes == (FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE,)
    assert result.store.conflicts[0].kind == MemoryConflictKind.EVIDENCE_LAUNDERING


def test_submit_non_add_proposal_quarantines_missing_target_memory() -> None:
    proposal = _proposal(
        proposal_id="proposal:missing-target",
        update_kind=MemoryUpdateKind.REVISE,
        proposed_record=_record("memory:revision"),
        target_memory_id="memory:missing",
        evidence_ids=("evidence:revision",),
    )
    store = MemoryStore(store_id="memory-store:missing-target")

    result = submit_memory_proposal(store, proposal)

    assert result.decision.quarantined
    assert result.decision.failure_modes == (FailureMode.STALE_MEMORY,)
    assert result.store.conflicts[0].kind == MemoryConflictKind.STALE_SOURCE


def test_submit_non_add_proposal_quarantines_rejected_target_memory() -> None:
    rejected = _record("memory:rejected-target", state=MemoryState.REJECTED)
    proposal = _proposal(
        proposal_id="proposal:rejected-target",
        update_kind=MemoryUpdateKind.REVISE,
        proposed_record=_record("memory:revision"),
        target_memory_id="memory:rejected-target",
        evidence_ids=("evidence:revision",),
    )
    store = MemoryStore(
        store_id="memory-store:rejected-target",
        records=(rejected,),
    )

    result = submit_memory_proposal(store, proposal)

    assert result.decision.quarantined
    assert result.decision.failure_modes == (FailureMode.STALE_MEMORY,)
    assert result.store.conflicts[0].kind == MemoryConflictKind.STALE_SOURCE


def test_promote_unknown_proposal_fails_closed() -> None:
    store = MemoryStore(store_id="memory-store:unknown-proposal")

    result = promote_memory_proposal(
        store,
        "proposal:missing",
        human_approval_id="human-approval:memory",
    )

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.UNSUPPORTED_CLAIM,)
    assert result.store == store


def test_promote_submitted_proposal_requires_evidence_or_human_approval() -> None:
    proposal = _proposal(
        proposal_id="proposal:no-evidence",
        proposed_record=_record("memory:no-evidence"),
        requires_human_review=False,
    )
    submitted = submit_memory_proposal(
        MemoryStore(store_id="memory-store:no-evidence"),
        proposal,
    ).store

    result = promote_memory_proposal(submitted, "proposal:no-evidence")

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.MISSING_EVIDENCE,)


def test_promote_submitted_proposal_requires_human_review_by_default() -> None:
    proposal = _proposal(
        proposal_id="proposal:needs-human",
        proposed_record=_record("memory:needs-human"),
        evidence_ids=("evidence:needs-human",),
    )
    submitted = submit_memory_proposal(
        MemoryStore(store_id="memory-store:needs-human"),
        proposal,
    ).store

    result = promote_memory_proposal(submitted, "proposal:needs-human")

    assert result.decision.outcome == DecisionOutcome.REVIEW_REQUIRED
    assert result.decision.failure_modes == (FailureMode.HUMAN_REVIEW_REQUIRED,)
    assert result.decision.requires_human_review is True


def test_promote_submitted_proposal_with_evidence_and_human_approval_adds_promoted_record() -> None:
    proposal = _proposal(
        proposal_id="proposal:promote",
        proposed_record=_record("memory:promote"),
        evidence_ids=("evidence:promote",),
    )
    submitted = submit_memory_proposal(
        MemoryStore(store_id="memory-store:promote"),
        proposal,
    ).store

    result = promote_memory_proposal(
        submitted,
        "proposal:promote",
        human_approval_id="human-approval:promote",
    )

    promoted = result.store.record_by_id("memory:promote")

    assert result.decision.allowed
    assert result.decision.kind == MemoryDecisionKind.PROMOTE
    assert promoted is not None
    assert promoted.state == MemoryState.PROMOTED
    assert promoted.evidence_ids == ("evidence:promote",)
    assert promoted.human_approval_id == "human-approval:promote"
    assert promoted.trust_eligible is True
    assert submitted.record_by_id("memory:promote") is None


def test_promote_high_risk_proposal_requires_human_review_even_if_proposal_does_not() -> None:
    proposal = _proposal(
        proposal_id="proposal:high-risk",
        proposed_record=_record(
            "memory:high-risk",
            risk_level=RiskLevel.HIGH,
        ),
        evidence_ids=("evidence:high-risk",),
        requires_human_review=False,
    )
    submitted = submit_memory_proposal(
        MemoryStore(store_id="memory-store:high-risk"),
        proposal,
    ).store

    result = promote_memory_proposal(submitted, "proposal:high-risk")

    assert result.decision.outcome == DecisionOutcome.REVIEW_REQUIRED
    assert result.decision.failure_modes == (FailureMode.HUMAN_REVIEW_REQUIRED,)


def test_promote_can_relax_high_risk_human_requirement_by_policy() -> None:
    proposal = _proposal(
        proposal_id="proposal:policy-relaxed",
        proposed_record=_record(
            "memory:policy-relaxed",
            risk_level=RiskLevel.HIGH,
        ),
        evidence_ids=("evidence:policy-relaxed",),
        requires_human_review=False,
    )
    submitted = submit_memory_proposal(
        MemoryStore(store_id="memory-store:policy-relaxed"),
        proposal,
    ).store
    policy = MemoryImmunePolicy(require_human_for_high_risk=False)

    result = promote_memory_proposal(
        submitted,
        "proposal:policy-relaxed",
        policy=policy,
    )

    promoted = result.store.record_by_id("memory:policy-relaxed")

    assert result.decision.allowed
    assert promoted is not None
    assert promoted.state == MemoryState.PROMOTED


def test_promote_proposal_with_store_conflict_fails_closed() -> None:
    proposal = _proposal(
        proposal_id="proposal:conflicted",
        proposed_record=_record("memory:conflicted"),
        evidence_ids=("evidence:conflicted",),
        requires_human_review=False,
    )
    conflict = MemoryConflict(
        conflict_id="conflict:conflicted",
        kind=MemoryConflictKind.CONTRADICTS_EXISTING_MEMORY,
        memory_id="memory:conflicted",
        statement="Unresolved conflict blocks promotion.",
    )
    store = MemoryStore(
        store_id="memory-store:conflicted",
        proposals=(proposal,),
        conflicts=(conflict,),
    )

    result = promote_memory_proposal(store, "proposal:conflicted")

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.MEMORY_CONFLICT,)
    assert result.store == store


def test_reject_proposal_preserves_audit_trail_and_adds_rejected_record() -> None:
    proposal = _proposal(
        proposal_id="proposal:reject",
        proposed_record=_record("memory:reject"),
    )
    submitted = submit_memory_proposal(
        MemoryStore(store_id="memory-store:reject"),
        proposal,
    ).store

    result = MemoryImmuneSystem().reject_proposal(
        submitted,
        "proposal:reject",
        reason="The proposed memory was rejected during review.",
    )

    rejected = result.store.record_by_id("memory:reject")

    assert result.decision.allowed
    assert result.decision.kind == MemoryDecisionKind.REJECT
    assert rejected is not None
    assert rejected.state == MemoryState.REJECTED
    assert result.store.proposals == submitted.proposals


def test_reject_unknown_proposal_fails_closed() -> None:
    store = MemoryStore(store_id="memory-store:reject-unknown")

    result = MemoryImmuneSystem().reject_proposal(
        store,
        "proposal:missing",
        reason="Unknown proposal cannot be rejected.",
    )

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.UNSUPPORTED_CLAIM,)


def test_reject_proposal_requires_nonblank_reason() -> None:
    proposal = _proposal(
        proposal_id="proposal:blank-rejection",
        proposed_record=_record("memory:blank-rejection"),
    )
    submitted = submit_memory_proposal(
        MemoryStore(store_id="memory-store:blank-rejection"),
        proposal,
    ).store

    result = MemoryImmuneSystem().reject_proposal(
        submitted,
        "proposal:blank-rejection",
        reason=" ",
    )

    assert result.decision.failed_closed
    assert result.decision.failure_modes == (FailureMode.UNSUPPORTED_CLAIM,)
