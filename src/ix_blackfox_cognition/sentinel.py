"""Cognitive sentinel checks for IX-BlackFox-Cognition.

The cognitive sentinel watches the thinking process before action, memory
promotion, self-improvement, or BlackFox-compatible handoff can proceed.

It detects visible cognition hazards such as scope creep, unsupported certainty,
circular reasoning, fake evidence, model self-approval, memory poisoning,
policy-bypass language, hallucinated references, unsafe authority expansion,
and goal drift.

The sentinel does not execute work. It produces reviewable reports.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, require_invariant


class CognitiveIssueKind(StrEnum):
    """Kinds of cognition hazards detected before action."""

    SCOPE_CREEP = "scope_creep"
    UNSUPPORTED_CERTAINTY = "unsupported_certainty"
    CIRCULAR_REASONING = "circular_reasoning"
    POLICY_BYPASS_LANGUAGE = "policy_bypass_language"
    MODEL_SELF_APPROVAL = "model_self_approval"
    MEMORY_POISONING = "memory_poisoning"
    FAKE_EVIDENCE = "fake_evidence"
    UNSAFE_AUTHORITY_EXPANSION = "unsafe_authority_expansion"
    HALLUCINATED_REFERENCE = "hallucinated_reference"
    GOAL_DRIFT = "goal_drift"


class CognitiveIssueSeverity(StrEnum):
    """Severity levels for sentinel findings."""

    INFO = "info"
    WARNING = "warning"
    REVIEW_REQUIRED = "review_required"
    BLOCKER = "blocker"

    def rank(self) -> int:
        """Return a stable rank for severity comparison."""

        ranks = {
            CognitiveIssueSeverity.INFO: 1,
            CognitiveIssueSeverity.WARNING: 2,
            CognitiveIssueSeverity.REVIEW_REQUIRED: 3,
            CognitiveIssueSeverity.BLOCKER: 4,
        }
        return ranks[self]


@dataclass(frozen=True, slots=True)
class CognitiveArtifactSnapshot:
    """Visible cognition artifacts inspected by the sentinel.

    This object deliberately uses explicit fields instead of opaque logs. Later
    layers can translate claims, beliefs, memory proposals, route decisions,
    plan nodes, and handoff candidates into this inspection shape.
    """

    snapshot_id: str
    subject_id: str
    mission_id: str
    mission_scope: str
    proposed_scope: str
    unsupported_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    contradicted_claim_ids: tuple[str, ...] = field(default_factory=tuple)
    missing_evidence_subject_ids: tuple[str, ...] = field(default_factory=tuple)
    stale_memory_ids: tuple[str, ...] = field(default_factory=tuple)
    self_approval_actor_ids: tuple[str, ...] = field(default_factory=tuple)
    policy_bypass_phrases: tuple[str, ...] = field(default_factory=tuple)
    fake_evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    memory_poisoning_indicators: tuple[str, ...] = field(default_factory=tuple)
    unsafe_authority_requests: tuple[str, ...] = field(default_factory=tuple)
    hallucinated_reference_ids: tuple[str, ...] = field(default_factory=tuple)
    circular_reasoning_markers: tuple[str, ...] = field(default_factory=tuple)
    goal_drift_markers: tuple[str, ...] = field(default_factory=tuple)
    unsupported_certainty_statements: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.snapshot_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive artifact snapshot id cannot be blank.",
        )
        require_invariant(
            bool(self.subject_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive artifact snapshot subject id cannot be blank.",
        )
        require_invariant(
            bool(self.mission_id.strip()),
            FailureMode.UNKNOWN_ACTION_TYPE,
            "Cognitive artifact snapshot mission id cannot be blank.",
        )
        require_invariant(
            bool(self.mission_scope.strip()),
            FailureMode.SCOPE_CREEP,
            "Cognitive artifact snapshot mission scope cannot be blank.",
        )
        require_invariant(
            bool(self.proposed_scope.strip()),
            FailureMode.SCOPE_CREEP,
            "Cognitive artifact snapshot proposed scope cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class CognitiveSentinelIssue:
    """Single reviewable cognition issue detected by the sentinel."""

    issue_id: str
    kind: CognitiveIssueKind
    severity: CognitiveIssueSeverity
    subject_id: str
    statement: str
    failure_mode: FailureMode
    evidence_ids: tuple[str, ...] = field(default_factory=tuple)
    requires_human_review: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.issue_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive sentinel issue id cannot be blank.",
        )
        require_invariant(
            bool(self.subject_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive sentinel issue subject id cannot be blank.",
        )
        require_invariant(
            bool(self.statement.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive sentinel issue statement cannot be blank.",
        )


@dataclass(frozen=True, slots=True)
class CognitiveSentinelReport:
    """Reviewable report generated by the cognitive sentinel."""

    report_id: str
    snapshot_id: str
    subject_id: str
    outcome: DecisionOutcome
    issues: tuple[CognitiveSentinelIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.report_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive sentinel report id cannot be blank.",
        )
        require_invariant(
            bool(self.snapshot_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive sentinel report snapshot id cannot be blank.",
        )
        require_invariant(
            bool(self.subject_id.strip()),
            FailureMode.UNSUPPORTED_CLAIM,
            "Cognitive sentinel report subject id cannot be blank.",
        )

    @property
    def passed(self) -> bool:
        """Return whether the sentinel found no blocking issue."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def blocked(self) -> bool:
        """Return whether the sentinel failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED

    @property
    def requires_human_review(self) -> bool:
        """Return whether any issue requires human review."""

        return any(issue.requires_human_review for issue in self.issues)

    @property
    def blocker_issues(self) -> tuple[CognitiveSentinelIssue, ...]:
        """Return blocker-level issues."""

        return tuple(
            issue for issue in self.issues if issue.severity == CognitiveIssueSeverity.BLOCKER
        )


@dataclass(frozen=True, slots=True)
class CognitiveSentinelPolicy:
    """Policy knobs for cognitive sentinel behavior."""

    policy_id: str = "default-cognitive-sentinel-policy"
    block_scope_creep: bool = True
    block_self_approval: bool = True
    block_fake_evidence: bool = True
    block_memory_poisoning: bool = True
    block_policy_bypass_language: bool = True
    block_unsafe_authority_expansion: bool = True
    block_hallucinated_references: bool = True
    review_unsupported_certainty: bool = True
    review_circular_reasoning: bool = True
    review_goal_drift: bool = True
    review_stale_memory: bool = True

    def __post_init__(self) -> None:
        require_invariant(
            bool(self.policy_id.strip()),
            FailureMode.MISSING_POLICY,
            "Cognitive sentinel policy id cannot be blank.",
        )


class CognitiveSentinel:
    """Deterministic cognitive sentinel."""

    def __init__(self, policy: CognitiveSentinelPolicy | None = None) -> None:
        self.policy = policy or CognitiveSentinelPolicy()

    def inspect(self, snapshot: CognitiveArtifactSnapshot) -> CognitiveSentinelReport:
        """Inspect a cognition snapshot and produce a sentinel report."""

        issues: list[CognitiveSentinelIssue] = []

        issues.extend(self._scope_creep_issues(snapshot, len(issues)))
        issues.extend(self._unsupported_claim_issues(snapshot, len(issues)))
        issues.extend(self._contradiction_issues(snapshot, len(issues)))
        issues.extend(self._missing_evidence_issues(snapshot, len(issues)))
        issues.extend(self._stale_memory_issues(snapshot, len(issues)))
        issues.extend(self._self_approval_issues(snapshot, len(issues)))
        issues.extend(self._policy_bypass_issues(snapshot, len(issues)))
        issues.extend(self._fake_evidence_issues(snapshot, len(issues)))
        issues.extend(self._memory_poisoning_issues(snapshot, len(issues)))
        issues.extend(self._unsafe_authority_issues(snapshot, len(issues)))
        issues.extend(self._hallucinated_reference_issues(snapshot, len(issues)))
        issues.extend(self._circular_reasoning_issues(snapshot, len(issues)))
        issues.extend(self._goal_drift_issues(snapshot, len(issues)))
        issues.extend(self._unsupported_certainty_issues(snapshot, len(issues)))

        outcome = DecisionOutcome.FAIL_CLOSED if self._has_blocker(issues) else DecisionOutcome.ALLOW

        return CognitiveSentinelReport(
            report_id=f"cognitive-sentinel-report:{snapshot.snapshot_id}",
            snapshot_id=snapshot.snapshot_id,
            subject_id=snapshot.subject_id,
            outcome=outcome,
            issues=tuple(issues),
        )

    def _scope_creep_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.block_scope_creep:
            return ()

        mission_scope = self._normalize(snapshot.mission_scope)
        proposed_scope = self._normalize(snapshot.proposed_scope)

        if proposed_scope == mission_scope or proposed_scope in mission_scope:
            return ()

        if mission_scope in proposed_scope:
            return (
                self._issue(
                    snapshot=snapshot,
                    index=start_index,
                    kind=CognitiveIssueKind.SCOPE_CREEP,
                    severity=CognitiveIssueSeverity.REVIEW_REQUIRED,
                    statement="Proposed scope expands beyond the original mission scope.",
                    failure_mode=FailureMode.SCOPE_CREEP,
                ),
            )

        return (
            self._issue(
                snapshot=snapshot,
                index=start_index,
                kind=CognitiveIssueKind.SCOPE_CREEP,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement="Proposed scope does not preserve the mission scope boundary.",
                failure_mode=FailureMode.SCOPE_CREEP,
            ),
        )

    def _unsupported_claim_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.UNSUPPORTED_CERTAINTY,
                severity=CognitiveIssueSeverity.REVIEW_REQUIRED,
                statement=f"Unsupported claim requires evidence before trust: {claim_id}",
                failure_mode=FailureMode.UNSUPPORTED_CLAIM,
            )
            for index, claim_id in enumerate(snapshot.unsupported_claim_ids)
        )

    def _contradiction_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.CIRCULAR_REASONING,
                severity=CognitiveIssueSeverity.REVIEW_REQUIRED,
                statement=f"Contradicted claim requires review before use: {claim_id}",
                failure_mode=FailureMode.CONTRADICTION,
            )
            for index, claim_id in enumerate(snapshot.contradicted_claim_ids)
        )

    def _missing_evidence_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.FAKE_EVIDENCE,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement=f"Subject is missing required evidence: {subject_id}",
                failure_mode=FailureMode.MISSING_EVIDENCE,
            )
            for index, subject_id in enumerate(snapshot.missing_evidence_subject_ids)
        )

    def _stale_memory_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.review_stale_memory:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.MEMORY_POISONING,
                severity=CognitiveIssueSeverity.REVIEW_REQUIRED,
                statement=f"Stale memory must not be treated as current truth: {memory_id}",
                failure_mode=FailureMode.STALE_MEMORY,
            )
            for index, memory_id in enumerate(snapshot.stale_memory_ids)
        )

    def _self_approval_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.block_self_approval:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.MODEL_SELF_APPROVAL,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement=f"Model or non-human actor attempted self-approval: {actor_id}",
                failure_mode=FailureMode.SELF_APPROVAL_ATTEMPT,
            )
            for index, actor_id in enumerate(snapshot.self_approval_actor_ids)
        )

    def _policy_bypass_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.block_policy_bypass_language:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.POLICY_BYPASS_LANGUAGE,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement=f"Policy-bypass language detected: {phrase}",
                failure_mode=FailureMode.POLICY_BYPASS_ATTEMPT,
            )
            for index, phrase in enumerate(snapshot.policy_bypass_phrases)
        )

    def _fake_evidence_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.block_fake_evidence:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.FAKE_EVIDENCE,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement=f"Fake or unverifiable evidence detected: {evidence_id}",
                failure_mode=FailureMode.FAKE_OR_UNVERIFIABLE_EVIDENCE,
            )
            for index, evidence_id in enumerate(snapshot.fake_evidence_ids)
        )

    def _memory_poisoning_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.block_memory_poisoning:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.MEMORY_POISONING,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement=f"Memory poisoning indicator detected: {indicator}",
                failure_mode=FailureMode.MEMORY_CONFLICT,
            )
            for index, indicator in enumerate(snapshot.memory_poisoning_indicators)
        )

    def _unsafe_authority_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.block_unsafe_authority_expansion:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.UNSAFE_AUTHORITY_EXPANSION,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement=f"Unsafe authority expansion requested: {request}",
                failure_mode=FailureMode.UNCLEAR_AUTHORITY,
            )
            for index, request in enumerate(snapshot.unsafe_authority_requests)
        )

    def _hallucinated_reference_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.block_hallucinated_references:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.HALLUCINATED_REFERENCE,
                severity=CognitiveIssueSeverity.BLOCKER,
                statement=f"Hallucinated or unknown reference detected: {reference_id}",
                failure_mode=FailureMode.UNSUPPORTED_CLAIM,
            )
            for index, reference_id in enumerate(snapshot.hallucinated_reference_ids)
        )

    def _circular_reasoning_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.review_circular_reasoning:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.CIRCULAR_REASONING,
                severity=CognitiveIssueSeverity.REVIEW_REQUIRED,
                statement=f"Circular reasoning marker detected: {marker}",
                failure_mode=FailureMode.CONTRADICTION,
            )
            for index, marker in enumerate(snapshot.circular_reasoning_markers)
        )

    def _goal_drift_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.review_goal_drift:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.GOAL_DRIFT,
                severity=CognitiveIssueSeverity.REVIEW_REQUIRED,
                statement=f"Goal drift marker detected: {marker}",
                failure_mode=FailureMode.SCOPE_CREEP,
            )
            for index, marker in enumerate(snapshot.goal_drift_markers)
        )

    def _unsupported_certainty_issues(
        self,
        snapshot: CognitiveArtifactSnapshot,
        start_index: int,
    ) -> tuple[CognitiveSentinelIssue, ...]:
        if not self.policy.review_unsupported_certainty:
            return ()

        return tuple(
            self._issue(
                snapshot=snapshot,
                index=start_index + index,
                kind=CognitiveIssueKind.UNSUPPORTED_CERTAINTY,
                severity=CognitiveIssueSeverity.REVIEW_REQUIRED,
                statement=f"Unsupported certainty statement detected: {statement}",
                failure_mode=FailureMode.UNSUPPORTED_CLAIM,
            )
            for index, statement in enumerate(snapshot.unsupported_certainty_statements)
        )

    def _issue(
        self,
        snapshot: CognitiveArtifactSnapshot,
        index: int,
        kind: CognitiveIssueKind,
        severity: CognitiveIssueSeverity,
        statement: str,
        failure_mode: FailureMode,
    ) -> CognitiveSentinelIssue:
        return CognitiveSentinelIssue(
            issue_id=f"cognitive-sentinel-issue:{snapshot.snapshot_id}:{index}:{kind.value}",
            kind=kind,
            severity=severity,
            subject_id=snapshot.subject_id,
            statement=statement,
            failure_mode=failure_mode,
            requires_human_review=severity.rank() >= CognitiveIssueSeverity.REVIEW_REQUIRED.rank(),
        )

    def _has_blocker(self, issues: list[CognitiveSentinelIssue]) -> bool:
        return any(issue.severity == CognitiveIssueSeverity.BLOCKER for issue in issues)

    def _normalize(self, value: str) -> str:
        return " ".join(value.strip().lower().split())


def inspect_cognition(
    snapshot: CognitiveArtifactSnapshot,
    policy: CognitiveSentinelPolicy | None = None,
) -> CognitiveSentinelReport:
    """Inspect a cognition snapshot with the default cognitive sentinel."""

    return CognitiveSentinel(policy=policy).inspect(snapshot)
