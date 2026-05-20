"""Validation logic for proof-carrying plan graphs.

Plan validation rejects vague, unbounded, under-evidenced, or unsafe plans before
they can become cognitive work packages or BlackFox-compatible handoff
candidates.

The validator does not execute plans. It verifies that the plan is inspectable:
- every node has scope,
- every node has proof obligations,
- every node has an evidence contract,
- action-adjacent nodes have review and rollback requirements,
- graph roots and terminal nodes are explicit,
- required paths are reachable,
- dependency cycles are rejected,
- falsified evidence contracts block trust.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from ix_blackfox_cognition.core import DecisionOutcome, FailureMode, RiskLevel, WorkState
from ix_blackfox_cognition.planning import (
    PlanDependencyKind,
    PlanEdge,
    PlanGraph,
    PlanNode,
)


class PlanValidationSeverity(StrEnum):
    """Severity levels for plan validation findings."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    BLOCKER = "blocker"


@dataclass(frozen=True, slots=True)
class PlanValidationFinding:
    """A validation finding that explains why a plan is or is not acceptable."""

    finding_id: str
    subject_id: str
    message: str
    failure_mode: FailureMode
    severity: PlanValidationSeverity = PlanValidationSeverity.BLOCKER
    requires_human_review: bool = True


@dataclass(frozen=True, slots=True)
class PlanValidationPolicy:
    """Policy knobs for conservative plan graph validation."""

    policy_id: str = "default-plan-validation-policy"
    require_nodes: bool = True
    require_root_nodes: bool = True
    require_terminal_nodes: bool = True
    require_scope_for_all_nodes: bool = True
    require_proof_for_all_nodes: bool = True
    require_evidence_contract_for_all_nodes: bool = True
    require_review_for_action_candidates: bool = True
    require_rollback_for_action_candidates: bool = True
    require_review_for_high_risk_nodes: bool = True
    high_risk_threshold: RiskLevel = RiskLevel.HIGH
    reject_dependency_cycles: bool = True
    require_terminal_reachability_from_root: bool = True


@dataclass(frozen=True, slots=True)
class PlanValidationResult:
    """Result of validating a proof-carrying plan graph."""

    graph_id: str
    outcome: DecisionOutcome
    findings: tuple[PlanValidationFinding, ...] = field(default_factory=tuple)

    @property
    def valid(self) -> bool:
        """Return whether the plan graph passed validation."""

        return self.outcome == DecisionOutcome.ALLOW

    @property
    def failed_closed(self) -> bool:
        """Return whether the plan graph failed closed."""

        return self.outcome == DecisionOutcome.FAIL_CLOSED

    @property
    def blocker_findings(self) -> tuple[PlanValidationFinding, ...]:
        """Return blocker-level findings."""

        return tuple(
            finding
            for finding in self.findings
            if finding.severity == PlanValidationSeverity.BLOCKER
        )


class PlanGraphValidator:
    """Conservative proof-carrying plan graph validator."""

    def __init__(self, policy: PlanValidationPolicy | None = None) -> None:
        self.policy = policy or PlanValidationPolicy()

    def validate(self, graph: PlanGraph) -> PlanValidationResult:
        """Validate that a plan graph is bounded, inspectable, and evidence-ready."""

        findings: list[PlanValidationFinding] = []

        findings.extend(self._validate_graph_shape(graph))
        findings.extend(self._validate_nodes(graph))
        findings.extend(self._validate_dependency_structure(graph))

        if findings:
            return PlanValidationResult(
                graph_id=graph.graph_id,
                outcome=DecisionOutcome.FAIL_CLOSED,
                findings=tuple(findings),
            )

        return PlanValidationResult(
            graph_id=graph.graph_id,
            outcome=DecisionOutcome.ALLOW,
            findings=(),
        )

    def _validate_graph_shape(self, graph: PlanGraph) -> tuple[PlanValidationFinding, ...]:
        findings: list[PlanValidationFinding] = []

        if self.policy.require_nodes and not graph.nodes:
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=graph.graph_id,
                    index=len(findings),
                    message="Plan graph must contain at least one plan node.",
                    failure_mode=FailureMode.UNKNOWN_ACTION_TYPE,
                )
            )

        if self.policy.require_root_nodes and not graph.root_node_ids:
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=graph.graph_id,
                    index=len(findings),
                    message="Plan graph must identify at least one root node.",
                    failure_mode=FailureMode.UNKNOWN_ACTION_TYPE,
                )
            )

        if self.policy.require_terminal_nodes and not graph.terminal_node_ids:
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=graph.graph_id,
                    index=len(findings),
                    message="Plan graph must identify at least one terminal node.",
                    failure_mode=FailureMode.UNKNOWN_ACTION_TYPE,
                )
            )

        return tuple(findings)

    def _validate_nodes(self, graph: PlanGraph) -> tuple[PlanValidationFinding, ...]:
        findings: list[PlanValidationFinding] = []

        for node in graph.nodes:
            findings.extend(self._validate_node_scope(graph, node, len(findings)))
            findings.extend(self._validate_node_proof(graph, node, len(findings)))
            findings.extend(self._validate_node_risk_review(graph, node, len(findings)))
            findings.extend(self._validate_node_action_boundaries(graph, node, len(findings)))

        return tuple(findings)

    def _validate_node_scope(
        self,
        graph: PlanGraph,
        node: PlanNode,
        start_index: int,
    ) -> tuple[PlanValidationFinding, ...]:
        findings: list[PlanValidationFinding] = []

        if self.policy.require_scope_for_all_nodes and not self._has_text(node.scope):
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=node.node_id,
                    index=start_index,
                    message="Plan node must define bounded scope.",
                    failure_mode=FailureMode.SCOPE_CREEP,
                )
            )

        return tuple(findings)

    def _validate_node_proof(
        self,
        graph: PlanGraph,
        node: PlanNode,
        start_index: int,
    ) -> tuple[PlanValidationFinding, ...]:
        findings: list[PlanValidationFinding] = []

        if self.policy.require_proof_for_all_nodes and not node.proof_obligations:
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=node.node_id,
                    index=start_index + len(findings),
                    message="Plan node must carry proof obligations.",
                    failure_mode=FailureMode.MISSING_EVIDENCE,
                )
            )

        if self.policy.require_evidence_contract_for_all_nodes and node.evidence_contract is None:
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=node.node_id,
                    index=start_index + len(findings),
                    message="Plan node must carry an evidence contract.",
                    failure_mode=FailureMode.MISSING_EVIDENCE,
                )
            )

        if node.evidence_contract is not None:
            if node.evidence_contract.falsified:
                findings.append(
                    self._finding(
                        graph=graph,
                        subject_id=node.node_id,
                        index=start_index + len(findings),
                        message="Plan node evidence contract is falsified.",
                        failure_mode=FailureMode.CONTRADICTION,
                    )
                )
            elif not node.evidence_contract.satisfied:
                findings.append(
                    self._finding(
                        graph=graph,
                        subject_id=node.node_id,
                        index=start_index + len(findings),
                        message="Plan node evidence contract has unsatisfied requirements.",
                        failure_mode=FailureMode.MISSING_EVIDENCE,
                    )
                )

        return tuple(findings)

    def _validate_node_risk_review(
        self,
        graph: PlanGraph,
        node: PlanNode,
        start_index: int,
    ) -> tuple[PlanValidationFinding, ...]:
        findings: list[PlanValidationFinding] = []

        if (
            self.policy.require_review_for_high_risk_nodes
            and node.risk_level.rank() >= self.policy.high_risk_threshold.rank()
            and not node.review_requirements
        ):
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=node.node_id,
                    index=start_index,
                    message="High-risk plan node must define review requirements.",
                    failure_mode=FailureMode.HUMAN_REVIEW_REQUIRED,
                )
            )

        return tuple(findings)

    def _validate_node_action_boundaries(
        self,
        graph: PlanGraph,
        node: PlanNode,
        start_index: int,
    ) -> tuple[PlanValidationFinding, ...]:
        findings: list[PlanValidationFinding] = []

        if self.policy.require_review_for_action_candidates and node.is_action_candidate:
            if not node.review_requirements:
                findings.append(
                    self._finding(
                        graph=graph,
                        subject_id=node.node_id,
                        index=start_index + len(findings),
                        message="Action-adjacent plan node must define review requirements.",
                        failure_mode=FailureMode.HUMAN_REVIEW_REQUIRED,
                    )
                )

        if self.policy.require_rollback_for_action_candidates and node.is_action_candidate:
            if not node.rollback_conditions:
                findings.append(
                    self._finding(
                        graph=graph,
                        subject_id=node.node_id,
                        index=start_index + len(findings),
                        message="Action-adjacent plan node must define rollback conditions.",
                        failure_mode=FailureMode.UNKNOWN_ACTION_TYPE,
                    )
                )

        return tuple(findings)

    def _validate_dependency_structure(
        self,
        graph: PlanGraph,
    ) -> tuple[PlanValidationFinding, ...]:
        findings: list[PlanValidationFinding] = []

        if self.policy.reject_dependency_cycles and self._has_cycle(graph.edges):
            findings.append(
                self._finding(
                    graph=graph,
                    subject_id=graph.graph_id,
                    index=len(findings),
                    message="Plan graph dependency structure cannot contain cycles.",
                    failure_mode=FailureMode.CONTRADICTION,
                )
            )

        if (
            self.policy.require_terminal_reachability_from_root
            and graph.nodes
            and graph.root_node_ids
            and graph.terminal_node_ids
        ):
            reachable = self._reachable_nodes(graph)
            for terminal_id in graph.terminal_node_ids:
                if terminal_id not in reachable:
                    findings.append(
                        self._finding(
                            graph=graph,
                            subject_id=terminal_id,
                            index=len(findings),
                            message="Terminal plan node must be reachable from a root node.",
                            failure_mode=FailureMode.UNSUPPORTED_CLAIM,
                        )
                    )

        return tuple(findings)

    def _reachable_nodes(self, graph: PlanGraph) -> set[str]:
        adjacency: dict[str, set[str]] = {node.node_id: set() for node in graph.nodes}

        for edge in graph.edges:
            if edge.dependency in self._forward_dependencies():
                adjacency.setdefault(edge.source_node_id, set()).add(edge.target_node_id)

        reachable: set[str] = set()
        stack = list(graph.root_node_ids)

        while stack:
            node_id = stack.pop()
            if node_id in reachable:
                continue
            reachable.add(node_id)
            stack.extend(sorted(adjacency.get(node_id, set()) - reachable))

        return reachable

    def _has_cycle(self, edges: tuple[PlanEdge, ...]) -> bool:
        adjacency: dict[str, set[str]] = {}

        for edge in edges:
            if edge.dependency not in self._cycle_checked_dependencies():
                continue
            adjacency.setdefault(edge.source_node_id, set()).add(edge.target_node_id)

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(node_id: str) -> bool:
            if node_id in visiting:
                return True
            if node_id in visited:
                return False

            visiting.add(node_id)
            for next_node_id in adjacency.get(node_id, set()):
                if visit(next_node_id):
                    return True
            visiting.remove(node_id)
            visited.add(node_id)
            return False

        return any(visit(node_id) for node_id in adjacency)

    def _forward_dependencies(self) -> tuple[PlanDependencyKind, ...]:
        return (
            PlanDependencyKind.REQUIRES,
            PlanDependencyKind.ENABLES,
            PlanDependencyKind.VERIFIES,
            PlanDependencyKind.REVIEWS,
            PlanDependencyKind.HANDS_OFF_TO,
        )

    def _cycle_checked_dependencies(self) -> tuple[PlanDependencyKind, ...]:
        return (
            PlanDependencyKind.REQUIRES,
            PlanDependencyKind.ENABLES,
            PlanDependencyKind.VERIFIES,
            PlanDependencyKind.REVIEWS,
            PlanDependencyKind.HANDS_OFF_TO,
        )

    def _finding(
        self,
        graph: PlanGraph,
        subject_id: str,
        index: int,
        message: str,
        failure_mode: FailureMode,
    ) -> PlanValidationFinding:
        return PlanValidationFinding(
            finding_id=f"plan-finding:{graph.graph_id}:{index}",
            subject_id=subject_id,
            message=message,
            failure_mode=failure_mode,
        )

    def _has_text(self, value: str | None) -> bool:
        return value is not None and bool(value.strip())


def validate_plan_graph(
    graph: PlanGraph,
    policy: PlanValidationPolicy | None = None,
) -> PlanValidationResult:
    """Validate a plan graph with the default conservative validator."""

    return PlanGraphValidator(policy=policy).validate(graph)
