"""Tests for mission envelopes and deterministic mission intake."""

import pytest

from ix_blackfox_cognition import (
    AcceptanceCriterion,
    CognitionInvariantError,
    FailureMode,
    HumanGoal,
    MissionAssumption,
    MissionAssumptionState,
    MissionConstraint,
    MissionConstraintKind,
    MissionEnvelope,
    MissionIntakeEngine,
    MissionIntakeRequest,
    MissionReviewTrigger,
    MissionRisk,
    RiskLevel,
    WorkState,
    structure_mission,
)


def _goal() -> HumanGoal:
    return HumanGoal(
        goal_id="goal:test",
        statement="Build a bounded cognition mission.",
    )


def _constraint() -> MissionConstraint:
    return MissionConstraint(
        constraint_id="constraint:test",
        kind=MissionConstraintKind.SCOPE,
        statement="The mission must remain inside bounded cognition scope.",
    )


def _criterion() -> AcceptanceCriterion:
    return AcceptanceCriterion(
        criterion_id="criterion:test",
        statement="The mission must expose evidence expectations.",
    )


def test_human_goal_rejects_blank_statement() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        HumanGoal(goal_id="goal:blank", statement=" ")

    assert exc_info.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE
    assert exc_info.value.failure.fail_closed is True


def test_mission_constraint_rejects_blank_statement() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        MissionConstraint(
            constraint_id="constraint:blank",
            kind=MissionConstraintKind.SAFETY,
            statement=" ",
        )

    assert exc_info.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_acceptance_criterion_rejects_blank_statement() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        AcceptanceCriterion(
            criterion_id="criterion:blank",
            statement=" ",
        )

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_mission_envelope_requires_at_least_one_constraint() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        MissionEnvelope(
            mission_id="mission:no-constraint",
            goal=_goal(),
            summary="Mission without a constraint should fail closed.",
            acceptance_criteria=(_criterion(),),
        )

    assert exc_info.value.failure.mode == FailureMode.UNKNOWN_ACTION_TYPE


def test_mission_envelope_requires_at_least_one_acceptance_criterion() -> None:
    with pytest.raises(CognitionInvariantError) as exc_info:
        MissionEnvelope(
            mission_id="mission:no-criterion",
            goal=_goal(),
            summary="Mission without evidence-oriented acceptance should fail closed.",
            constraints=(_constraint(),),
        )

    assert exc_info.value.failure.mode == FailureMode.MISSING_EVIDENCE


def test_mission_envelope_reports_bounded_shape() -> None:
    envelope = MissionEnvelope(
        mission_id="mission:bounded",
        goal=_goal(),
        summary="Bounded mission envelope.",
        status=WorkState.DRAFT,
        constraints=(_constraint(),),
        acceptance_criteria=(_criterion(),),
    )

    assert envelope.is_bounded
    assert envelope.highest_risk == RiskLevel.LOW
    assert envelope.requires_human_review is False
    assert envelope.has_unverified_assumptions is False


def test_mission_envelope_reports_highest_risk() -> None:
    envelope = MissionEnvelope(
        mission_id="mission:risk",
        goal=_goal(),
        summary="Mission with visible risk.",
        constraints=(_constraint(),),
        acceptance_criteria=(_criterion(),),
        risks=(
            MissionRisk(
                risk_id="risk:low",
                statement="Low risk.",
                level=RiskLevel.LOW,
            ),
            MissionRisk(
                risk_id="risk:critical",
                statement="Critical risk.",
                level=RiskLevel.CRITICAL,
            ),
        ),
    )

    assert envelope.highest_risk == RiskLevel.CRITICAL


def test_mission_envelope_reports_unverified_assumptions() -> None:
    envelope = MissionEnvelope(
        mission_id="mission:assumption",
        goal=_goal(),
        summary="Mission with visible assumptions.",
        constraints=(_constraint(),),
        acceptance_criteria=(_criterion(),),
        assumptions=(
            MissionAssumption(
                assumption_id="assumption:unverified",
                statement="This assumption still needs evidence.",
                state=MissionAssumptionState.NEEDS_EVIDENCE,
            ),
        ),
    )

    assert envelope.has_unverified_assumptions is True


def test_structure_mission_adds_default_governance_boundaries() -> None:
    result = structure_mission(
        MissionIntakeRequest(
            goal_id="goal:intake",
            statement="Create a governed cognition mission.",
        )
    )

    envelope = result.envelope

    assert result.bounded
    assert envelope.mission_id == "mission:goal:intake"
    assert envelope.goal.statement == "Create a governed cognition mission."
    assert len(envelope.constraints) == 3
    assert len(envelope.acceptance_criteria) == 1
    assert len(envelope.rollback_needs) == 1
    assert envelope.review_checkpoints == ()
    assert "self_approve" in envelope.forbidden_actions
    assert "claim_agi" in envelope.forbidden_actions
    assert "execute_operational_action_without_blackfox_handoff" in envelope.forbidden_actions


def test_structure_mission_preserves_custom_constraints_and_acceptance_criteria() -> None:
    result = structure_mission(
        MissionIntakeRequest(
            goal_id="goal:custom",
            statement="Preserve caller-supplied mission details.",
            constraints=(
                MissionConstraint(
                    constraint_id="constraint:custom",
                    kind=MissionConstraintKind.LEGAL,
                    statement="Respect the source-available license boundary.",
                ),
            ),
            acceptance_criteria=(
                AcceptanceCriterion(
                    criterion_id="criterion:custom",
                    statement="Caller-provided acceptance criterion remains visible.",
                ),
            ),
        )
    )

    constraint_ids = {constraint.constraint_id for constraint in result.envelope.constraints}
    criterion_ids = {
        criterion.criterion_id for criterion in result.envelope.acceptance_criteria
    }

    assert "constraint:custom" in constraint_ids
    assert "criterion:custom" in criterion_ids
    assert "constraint:goal:custom:evidence-required" in constraint_ids
    assert "criterion:goal:custom:bounded-envelope" in criterion_ids


def test_structure_mission_adds_high_risk_review_checkpoint() -> None:
    result = structure_mission(
        MissionIntakeRequest(
            goal_id="goal:high-risk",
            statement="Prepare a high-risk BlackFox handoff candidate.",
            risks=(
                MissionRisk(
                    risk_id="risk:high",
                    statement="The handoff may affect guarded execution behavior.",
                    level=RiskLevel.HIGH,
                ),
            ),
        )
    )

    envelope = result.envelope

    assert envelope.requires_human_review is True
    assert result.added_default_review_checkpoints == ("review:goal:high-risk:high-risk",)
    assert envelope.review_checkpoints[0].trigger == MissionReviewTrigger.HIGH_RISK


def test_structure_mission_keeps_duplicate_forbidden_actions_unique() -> None:
    result = structure_mission(
        MissionIntakeRequest(
            goal_id="goal:forbidden",
            statement="Preserve explicit forbidden action boundaries.",
            forbidden_actions=("self_approve", "custom_forbidden_action"),
        )
    )

    forbidden_actions = result.envelope.forbidden_actions

    assert forbidden_actions.count("self_approve") == 1
    assert "custom_forbidden_action" in forbidden_actions


def test_mission_intake_engine_can_use_custom_summary_and_mission_id() -> None:
    result = MissionIntakeEngine().structure(
        MissionIntakeRequest(
            goal_id="goal:custom-id",
            mission_id="mission:explicit",
            statement="Build a mission with an explicit identifier.",
            summary="Explicit mission summary.",
        )
    )

    assert result.envelope.mission_id == "mission:explicit"
    assert result.envelope.summary == "Explicit mission summary."


def test_mission_intake_truncates_long_default_summary() -> None:
    long_statement = " ".join(["bounded"] * 40)

    result = structure_mission(
        MissionIntakeRequest(
            goal_id="goal:long-summary",
            statement=long_statement,
        )
    )

    assert len(result.envelope.summary) <= 160
    assert result.envelope.summary.endswith("...")
