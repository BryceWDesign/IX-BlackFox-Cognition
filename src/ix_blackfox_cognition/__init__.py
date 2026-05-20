"""IX-BlackFox-Cognition public API.

IX-BlackFox-Cognition is a source-available governed cognition substrate for
evidence-bound AI-assisted engineering workflows.

The current public API exports stable project metadata, dependency-light core
types, authority-kernel models, authority firewall logic, mission-envelope
models, and deterministic mission intake. Higher-level planning, memory,
routing, sentinel, and handoff APIs will be exported only after their modules
exist and are covered by tests.
"""

from ix_blackfox_cognition.authority import (
    AuthorityBoundary,
    AuthorityDecisionRecord,
    AuthorityLevel,
    AuthorityRequest,
    AuthoritySnapshot,
    CognitionPermission,
    ForbiddenAction,
    HumanAuthorityRequirement,
)
from ix_blackfox_cognition.authority_firewall import (
    AuthorityFirewall,
    AuthorityFirewallPolicy,
    evaluate_authority_request,
)
from ix_blackfox_cognition.core import (
    ActorKind,
    AuthorityError,
    ClaimState,
    CognitionError,
    CognitionInvariantError,
    CognitionMemoryError,
    DecisionOutcome,
    EvidenceError,
    EvidenceState,
    FailureMode,
    FailureRecord,
    MemoryState,
    PolicyError,
    RiskLevel,
    SelfImprovementError,
    WorkState,
    fail_closed,
    require_invariant,
)
from ix_blackfox_cognition.metadata import (
    CORE_DOCTRINE,
    FOUNDATIONAL_LAW,
    LICENSE_NAME,
    PACKAGE_NAME,
    PROJECT_NAME,
    PROHIBITED_CLAIMS,
    PUBLIC_DESCRIPTION,
    RESEARCH_STATUS,
    VERSION,
    PackageIdentity,
    get_package_identity,
)
from ix_blackfox_cognition.mission import (
    AcceptanceCriterion,
    HumanGoal,
    MissionAssumption,
    MissionAssumptionState,
    MissionConstraint,
    MissionConstraintKind,
    MissionEnvelope,
    MissionReviewTrigger,
    MissionRisk,
    ReviewCheckpoint,
    RollbackNeed,
)
from ix_blackfox_cognition.mission_intake import (
    DEFAULT_FORBIDDEN_ACTIONS,
    MissionIntakeDefaults,
    MissionIntakeEngine,
    MissionIntakeRequest,
    MissionIntakeResult,
    structure_mission,
)

__version__ = VERSION

__all__ = [
    "DEFAULT_FORBIDDEN_ACTIONS",
    "AcceptanceCriterion",
    "ActorKind",
    "AuthorityBoundary",
    "AuthorityDecisionRecord",
    "AuthorityError",
    "AuthorityFirewall",
    "AuthorityFirewallPolicy",
    "AuthorityLevel",
    "AuthorityRequest",
    "AuthoritySnapshot",
    "CORE_DOCTRINE",
    "ClaimState",
    "CognitionError",
    "CognitionInvariantError",
    "CognitionMemoryError",
    "CognitionPermission",
    "DecisionOutcome",
    "EvidenceError",
    "EvidenceState",
    "FOUNDATIONAL_LAW",
    "FailureMode",
    "FailureRecord",
    "ForbiddenAction",
    "HumanAuthorityRequirement",
    "HumanGoal",
    "LICENSE_NAME",
    "MemoryState",
    "MissionAssumption",
    "MissionAssumptionState",
    "MissionConstraint",
    "MissionConstraintKind",
    "MissionEnvelope",
    "MissionIntakeDefaults",
    "MissionIntakeEngine",
    "MissionIntakeRequest",
    "MissionIntakeResult",
    "MissionReviewTrigger",
    "MissionRisk",
    "PACKAGE_NAME",
    "PROJECT_NAME",
    "PROHIBITED_CLAIMS",
    "PUBLIC_DESCRIPTION",
    "PackageIdentity",
    "PolicyError",
    "RESEARCH_STATUS",
    "ReviewCheckpoint",
    "RiskLevel",
    "RollbackNeed",
    "SelfImprovementError",
    "VERSION",
    "WorkState",
    "evaluate_authority_request",
    "fail_closed",
    "get_package_identity",
    "require_invariant",
    "structure_mission",
]
