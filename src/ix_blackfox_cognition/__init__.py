"""IX-BlackFox-Cognition public API.

IX-BlackFox-Cognition is a source-available governed cognition substrate for
evidence-bound AI-assisted engineering workflows.

The current public API exports stable project metadata, dependency-light core
types, and authority-kernel data models. Higher-level firewall, planning,
memory, routing, sentinel, and handoff APIs will be exported only after their
modules exist and are covered by tests.
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

__version__ = VERSION

__all__ = [
    "ActorKind",
    "AuthorityBoundary",
    "AuthorityDecisionRecord",
    "AuthorityError",
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
    "LICENSE_NAME",
    "MemoryState",
    "PACKAGE_NAME",
    "PROJECT_NAME",
    "PROHIBITED_CLAIMS",
    "PUBLIC_DESCRIPTION",
    "PackageIdentity",
    "PolicyError",
    "RESEARCH_STATUS",
    "RiskLevel",
    "SelfImprovementError",
    "VERSION",
    "WorkState",
    "fail_closed",
    "get_package_identity",
    "require_invariant",
]
