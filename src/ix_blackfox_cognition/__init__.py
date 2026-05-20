"""IX-BlackFox-Cognition public API.

IX-BlackFox-Cognition is a source-available governed cognition substrate for
evidence-bound AI-assisted engineering workflows.

The current public API exports only stable package identity metadata. Governance,
authority, planning, memory, routing, sentinel, and handoff APIs will be exported
only after their modules exist and are covered by tests.
"""

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
    "CORE_DOCTRINE",
    "FOUNDATIONAL_LAW",
    "LICENSE_NAME",
    "PACKAGE_NAME",
    "PROJECT_NAME",
    "PROHIBITED_CLAIMS",
    "PUBLIC_DESCRIPTION",
    "RESEARCH_STATUS",
    "VERSION",
    "PackageIdentity",
    "get_package_identity",
]
