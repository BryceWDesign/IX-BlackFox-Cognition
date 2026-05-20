"""Project identity metadata for IX-BlackFox-Cognition.

This module intentionally contains only stable package identity values. It does
not import future governance, planning, memory, routing, sentinel, or handoff
modules, so the public API can remain safe while the package evolves.
"""

from __future__ import annotations

from dataclasses import dataclass


PROJECT_NAME = "IX-BlackFox-Cognition"
PACKAGE_NAME = "ix_blackfox_cognition"
VERSION = "0.1.0"
LICENSE_NAME = "IX-BlackFox-Cognition Source-Available Evaluation License v1.0"

CORE_DOCTRINE = (
    "Model thinks → Cognition structures → BlackFox governs → humans authorize "
    "→ evidence decides trust."
)

FOUNDATIONAL_LAW = (
    "Do not grant trust to intelligence. Grant trust only to verified, bounded, "
    "reviewable action."
)

PUBLIC_DESCRIPTION = (
    "IX-BlackFox-Cognition is a source-available governed cognition substrate "
    "for AI-assisted engineering agents, converting human intent into inspectable "
    "mission envelopes, evidence-bound belief and plan graphs, quarantined memory "
    "updates, role-separated model reviews, self-improvement proposals, and "
    "BlackFox-compatible action candidates under human authority."
)

RESEARCH_STATUS = "research-prototype"

PROHIBITED_CLAIMS = (
    "AGI",
    "autonomous AGI",
    "self-aware",
    "production-ready",
    "certified",
    "government-affiliated",
    "defense-affiliated",
    "autonomous authority",
)


@dataclass(frozen=True, slots=True)
class PackageIdentity:
    """Stable public identity information for the package."""

    project_name: str
    package_name: str
    version: str
    license_name: str
    research_status: str
    doctrine: str
    foundational_law: str
    public_description: str
    prohibited_claims: tuple[str, ...]


def get_package_identity() -> PackageIdentity:
    """Return stable package identity metadata.

    The returned object is intentionally immutable so callers cannot accidentally
    mutate project identity values at runtime.
    """

    return PackageIdentity(
        project_name=PROJECT_NAME,
        package_name=PACKAGE_NAME,
        version=VERSION,
        license_name=LICENSE_NAME,
        research_status=RESEARCH_STATUS,
        doctrine=CORE_DOCTRINE,
        foundational_law=FOUNDATIONAL_LAW,
        public_description=PUBLIC_DESCRIPTION,
        prohibited_claims=PROHIBITED_CLAIMS,
    )
