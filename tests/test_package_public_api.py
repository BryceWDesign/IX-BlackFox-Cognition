"""Tests for the stable IX-BlackFox-Cognition public API shell."""

from ix_blackfox_cognition import (
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


def test_package_identity_exports_stable_project_metadata() -> None:
    identity = get_package_identity()

    assert isinstance(identity, PackageIdentity)
    assert identity.project_name == PROJECT_NAME
    assert identity.package_name == PACKAGE_NAME
    assert identity.version == VERSION
    assert identity.license_name == LICENSE_NAME
    assert identity.research_status == RESEARCH_STATUS
    assert identity.doctrine == CORE_DOCTRINE
    assert identity.foundational_law == FOUNDATIONAL_LAW
    assert identity.public_description == PUBLIC_DESCRIPTION
    assert identity.prohibited_claims == PROHIBITED_CLAIMS


def test_public_identity_preserves_core_doctrine() -> None:
    identity = get_package_identity()

    assert "Model thinks" in identity.doctrine
    assert "Cognition structures" in identity.doctrine
    assert "BlackFox governs" in identity.doctrine
    assert "humans authorize" in identity.doctrine
    assert "evidence decides trust" in identity.doctrine


def test_public_identity_does_not_claim_agi_or_production_readiness() -> None:
    identity = get_package_identity()

    assert "AGI" in identity.prohibited_claims
    assert "autonomous AGI" in identity.prohibited_claims
    assert "production-ready" in identity.prohibited_claims
    assert "certified" in identity.prohibited_claims
    assert "governed cognition substrate" in identity.public_description
