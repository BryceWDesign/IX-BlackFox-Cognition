"""Tests ensuring the repository produces basic CI evidence."""

from pathlib import Path


def test_github_actions_ci_workflow_exists() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = repo_root / ".github" / "workflows" / "ci.yml"

    assert workflow.is_file()


def test_ci_workflow_runs_pytest_without_continue_on_error() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    workflow = repo_root / ".github" / "workflows" / "ci.yml"
    contents = workflow.read_text(encoding="utf-8")

    assert "pull_request:" in contents
    assert "push:" in contents
    assert "python -m pytest" in contents
    assert "continue-on-error" not in contents
