"""Tests for auto-committing a Generation Run's Export (issue #6)."""

import subprocess
from pathlib import Path

import pytest

from export_vcs import commit_export, should_commit


def git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args], check=True, capture_output=True, text=True
    ).stdout


@pytest.fixture
def repo(tmp_path):
    """A git repo with one commit, an unstaged edit and a staged unrelated change."""
    root = tmp_path / "repo"
    root.mkdir()
    git(root, "init", "-q")
    git(root, "config", "user.name", "Test")
    git(root, "config", "user.email", "test@example.com")
    git(root, "config", "commit.gpgsign", "false")
    (root / "generator.py").write_text("v1\n")
    (root / "README.md").write_text("v1\n")
    git(root, "add", ".")
    git(root, "commit", "-q", "-m", "initial")
    (root / "generator.py").write_text("work in progress\n")
    (root / "README.md").write_text("staged change\n")
    git(root, "add", "README.md")
    return root


def write_export(repo: Path, name: str = "SAA-C03_10q_20261118T120000.json") -> Path:
    export = repo / "output" / name
    export.parent.mkdir(exist_ok=True)
    export.write_text('{"questions": []}\n')
    return export


def test_commits_new_export_in_output_dir_when_enabled(tmp_path):
    output_dir = tmp_path / "output"

    decision = should_commit(
        enabled=True,
        export_path=output_dir / "SAA-C03_10q_20261118T120000.json",
        approved_count=10,
        output_dir=output_dir,
    )

    assert decision.commit is True


@pytest.mark.parametrize(
    "case, enabled, export_name, approved_count, reason_mentions",
    [
        ("auto-commit off",       False, "output/SAA-C03_10q_20261118T120000.json", 10, "off"),
        ("no Export written",     True,  None,                                       10, "no export"),
        ("zero approved",         True,  "output/SAA-C03_0q_20261118T120000.json",   0,  "no approved"),
        ("Export outside output", True,  "elsewhere/SAA-C03_10q_20261118T120000.json", 10, "outside"),
    ],
)
def test_skips_commit(tmp_path, case, enabled, export_name, approved_count, reason_mentions):
    decision = should_commit(
        enabled=enabled,
        export_path=tmp_path / export_name if export_name else None,
        approved_count=approved_count,
        output_dir=tmp_path / "output",
    )

    assert decision.commit is False, case
    assert reason_mentions in decision.reason.lower(), case


def test_commits_only_the_export_and_leaves_work_in_progress_alone(repo):
    export = write_export(repo)

    outcome = commit_export(export, exam_code="SAA-C03", count=10)

    assert outcome.committed is True
    assert git(repo, "log", "-1", "--format=%s").strip() == (
        "Add 10 SAA-C03 questions (20261118T120000)"
    )
    assert git(repo, "show", "--name-only", "--format=", "HEAD").split() == [
        "output/SAA-C03_10q_20261118T120000.json"
    ]
    assert outcome.sha == git(repo, "rev-parse", "HEAD").strip()
    status = git(repo, "status", "--porcelain").splitlines()
    assert sorted(status) == sorted(["M  README.md", " M generator.py"])


def test_outside_a_git_repo_reports_why_and_keeps_the_export(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CEILING_DIRECTORIES", str(tmp_path))
    plain_dir = tmp_path / "not-a-repo"
    plain_dir.mkdir()
    export = write_export(plain_dir)

    outcome = commit_export(export, exam_code="SAA-C03", count=10)

    assert outcome.committed is False
    assert "not a git repository" in outcome.reason.lower()
    assert export.is_file()


def test_rejected_commit_reports_why_and_leaves_the_index_as_it_was(repo):
    hook = repo / ".git" / "hooks" / "pre-commit"
    hook.write_text("#!/bin/sh\necho 'questions need review' >&2\nexit 1\n", newline="\n")
    hook.chmod(0o755)
    export = write_export(repo)
    head_before = git(repo, "rev-parse", "HEAD")

    outcome = commit_export(export, exam_code="SAA-C03", count=10)

    assert outcome.committed is False
    assert "questions need review" in outcome.reason
    assert git(repo, "rev-parse", "HEAD") == head_before
    status = git(repo, "status", "--porcelain").splitlines()
    assert sorted(status) == sorted(["M  README.md", " M generator.py", "?? output/"])
    assert export.is_file()


def test_missing_git_reports_why_and_keeps_the_export(tmp_path, monkeypatch):
    export = write_export(tmp_path)
    monkeypatch.setenv("PATH", "")

    outcome = commit_export(export, exam_code="SAA-C03", count=10)

    assert outcome.committed is False
    assert "git is not installed" in outcome.reason.lower()
    assert export.is_file()
