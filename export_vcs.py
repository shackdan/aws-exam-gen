"""
export_vcs.py
─────────────
Optional auto-commit of a Generation Run's Export to git (issue #6).

The Exports in output/ are the only copy of each Exam's Question Bank,
so when auto-commit is on, the new Export is committed straight after
the run.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class CommitOutcome:
    committed: bool
    sha:       Optional[str] = None
    reason:    str = ""


@dataclass(frozen=True)
class CommitDecision:
    commit: bool
    reason: str = ""


def should_commit(
    enabled: bool,
    export_path: Optional[Path],
    approved_count: int,
    output_dir: Path,
) -> CommitDecision:
    """
    Decide whether this Generation Run's Export should be committed.

    Only an Export of Approved Questions that lives in output_dir (the
    Question Bank) is committed. Everything else is skipped with a
    reason the CLI can show.
    """
    if not enabled:
        return CommitDecision(False, "Auto-commit is off.")
    if export_path is None:
        return CommitDecision(False, "No Export was written, so there is nothing to commit.")
    if approved_count == 0:
        return CommitDecision(False, "No Approved Questions, so the Export is not committed.")
    if export_path.resolve().parent != output_dir.resolve():
        return CommitDecision(
            False,
            f"The Export is outside {output_dir}, so it is not part of the "
            "Question Bank and is not committed.",
        )
    return CommitDecision(True)


def _git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, capture_output=True, text=True
    )


def commit_export(export_path: Path, exam_code: str, count: int) -> CommitOutcome:
    """
    Commit just this Export, leaving any other staged or unstaged work as it was.

    Never raises: any git problem becomes a CommitOutcome with a reason, and
    the Export stays on disk.
    """
    timestamp = export_path.stem.rsplit("_", 1)[-1]
    message   = f"Add {count} {exam_code} questions ({timestamp})"
    cwd       = export_path.parent

    try:
        in_repo = _git(cwd, "rev-parse", "--is-inside-work-tree").returncode == 0
    except FileNotFoundError:
        return CommitOutcome(
            committed=False,
            reason="git is not installed, so the Export was not committed.",
        )
    if not in_repo:
        return CommitOutcome(
            committed=False,
            reason=f"{cwd} is not a git repository, so the Export was not committed.",
        )

    _git(cwd, "add", "--", export_path.name)
    # A pathspec commit (--only) records only this file; anything else
    # already staged stays staged and out of the commit.
    commit = _git(cwd, "commit", "-q", "-m", message, "--only", "--", export_path.name)
    if commit.returncode != 0:
        # Un-stage the Export so the index is exactly as the user left it.
        _git(cwd, "rm", "-q", "--cached", "--", export_path.name)
        detail = (commit.stderr or commit.stdout).strip()
        return CommitOutcome(
            committed=False,
            reason=f"git refused the commit, so the Export was not committed: {detail}",
        )
    sha = _git(cwd, "rev-parse", "HEAD").stdout.strip()
    return CommitOutcome(committed=True, sha=sha)
