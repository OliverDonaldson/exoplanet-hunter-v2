"""What code produced a run, recorded in the artefact that carries its numbers.

A `cv_summary.json` is the only thing a promotion decision reads, and until
2026-08-08 it recorded the training config but not the code. Two runs with
identical `CVConfig` and different architectures were indistinguishable after
the fact — which this project has already paid for once, in a run whose
configuration was not recoverable from its own summary.

**A dirty tree is the case that matters.** Recording a commit SHA while
uncommitted changes are in the working tree states something false: the SHA
names code that is not what ran. That is the house defect — a plausible answer
in place of a loud one — so `dirty` travels beside `sha` and is never inferred
from its absence.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

#: The repository root, resolved from this file rather than the working
#: directory: a script run from elsewhere would otherwise probe whatever
#: repository it happened to be standing in, or none.
_REPO_ROOT = Path(__file__).resolve().parents[4]


def _git(*args: str) -> str | None:
    """A git command's stdout, or None when git or the repository is absent.

    Narrow by design: a missing git binary and a non-repository are the two
    expected absences, and anything else is a real failure that should surface
    rather than be recorded as "no provenance".
    """
    try:
        return (
            subprocess.check_output(
                ["git", "-C", str(_REPO_ROOT), *args], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


@dataclass(frozen=True)
class GitProvenance:
    """The commit a run was trained from, and whether that commit is the truth."""

    sha: str | None
    dirty: bool | None
    branch: str | None

    @property
    def reproducible(self) -> bool:
        """Whether the recorded SHA fully describes the code that ran."""
        return self.sha is not None and self.dirty is False

    def as_dict(self) -> dict[str, str | bool | None]:
        return {"git_sha": self.sha, "git_dirty": self.dirty, "git_branch": self.branch}


def git_provenance() -> GitProvenance:
    """The current commit, dirtiness and branch, each None if unknowable.

    `dirty` is None rather than False when git could not be reached, because
    "no uncommitted changes" and "could not tell" are different claims and only
    one of them licenses a promotion.
    """
    sha = _git("rev-parse", "HEAD")
    status = _git("status", "--porcelain")
    return GitProvenance(
        sha=sha,
        dirty=None if status is None else bool(status),
        branch=_git("rev-parse", "--abbrev-ref", "HEAD"),
    )
