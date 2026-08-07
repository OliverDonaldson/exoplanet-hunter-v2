"""Tests for `utils.provenance` — what code produced a run.

The distinction these pin is between "the tree was clean" and "we could not
tell". Collapsing the second into the first is what would let a run trained
from uncommitted changes record a commit that does not describe it.
"""

from __future__ import annotations

from exoplanet_hunter.utils.provenance import GitProvenance, git_provenance


def test_a_clean_commit_is_reproducible():
    assert GitProvenance(sha="abc123", dirty=False, branch="v2").reproducible


def test_a_dirty_tree_is_not_reproducible():
    assert not GitProvenance(sha="abc123", dirty=True, branch="v2").reproducible


def test_unknown_dirtiness_is_not_reproducible():
    """None is not False. "No uncommitted changes" and "could not tell" are
    different claims and only one of them licenses a promotion."""
    assert not GitProvenance(sha="abc123", dirty=None, branch="v2").reproducible


def test_a_missing_sha_is_not_reproducible():
    assert not GitProvenance(sha=None, dirty=False, branch=None).reproducible


def test_as_dict_carries_dirtiness_beside_the_sha():
    keys = set(GitProvenance(sha="abc", dirty=False, branch="v2").as_dict())
    assert keys == {"git_sha", "git_dirty", "git_branch"}


def test_this_repository_resolves():
    """Resolved from the package location, not the working directory, so a
    script run from elsewhere does not probe whatever repository it stands in."""
    provenance = git_provenance()
    assert provenance.sha is not None
    assert len(provenance.sha) == 40
    assert provenance.dirty is not None
