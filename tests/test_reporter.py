"""Tests for git_review.reporter.ReviewReporter."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest

from git_review.models import (
    AuthorSummary,
    Commit,
    Contributor,
    Issue,
    PullRequest,
    Release,
    ReviewSummary,
)
from git_review.reporter import ReviewReporter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_SINCE = datetime(2024, 1, 1, tzinfo=timezone.utc)
_UNTIL = datetime(2024, 1, 31, tzinfo=timezone.utc)


def _make_summary(*, repo: str = "app", all_repos: bool = False) -> ReviewSummary:
    return ReviewSummary(
        owner="acme",
        repo="*" if all_repos else repo,
        since=_SINCE,
        until=_UNTIL,
        commits=[
            Commit(
                sha="abc1234567890",
                message="feat: add feature",
                author="alice",
                authored_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
                url="https://example.com/c/1",
                repo="acme/app",
                additions=10,
                deletions=3,
            )
        ],
        issues=[
            Issue(
                number=1,
                title="Bug in module",
                state="closed",
                author="bob",
                created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
                closed_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
                url="https://example.com/i/1",
                repo="acme/app",
                labels=["bug"],
            ),
            Issue(
                number=2,
                title="Enhancement request",
                state="open",
                author="carol",
                created_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
                closed_at=None,
                url="https://example.com/i/2",
                repo="acme/app",
            ),
        ],
        pull_requests=[
            PullRequest(
                number=5,
                title="Fix the bug",
                state="closed",
                author="alice",
                created_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
                merged_at=datetime(2024, 1, 6, tzinfo=timezone.utc),
                url="https://example.com/pr/5",
                repo="acme/app",
                reviewer_comments={"dave": 2},
            )
        ],
        releases=[
            Release(
                tag="v1.0.0",
                name="v1.0.0",
                body="First release",
                created_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                published_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                url="https://example.com/r/1",
                repo="acme/app",
                author="alice",
            )
        ],
        contributors=[
            Contributor(
                login="alice",
                contributions=42,
                url="https://github.com/alice",
                repo="acme/app",
            )
        ],
    )


# ---------------------------------------------------------------------------
# to_markdown – structure checks
# ---------------------------------------------------------------------------


def test_to_markdown_contains_header() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "# git-review: acme/app" in md
    assert "2024-01-01 → 2024-01-31" in md


def test_to_markdown_commits_section() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## Commits (1)" in md
    assert "abc1234" in md
    assert "feat: add feature" in md
    assert "+10" in md
    assert "-3" in md


def test_to_markdown_repo_stats_section() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## Repo Stats" in md
    assert "acme/app" in md


def test_to_markdown_issues_section() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## Issues (2)" in md
    assert "Bug in module" in md
    assert "Enhancement request" in md
    assert "bug" in md


def test_to_markdown_issue_age_section() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## Issue Age" in md
    assert "### Open Issue Age" in md
    assert "### Closed Issue Age" in md
    assert "0–7 days" in md


def test_to_markdown_prs_section() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## Pull Requests (1)" in md
    assert "Fix the bug" in md
    assert "2024-01-06" in md
    assert "dave(2)" in md


def test_to_markdown_releases_section() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## Releases (1)" in md
    assert "v1.0.0" in md
    assert "2024-01-15" in md


def test_to_markdown_contributors_section() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## Contributors (1)" in md
    assert "alice" in md
    assert "42" in md


def test_to_markdown_all_repos_mode_adds_repo_column() -> None:
    md = ReviewReporter.to_markdown(_make_summary(all_repos=True))
    assert "# git-review: acme/*" in md
    # Repo column header should appear in the Commits table
    lines = [l for l in md.splitlines() if "| SHA" in l or "| Repo" in l]
    assert any("Repo" in line for line in lines)


# ---------------------------------------------------------------------------
# to_markdown – empty data
# ---------------------------------------------------------------------------


def test_to_markdown_no_commits() -> None:
    summary = ReviewSummary(owner="acme", repo="empty", since=_SINCE, until=_UNTIL)
    md = ReviewReporter.to_markdown(summary)
    assert "_No commits found" in md


def test_to_markdown_no_issues() -> None:
    summary = ReviewSummary(owner="acme", repo="empty", since=_SINCE, until=_UNTIL)
    md = ReviewReporter.to_markdown(summary)
    assert "_No issues found" in md


def test_to_markdown_no_prs() -> None:
    summary = ReviewSummary(owner="acme", repo="empty", since=_SINCE, until=_UNTIL)
    md = ReviewReporter.to_markdown(summary)
    assert "_No pull requests found" in md


def test_to_markdown_no_releases_omits_section() -> None:
    summary = ReviewSummary(owner="acme", repo="empty", since=_SINCE, until=_UNTIL)
    md = ReviewReporter.to_markdown(summary)
    assert "## Releases" not in md


def test_to_markdown_no_contributors_omits_section() -> None:
    summary = ReviewSummary(owner="acme", repo="empty", since=_SINCE, until=_UNTIL)
    md = ReviewReporter.to_markdown(summary)
    assert "## Contributors" not in md


# ---------------------------------------------------------------------------
# to_markdown – GFM table format
# ---------------------------------------------------------------------------


def test_to_markdown_gfm_table_pipes() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    # Every table row starts and ends with |
    table_rows = [line for line in md.splitlines() if line.startswith("|")]
    assert table_rows, "Expected at least one GFM table row"
    for row in table_rows:
        assert row.endswith("|"), f"Row does not end with pipe: {row!r}"


def test_to_markdown_has_separator_rows() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    separator_rows = [
        line for line in md.splitlines()
        if line.startswith("|") and "---" in line
    ]
    assert separator_rows, "Expected GFM separator rows (|---|---|)"


# ---------------------------------------------------------------------------
# fetch – delegates to GitHubClient
# ---------------------------------------------------------------------------


def _mock_gh() -> MagicMock:
    gh = MagicMock(spec=["get_commits", "get_issues", "get_pull_requests",
                          "get_releases", "get_contributors", "list_repos"])
    gh.get_commits.return_value = []
    gh.get_issues.return_value = []
    gh.get_pull_requests.return_value = []
    gh.get_releases.return_value = []
    gh.get_contributors.return_value = []
    gh.list_repos.return_value = ["repo-a", "repo-b"]
    return gh


def test_fetch_single_repo_calls_client_methods() -> None:
    gh = _mock_gh()
    reporter = ReviewReporter(gh)
    summary = reporter.fetch("acme", _SINCE, _UNTIL, repo="my-repo")

    gh.get_commits.assert_called_once_with(
        "acme", "my-repo", _SINCE, _UNTIL, author=None, include_stats=True, branch="*"
    )
    gh.get_issues.assert_called_once_with("acme", "my-repo", _SINCE, _UNTIL)
    gh.get_pull_requests.assert_called_once_with(
        "acme", "my-repo", _SINCE, _UNTIL, include_details=True
    )
    gh.get_releases.assert_called_once_with("acme", "my-repo", _SINCE, _UNTIL)
    gh.get_contributors.assert_called_once_with("acme", "my-repo")

    assert summary.owner == "acme"
    assert summary.repo == "my-repo"
    assert summary.since == _SINCE
    assert summary.until == _UNTIL


def test_fetch_owner_lists_all_repos() -> None:
    gh = _mock_gh()
    reporter = ReviewReporter(gh)
    summary = reporter.fetch("acme", _SINCE, _UNTIL)

    gh.list_repos.assert_called_once_with("acme")
    assert gh.get_commits.call_count == 2  # once per repo
    assert summary.repo == "*"


def test_fetch_author_forwarded_to_get_commits() -> None:
    gh = _mock_gh()
    reporter = ReviewReporter(gh)
    reporter.fetch("acme", _SINCE, _UNTIL, repo="my-repo", author="alice")

    gh.get_commits.assert_called_once_with(
        "acme", "my-repo", _SINCE, _UNTIL, author="alice", include_stats=True, branch="*"
    )


def test_fetch_branch_forwarded_to_get_commits() -> None:
    gh = _mock_gh()
    reporter = ReviewReporter(gh)
    reporter.fetch("acme", _SINCE, _UNTIL, repo="my-repo", branch="feature-x")

    gh.get_commits.assert_called_once_with(
        "acme", "my-repo", _SINCE, _UNTIL, author=None, include_stats=True, branch="feature-x"
    )


def test_fetch_default_branch_is_all_branches() -> None:
    """SDK fetch() defaults to branch='*' so all branches are included."""
    gh = _mock_gh()
    reporter = ReviewReporter(gh)
    reporter.fetch("acme", _SINCE, _UNTIL, repo="my-repo")

    _, kwargs = gh.get_commits.call_args
    assert kwargs.get("branch") == "*"


def test_fetch_tolerates_partial_errors() -> None:
    gh = _mock_gh()
    gh.get_commits.side_effect = RuntimeError("network error")
    reporter = ReviewReporter(gh)
    # Should not raise; failed sections are silently skipped
    summary = reporter.fetch("acme", _SINCE, _UNTIL, repo="my-repo")
    assert summary.commits == []
    # Other sections still attempted
    gh.get_issues.assert_called_once()


# ---------------------------------------------------------------------------
# partition_by_author
# ---------------------------------------------------------------------------


def test_partition_by_author_groups_by_author_field() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)

    assert "alice" in partitioned
    assert "bob" in partitioned
    assert "carol" in partitioned

    assert len(partitioned["alice"].commits) == 1
    assert len(partitioned["bob"].issues) == 1
    assert len(partitioned["carol"].issues) == 1
    assert len(partitioned["alice"].pull_requests) == 1
    assert len(partitioned["alice"].releases) == 1


def test_partition_by_author_is_sorted_alphabetically() -> None:
    summary = _make_summary()
    keys = list(ReviewReporter.partition_by_author(summary).keys())
    assert keys == sorted(keys)


def test_partition_by_author_releases_without_author_excluded() -> None:
    summary = ReviewSummary(
        owner="acme",
        repo="app",
        since=_SINCE,
        until=_UNTIL,
        releases=[
            Release(
                tag="v0.1",
                name="v0.1",
                body="",
                created_at=_SINCE,
                published_at=_SINCE,
                url="https://example.com",
                repo="acme/app",
                author="",  # no author
            )
        ],
    )
    partitioned = ReviewReporter.partition_by_author(summary)
    # The empty-author release must not create a spurious "" key
    assert "" not in partitioned


def test_partition_by_author_empty_summary() -> None:
    summary = ReviewSummary(owner="acme", repo="app", since=_SINCE, until=_UNTIL)
    assert ReviewReporter.partition_by_author(summary) == {}


def test_partition_by_author_commit_not_duplicated_in_other_authors() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    # alice's commit should only be in alice's bucket
    for author, bucket in partitioned.items():
        if author != "alice":
            assert all(c.sha != "abc1234567890" for c in bucket.commits)


# ---------------------------------------------------------------------------
# author_summaries_to_markdown
# ---------------------------------------------------------------------------


def test_author_summaries_to_markdown_has_by_author_header() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    md = ReviewReporter.author_summaries_to_markdown(partitioned)
    assert "## By Author" in md


def test_author_summaries_to_markdown_has_per_author_sections() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    md = ReviewReporter.author_summaries_to_markdown(partitioned)

    assert "### alice" in md
    assert "### bob" in md
    assert "### carol" in md


def test_author_summaries_to_markdown_shows_commit_table_for_author() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    md = ReviewReporter.author_summaries_to_markdown(partitioned)
    # alice has commits
    assert "#### Commits (1)" in md
    assert "feat: add feature" in md


def test_author_summaries_to_markdown_shows_issue_table() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    md = ReviewReporter.author_summaries_to_markdown(partitioned)
    assert "#### Issues" in md
    assert "Bug in module" in md


def test_author_summaries_to_markdown_shows_pr_table() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    md = ReviewReporter.author_summaries_to_markdown(partitioned)
    assert "#### Pull Requests" in md
    assert "Fix the bug" in md


def test_author_summaries_to_markdown_shows_release_table() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    md = ReviewReporter.author_summaries_to_markdown(partitioned)
    assert "#### Releases" in md
    assert "v1.0.0" in md


def test_author_summaries_to_markdown_headline_stats() -> None:
    summary = _make_summary()
    partitioned = ReviewReporter.partition_by_author(summary)
    md = ReviewReporter.author_summaries_to_markdown(partitioned)
    # alice has 1 commit, 1 merged PR, 1 release
    assert "**Commits:** 1" in md
    assert "**Pull Requests:**" in md
    assert "**Releases:** 1" in md


def test_author_summaries_to_markdown_empty_returns_empty_string() -> None:
    assert ReviewReporter.author_summaries_to_markdown({}) == ""


# ---------------------------------------------------------------------------
# to_markdown – include_author_summaries flag
# ---------------------------------------------------------------------------


def test_to_markdown_includes_author_summaries_by_default() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    assert "## By Author" in md
    assert "### alice" in md


def test_to_markdown_exclude_author_summaries() -> None:
    md = ReviewReporter.to_markdown(_make_summary(), include_author_summaries=False)
    assert "## By Author" not in md
    assert "### alice" not in md


def test_to_markdown_author_section_at_end() -> None:
    md = ReviewReporter.to_markdown(_make_summary())
    by_author_idx = md.index("## By Author")
    contributors_idx = md.index("## Contributors")
    assert by_author_idx > contributors_idx
