from __future__ import annotations

from datetime import datetime, timezone

from rich.table import Table

from git_review.models import Commit, Issue, PullRequest, ReviewSummary
from git_review.tables import build_review_renderables


def _summary_with_data(repo: str = "app") -> ReviewSummary:
    return ReviewSummary(
        owner="acme",
        repo=repo,
        since=datetime(2024, 1, 1, tzinfo=timezone.utc),
        until=datetime(2024, 1, 31, tzinfo=timezone.utc),
        commits=[
            Commit(
                sha="abc1234567890",
                message="feat: test",
                author="alice",
                authored_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
                url="https://example.com",
                repo="acme/app",
                additions=5,
                deletions=2,
            )
        ],
        issues=[
            Issue(
                number=1,
                title="Issue",
                state="closed",
                author="alice",
                created_at=datetime(2024, 1, 2, tzinfo=timezone.utc),
                closed_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
                url="https://example.com",
                repo="acme/app",
            )
        ],
        pull_requests=[
            PullRequest(
                number=7,
                title="PR",
                state="closed",
                author="alice",
                created_at=datetime(2024, 1, 3, tzinfo=timezone.utc),
                merged_at=datetime(2024, 1, 6, tzinfo=timezone.utc),
                url="https://example.com",
                repo="acme/app",
            )
        ],
    )


def test_build_review_renderables_contains_expected_tables() -> None:
    renderables = build_review_renderables(_summary_with_data())
    titles = [r.title for r in renderables if isinstance(r, Table)]
    assert "Commits (1)" in titles
    assert "Repo Stats" in titles
    assert "Issues (1)" in titles
    assert "Pull Requests (1)" in titles
    assert "Closed Issue Age (Days Open)" in titles


def test_build_review_renderables_in_all_repos_mode_adds_repo_columns() -> None:
    renderables = build_review_renderables(_summary_with_data(repo="*"))
    commits_table = next(
        r for r in renderables if isinstance(r, Table) and r.title == "Commits (1)"
    )
    headers = [col.header for col in commits_table.columns]
    assert "Repo" in headers
