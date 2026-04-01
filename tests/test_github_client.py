"""Tests for GitHubClient."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
import responses as responses_lib

from git_review.github_client import GitHubClient, _to_iso

SINCE = datetime(2024, 1, 1, tzinfo=timezone.utc)
UNTIL = datetime(2024, 1, 31, 23, 59, 59, tzinfo=timezone.utc)
BASE = "https://api.github.com"


# ---------------------------------------------------------------------------
# _to_iso helper
# ---------------------------------------------------------------------------

def test_to_iso_utc() -> None:
    dt = datetime(2024, 3, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert _to_iso(dt) == "2024-03-15T12:00:00Z"


def test_to_iso_naive_treated_as_utc() -> None:
    dt = datetime(2024, 3, 15, 12, 0, 0)  # naive
    assert _to_iso(dt) == "2024-03-15T12:00:00Z"


# ---------------------------------------------------------------------------
# GitHubClient.get_commits
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_get_commits_returns_list() -> None:
    payload = [
        {
            "sha": "abc1234567890",
            "commit": {
                "message": "feat: add login page",
                "author": {"name": "Alice", "date": "2024-01-10T08:00:00Z"},
            },
            "html_url": "https://github.com/acme/app/commit/abc1234567890",
            "author": {"login": "alice"},
        }
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits",
        json=payload,
        status=200,
    )
    # Second call returns empty list (end of pagination)
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits",
        json=[],
        status=200,
    )

    client = GitHubClient()
    commits = client.get_commits("acme", "app", SINCE, UNTIL)

    assert len(commits) == 1
    assert commits[0].sha == "abc1234567890"
    assert commits[0].message == "feat: add login page"
    assert commits[0].author == "Alice"
    assert commits[0].repo == "acme/app"


@responses_lib.activate
def test_get_commits_author_filter() -> None:
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits",
        json=[],
        status=200,
    )

    client = GitHubClient(token="ghp_fake")
    commits = client.get_commits("acme", "app", SINCE, UNTIL, author="bob")

    assert commits == []
    # Ensure author param was forwarded
    assert "author=bob" in responses_lib.calls[0].request.url


@responses_lib.activate
def test_get_commits_http_error_raises() -> None:
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits",
        json={"message": "Not Found"},
        status=404,
    )

    client = GitHubClient()
    with pytest.raises(Exception):
        client.get_commits("acme", "app", SINCE, UNTIL)


# ---------------------------------------------------------------------------
# GitHubClient.get_issues
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_get_issues_excludes_prs() -> None:
    payload = [
        {
            "number": 1,
            "title": "Bug in login",
            "state": "open",
            "user": {"login": "alice"},
            "created_at": "2024-01-05T10:00:00Z",
            "updated_at": "2024-01-06T10:00:00Z",
            "closed_at": None,
            "html_url": "https://github.com/acme/app/issues/1",
            "labels": [{"name": "bug"}],
            "body": "Something is wrong",
        },
        {
            "number": 2,
            "title": "PR mixed in",
            "state": "open",
            "user": {"login": "bob"},
            "created_at": "2024-01-07T10:00:00Z",
            "updated_at": "2024-01-07T10:00:00Z",
            "closed_at": None,
            "html_url": "https://github.com/acme/app/pull/2",
            "labels": [],
            "body": "",
            "pull_request": {"url": "https://api.github.com/repos/acme/app/pulls/2"},
        },
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/issues",
        json=payload,
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/issues",
        json=[],
        status=200,
    )

    client = GitHubClient()
    issues = client.get_issues("acme", "app", SINCE, UNTIL)

    assert len(issues) == 1
    assert issues[0].number == 1
    assert issues[0].labels == ["bug"]


# ---------------------------------------------------------------------------
# GitHubClient.get_pull_requests
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_get_pull_requests_filters_by_date() -> None:
    # This PR is within range
    pr_in = {
        "number": 10,
        "title": "Add dark mode",
        "state": "closed",
        "user": {"login": "carol"},
        "created_at": "2024-01-05T00:00:00Z",
        "updated_at": "2024-01-15T00:00:00Z",
        "merged_at": "2024-01-15T00:00:00Z",
        "html_url": "https://github.com/acme/app/pull/10",
        "labels": [],
        "body": "",
    }
    # This PR is outside range (updated before SINCE)
    pr_out = {
        "number": 9,
        "title": "Old PR",
        "state": "closed",
        "user": {"login": "dave"},
        "created_at": "2023-12-01T00:00:00Z",
        "updated_at": "2023-12-20T00:00:00Z",
        "merged_at": None,
        "html_url": "https://github.com/acme/app/pull/9",
        "labels": [],
        "body": "",
    }
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[pr_in, pr_out],
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[],
        status=200,
    )

    client = GitHubClient()
    prs = client.get_pull_requests("acme", "app", SINCE, UNTIL)

    assert len(prs) == 1
    assert prs[0].number == 10
    assert prs[0].merged_at is not None
