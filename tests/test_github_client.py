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
    # stats default to 0 when include_stats=False
    assert commits[0].additions == 0
    assert commits[0].deletions == 0


@responses_lib.activate
def test_get_commits_with_stats() -> None:
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
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits",
        json=[],
        status=200,
    )
    # Individual commit detail response with stats
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits/abc1234567890",
        json={
            "sha": "abc1234567890",
            "stats": {"additions": 42, "deletions": 7, "total": 49},
        },
        status=200,
    )

    client = GitHubClient()
    commits = client.get_commits("acme", "app", SINCE, UNTIL, include_stats=True)

    assert len(commits) == 1
    assert commits[0].additions == 42
    assert commits[0].deletions == 7


@responses_lib.activate
def test_get_commits_with_stats_gracefully_handles_error() -> None:
    """Stats should default to 0 when the individual commit endpoint fails."""
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
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits",
        json=[],
        status=200,
    )
    # Simulate failure on individual commit endpoint
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/commits/abc1234567890",
        json={"message": "Not Found"},
        status=404,
    )

    client = GitHubClient()
    commits = client.get_commits("acme", "app", SINCE, UNTIL, include_stats=True)

    assert len(commits) == 1
    assert commits[0].additions == 0
    assert commits[0].deletions == 0


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


# ---------------------------------------------------------------------------
# GitHubClient.list_repos
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_list_repos_for_org() -> None:
    payload = [
        {"name": "api-service", "archived": False},
        {"name": "frontend", "archived": False},
        {"name": "old-stuff", "archived": True},  # should be excluded
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/orgs/acme/repos",
        json=payload,
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/orgs/acme/repos",
        json=[],
        status=200,
    )

    client = GitHubClient()
    repos = client.list_repos("acme")

    assert repos == ["api-service", "frontend"]


@responses_lib.activate
def test_list_repos_falls_back_to_user_on_404() -> None:
    # Org endpoint returns 404 → fall back to user endpoint
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/orgs/alice/repos",
        json={"message": "Not Found"},
        status=404,
    )
    payload = [
        {"name": "personal-project", "archived": False},
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/users/alice/repos",
        json=payload,
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/users/alice/repos",
        json=[],
        status=200,
    )

    client = GitHubClient()
    repos = client.list_repos("alice")

    assert repos == ["personal-project"]


@responses_lib.activate
def test_list_repos_excludes_archived() -> None:
    payload = [
        {"name": "active", "archived": False},
        {"name": "deprecated", "archived": True},
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/orgs/acme/repos",
        json=payload,
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/orgs/acme/repos",
        json=[],
        status=200,
    )

    client = GitHubClient()
    repos = client.list_repos("acme")

    assert "deprecated" not in repos
    assert "active" in repos


# ---------------------------------------------------------------------------
# GitHubClient.get_issues – enriched fields
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_get_issues_populates_comments_and_assignees() -> None:
    payload = [
        {
            "number": 5,
            "title": "Performance issue",
            "state": "open",
            "user": {"login": "carol"},
            "created_at": "2024-01-10T09:00:00Z",
            "updated_at": "2024-01-11T09:00:00Z",
            "closed_at": None,
            "html_url": "https://github.com/acme/app/issues/5",
            "labels": [],
            "body": "It's slow",
            "comments": 3,
            "assignees": [{"login": "dave"}, {"login": "eve"}],
        }
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
    assert issues[0].comments == 3
    assert issues[0].assignees == ["dave", "eve"]


# ---------------------------------------------------------------------------
# GitHubClient.get_pull_requests – enriched fields
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_get_pull_requests_populates_new_fields() -> None:
    pr_item = {
        "number": 15,
        "title": "Add dark mode",
        "state": "open",
        "user": {"login": "carol"},
        "created_at": "2024-01-05T00:00:00Z",
        "updated_at": "2024-01-15T00:00:00Z",
        "merged_at": None,
        "html_url": "https://github.com/acme/app/pull/15",
        "labels": [],
        "body": "",
        "draft": True,
        "base": {"ref": "main"},
        "head": {"ref": "feature/dark-mode"},
        "requested_reviewers": [{"login": "alice"}, {"login": "bob"}],
    }
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[pr_item],
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
    assert prs[0].draft is True
    assert prs[0].base_branch == "main"
    assert prs[0].head_branch == "feature/dark-mode"
    assert prs[0].requested_reviewers == ["alice", "bob"]


@responses_lib.activate
def test_get_pull_requests_with_details() -> None:
    pr_item = {
        "number": 15,
        "title": "Refactor auth",
        "state": "closed",
        "user": {"login": "alice"},
        "created_at": "2024-01-05T00:00:00Z",
        "updated_at": "2024-01-10T00:00:00Z",
        "merged_at": "2024-01-10T00:00:00Z",
        "html_url": "https://github.com/acme/app/pull/15",
        "labels": [],
        "body": "",
        "draft": False,
        "base": {"ref": "main"},
        "head": {"ref": "refactor/auth"},
        "requested_reviewers": [],
    }
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[pr_item],
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[],
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls/15",
        json={
            "number": 15,
            "additions": 120,
            "deletions": 30,
            "changed_files": 8,
            "commits": 5,
            "review_comments": 4,
        },
        status=200,
    )
    # Empty review comments list for this test
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls/15/review_comments",
        json=[],
        status=200,
    )

    client = GitHubClient()
    prs = client.get_pull_requests("acme", "app", SINCE, UNTIL, include_details=True)

    assert len(prs) == 1
    assert prs[0].additions == 120
    assert prs[0].deletions == 30
    assert prs[0].changed_files == 8
    assert prs[0].commits_count == 5
    assert prs[0].review_comments == 4


@responses_lib.activate
def test_get_pull_requests_with_details_reviewer_comments() -> None:
    pr_item = {
        "number": 20,
        "title": "Add caching",
        "state": "closed",
        "user": {"login": "dave"},
        "created_at": "2024-01-08T00:00:00Z",
        "updated_at": "2024-01-12T00:00:00Z",
        "merged_at": "2024-01-12T00:00:00Z",
        "html_url": "https://github.com/acme/app/pull/20",
        "labels": [],
        "body": "",
        "draft": False,
        "base": {"ref": "main"},
        "head": {"ref": "feature/caching"},
        "requested_reviewers": [],
    }
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[pr_item],
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[],
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls/20",
        json={"number": 20, "additions": 0, "deletions": 0,
              "changed_files": 1, "commits": 1, "review_comments": 3},
        status=200,
    )
    # Three review comments: alice × 2, bob × 1
    review_comments_payload = [
        {"user": {"login": "alice"}, "body": "LGTM"},
        {"user": {"login": "bob"}, "body": "Nit"},
        {"user": {"login": "alice"}, "body": "Please add test"},
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls/20/review_comments",
        json=review_comments_payload,
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls/20/review_comments",
        json=[],
        status=200,
    )

    client = GitHubClient()
    prs = client.get_pull_requests("acme", "app", SINCE, UNTIL, include_details=True)

    assert len(prs) == 1
    assert prs[0].reviewer_comments == {"alice": 2, "bob": 1}


@responses_lib.activate
def test_get_pull_requests_reviewer_comments_default_empty() -> None:
    """reviewer_comments defaults to empty dict when include_details=False."""
    pr_item = {
        "number": 5,
        "title": "Quick fix",
        "state": "open",
        "user": {"login": "eve"},
        "created_at": "2024-01-05T00:00:00Z",
        "updated_at": "2024-01-06T00:00:00Z",
        "merged_at": None,
        "html_url": "https://github.com/acme/app/pull/5",
        "labels": [],
        "body": "",
        "draft": False,
        "base": {"ref": "main"},
        "head": {"ref": "fix/typo"},
        "requested_reviewers": [],
    }
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/pulls",
        json=[pr_item],
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
    assert prs[0].reviewer_comments == {}


# ---------------------------------------------------------------------------
# GitHubClient.get_releases
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_get_releases_returns_list() -> None:
    payload = [
        {
            "tag_name": "v1.2.0",
            "name": "Version 1.2.0",
            "body": "Bug fixes and improvements",
            "created_at": "2024-01-20T10:00:00Z",
            "published_at": "2024-01-20T12:00:00Z",
            "html_url": "https://github.com/acme/app/releases/tag/v1.2.0",
            "author": {"login": "alice"},
            "prerelease": False,
            "draft": False,
        }
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/releases",
        json=payload,
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/releases",
        json=[],
        status=200,
    )

    client = GitHubClient()
    releases = client.get_releases("acme", "app")

    assert len(releases) == 1
    assert releases[0].tag == "v1.2.0"
    assert releases[0].name == "Version 1.2.0"
    assert releases[0].author == "alice"
    assert releases[0].prerelease is False
    assert releases[0].repo == "acme/app"


@responses_lib.activate
def test_get_releases_filters_by_date() -> None:
    old_release = {
        "tag_name": "v0.9.0",
        "name": "Old release",
        "body": "",
        "created_at": "2023-06-01T10:00:00Z",
        "published_at": "2023-06-01T10:00:00Z",
        "html_url": "https://github.com/acme/app/releases/tag/v0.9.0",
        "author": {"login": "bob"},
        "prerelease": False,
        "draft": False,
    }
    new_release = {
        "tag_name": "v1.2.0",
        "name": "New release",
        "body": "",
        "created_at": "2024-01-15T10:00:00Z",
        "published_at": "2024-01-15T10:00:00Z",
        "html_url": "https://github.com/acme/app/releases/tag/v1.2.0",
        "author": {"login": "alice"},
        "prerelease": False,
        "draft": False,
    }
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/releases",
        json=[new_release, old_release],
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/releases",
        json=[],
        status=200,
    )

    client = GitHubClient()
    releases = client.get_releases("acme", "app", since=SINCE, until=UNTIL)

    assert len(releases) == 1
    assert releases[0].tag == "v1.2.0"


# ---------------------------------------------------------------------------
# GitHubClient.get_contributors
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_get_contributors_returns_list() -> None:
    payload = [
        {"login": "alice", "contributions": 150, "html_url": "https://github.com/alice"},
        {"login": "bob", "contributions": 42, "html_url": "https://github.com/bob"},
    ]
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/contributors",
        json=payload,
        status=200,
    )
    responses_lib.add(
        responses_lib.GET,
        f"{BASE}/repos/acme/app/contributors",
        json=[],
        status=200,
    )

    client = GitHubClient()
    contributors = client.get_contributors("acme", "app")

    assert len(contributors) == 2
    assert contributors[0].login == "alice"
    assert contributors[0].contributions == 150
    assert contributors[0].repo == "acme/app"


# ---------------------------------------------------------------------------
# GitHubClient.create_issue
# ---------------------------------------------------------------------------

@responses_lib.activate
def test_create_issue_posts_and_returns_response() -> None:
    response_payload = {
        "number": 99,
        "title": "Implement feature X",
        "html_url": "https://github.com/acme/app/issues/99",
        "state": "open",
    }
    responses_lib.add(
        responses_lib.POST,
        f"{BASE}/repos/acme/app/issues",
        json=response_payload,
        status=201,
    )

    client = GitHubClient(token="ghp_fake")
    result = client.create_issue(
        "acme",
        "app",
        title="Implement feature X",
        body="This is the body",
        labels=["enhancement"],
        assignees=["alice"],
    )

    assert result["number"] == 99
    assert result["html_url"] == "https://github.com/acme/app/issues/99"
    # Verify the request body
    request_body = responses_lib.calls[0].request.body
    import json
    body = json.loads(request_body)
    assert body["title"] == "Implement feature X"
    assert body["labels"] == ["enhancement"]
    assert body["assignees"] == ["alice"]
