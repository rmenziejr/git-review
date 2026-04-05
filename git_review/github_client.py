"""GitHub REST API client for git-review."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from dateutil.parser import isoparse

from .models import Commit, Contributor, Issue, Milestone, PullRequest, Release

logger = logging.getLogger(__name__)

_GITHUB_API = "https://api.github.com/"


class GitHubClient:
    """Thin wrapper around the GitHub REST API v3.

    Parameters
    ----------
    token:
        A GitHub personal access token (PAT) or fine-grained token with at
        least ``repo`` read scope.  When *None* the client operates in
        unauthenticated mode which is subject to strict rate limits.
    base_url:
        Override the API base URL, e.g. for GitHub Enterprise deployments.
    """

    def __init__(
        self,
        token: Optional[str] = None,
        base_url: str = _GITHUB_API,
    ) -> None:
        self._base_url = base_url.rstrip("/") + "/"
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
            }
        )
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get(self, path: str, **params: Any) -> Any:
        """GET a single page and return the parsed JSON."""
        url = urljoin(self._base_url, path.lstrip("/"))
        response = self._session.get(url, params=params)
        response.raise_for_status()
        return response.json()

    def _paginate(self, path: str, **params: Any) -> list[dict]:
        """GET all pages for a list endpoint and return merged results."""
        results: list[dict] = []
        params.setdefault("per_page", 100)
        page = 1
        while True:
            params["page"] = page
            data = self._get(path, **params)
            if not data:
                break
            results.extend(data)
            if len(data) < params["per_page"]:
                break
            page += 1
        return results

    def _post(self, path: str, json: Any) -> Any:
        """POST *json* to *path* and return the parsed response JSON."""
        url = urljoin(self._base_url, path.lstrip("/"))
        response = self._session.post(url, json=json)
        response.raise_for_status()
        return response.json()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def list_repos(self, owner: str) -> list[str]:
        """Return all non-archived repository names for *owner*.

        Works for both individual GitHub users and organisations.  Tries the
        organisation endpoint first and falls back to the user endpoint on a
        404.

        Parameters
        ----------
        owner:
            GitHub username or organisation name.
        """
        try:
            raw = self._paginate(f"orgs/{owner}/repos", type="all")
        except requests.HTTPError as exc:
            if exc.response is not None and exc.response.status_code == 404:
                raw = self._paginate(f"users/{owner}/repos", type="all")
            else:
                raise
        return [item["name"] for item in raw if not item.get("archived", False)]

    def get_commits(
        self,
        owner: str,
        repo: str,
        since: datetime,
        until: datetime,
        author: Optional[str] = None,
        include_stats: bool = False,
    ) -> list[Commit]:
        """Return commits in *repo* between *since* and *until* (inclusive).

        Parameters
        ----------
        owner:
            Repository owner (user or organisation).
        repo:
            Repository name.
        since:
            Lower bound (inclusive) – any timezone-aware or naive UTC datetime.
        until:
            Upper bound (inclusive).
        author:
            Optional GitHub username to filter commits by.
        include_stats:
            When *True*, fetch each commit's detail endpoint to populate
            ``additions`` and ``deletions``.  This makes one extra API call per
            commit, so use with awareness of rate limits.
        """
        params: dict[str, Any] = {
            "since": _to_iso(since),
            "until": _to_iso(until),
        }
        if author:
            params["author"] = author

        raw = self._paginate(f"repos/{owner}/{repo}/commits", **params)
        commits: list[Commit] = []
        for item in raw:
            commit = item.get("commit", {})
            author_info = commit.get("author") or {}
            stats: dict[str, int] = {}
            if include_stats:
                try:
                    detail = self._get(f"repos/{owner}/{repo}/commits/{item['sha']}")
                    stats = detail.get("stats", {})
                except requests.RequestException:
                    pass
            commits.append(
                Commit(
                    sha=item["sha"],
                    message=commit.get("message", "").splitlines()[0],
                    author=author_info.get("name") or item.get("author", {}).get("login", "unknown"),
                    authored_at=isoparse(author_info.get("date", "1970-01-01T00:00:00Z")),
                    url=item.get("html_url", ""),
                    repo=f"{owner}/{repo}",
                    additions=stats.get("additions", 0),
                    deletions=stats.get("deletions", 0),
                )
            )
        return commits

    def get_issues(
        self,
        owner: str,
        repo: str,
        since: datetime,
        until: datetime,
        state: str = "all",
    ) -> list[Issue]:
        """Return issues (excluding PRs) updated within [since, until].

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        since:
            Lower bound for *updated_at*.
        until:
            Upper bound for *updated_at*.
        state:
            ``"open"``, ``"closed"``, or ``"all"`` (default).
        """
        raw = self._paginate(
            f"repos/{owner}/{repo}/issues",
            state=state,
            since=_to_iso(since),
            per_page=100,
            direction="asc",
            sort="updated",
        )
        issues: list[Issue] = []
        until_aware = _ensure_utc(until)
        for item in raw:
            # GitHub's /issues endpoint mixes in PRs; skip them.
            if item.get("pull_request"):
                continue
            updated = isoparse(item.get("updated_at", "1970-01-01T00:00:00Z"))
            if _ensure_utc(updated) > until_aware:
                continue
            closed_at_str = item.get("closed_at")
            issues.append(
                Issue(
                    number=item["number"],
                    title=item.get("title", ""),
                    state=item.get("state", ""),
                    author=(item.get("user") or {}).get("login", "unknown"),
                    created_at=isoparse(item.get("created_at", "1970-01-01T00:00:00Z")),
                    closed_at=isoparse(closed_at_str) if closed_at_str else None,
                    url=item.get("html_url", ""),
                    repo=f"{owner}/{repo}",
                    labels=[label.get("name", "") for label in item.get("labels", [])],
                    body=item.get("body") or "",
                    comments=item.get("comments", 0),
                    assignees=[
                        (a.get("login") or "") for a in item.get("assignees", []) if a
                    ],
                )
            )
        return issues

    def get_pull_requests(
        self,
        owner: str,
        repo: str,
        since: datetime,
        until: datetime,
        state: str = "all",
        include_details: bool = False,
    ) -> list[PullRequest]:
        """Return pull requests whose *updated_at* falls within [since, until].

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        since:
            Lower bound.
        until:
            Upper bound.
        state:
            ``"open"``, ``"closed"``, or ``"all"`` (default).
        include_details:
            When *True*, fetch each PR's detail endpoint to populate
            ``additions``, ``deletions``, ``changed_files``, ``commits_count``,
            ``review_comments``, and ``reviewer_comments`` (per-reviewer comment
            counts).  This makes two extra API calls per PR.
        """
        raw = self._paginate(
            f"repos/{owner}/{repo}/pulls",
            state=state,
            per_page=100,
            direction="asc",
            sort="updated",
        )
        prs: list[PullRequest] = []
        since_aware = _ensure_utc(since)
        until_aware = _ensure_utc(until)
        for item in raw:
            updated = isoparse(item.get("updated_at", "1970-01-01T00:00:00Z"))
            updated_aware = _ensure_utc(updated)
            if updated_aware < since_aware or updated_aware > until_aware:
                continue
            merged_at_str = item.get("merged_at")

            additions = 0
            deletions = 0
            changed_files = 0
            commits_count = 0
            review_comments = 0
            reviewer_comments: dict[str, int] = {}
            if include_details:
                try:
                    detail = self._get(f"repos/{owner}/{repo}/pulls/{item['number']}")
                    additions = detail.get("additions", 0)
                    deletions = detail.get("deletions", 0)
                    changed_files = detail.get("changed_files", 0)
                    commits_count = detail.get("commits", 0)
                    review_comments = detail.get("review_comments", 0)
                except requests.RequestException:
                    pass
                try:
                    raw_review_comments = self._paginate(
                        f"repos/{owner}/{repo}/pulls/{item['number']}/review_comments"
                    )
                    for rc in raw_review_comments:
                        login = (rc.get("user") or {}).get("login", "")
                        if login:
                            reviewer_comments[login] = reviewer_comments.get(login, 0) + 1
                except requests.RequestException:
                    pass

            prs.append(
                PullRequest(
                    number=item["number"],
                    title=item.get("title", ""),
                    state=item.get("state", ""),
                    author=(item.get("user") or {}).get("login", "unknown"),
                    created_at=isoparse(item.get("created_at", "1970-01-01T00:00:00Z")),
                    merged_at=isoparse(merged_at_str) if merged_at_str else None,
                    url=item.get("html_url", ""),
                    repo=f"{owner}/{repo}",
                    labels=[label.get("name", "") for label in item.get("labels", [])],
                    body=item.get("body") or "",
                    draft=item.get("draft", False),
                    base_branch=(item.get("base") or {}).get("ref", ""),
                    head_branch=(item.get("head") or {}).get("ref", ""),
                    requested_reviewers=[
                        (r.get("login") or "") for r in item.get("requested_reviewers", []) if r
                    ],
                    additions=additions,
                    deletions=deletions,
                    changed_files=changed_files,
                    commits_count=commits_count,
                    review_comments=review_comments,
                    reviewer_comments=reviewer_comments,
                )
            )
        return prs

    def get_releases(
        self,
        owner: str,
        repo: str,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
    ) -> list[Release]:
        """Return releases for *repo*, optionally filtered to [since, until].

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        since:
            When provided, excludes releases published before this datetime.
        until:
            When provided, excludes releases published after this datetime.
        """
        raw = self._paginate(f"repos/{owner}/{repo}/releases")
        releases: list[Release] = []
        since_aware = _ensure_utc(since) if since else None
        until_aware = _ensure_utc(until) if until else None
        for item in raw:
            published_at_str = item.get("published_at")
            created_at_str = item.get("created_at", "1970-01-01T00:00:00Z")
            published_at = isoparse(published_at_str) if published_at_str else None
            if since_aware or until_aware:
                ref_dt = _ensure_utc(published_at) if published_at else _ensure_utc(isoparse(created_at_str))
                if since_aware and ref_dt < since_aware:
                    continue
                if until_aware and ref_dt > until_aware:
                    continue
            releases.append(
                Release(
                    tag=item.get("tag_name", ""),
                    name=item.get("name") or item.get("tag_name", ""),
                    body=item.get("body") or "",
                    created_at=isoparse(created_at_str),
                    published_at=published_at,
                    url=item.get("html_url", ""),
                    repo=f"{owner}/{repo}",
                    author=(item.get("author") or {}).get("login", ""),
                    prerelease=item.get("prerelease", False),
                    draft=item.get("draft", False),
                )
            )
        return releases

    def get_contributors(
        self,
        owner: str,
        repo: str,
    ) -> list[Contributor]:
        """Return contributors for *repo* sorted by contribution count descending.

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        """
        raw = self._paginate(f"repos/{owner}/{repo}/contributors", anon="false")
        contributors: list[Contributor] = []
        for item in raw:
            contributors.append(
                Contributor(
                    login=item.get("login", ""),
                    contributions=item.get("contributions", 0),
                    url=item.get("html_url", ""),
                    repo=f"{owner}/{repo}",
                )
            )
        return contributors

    def create_issue(
        self,
        owner: str,
        repo: str,
        title: str,
        body: str = "",
        labels: Optional[list[str]] = None,
        assignees: Optional[list[str]] = None,
        milestone: Optional[int] = None,
    ) -> dict:
        """Create a new issue in *repo* and return the raw API response.

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        title:
            Issue title.
        body:
            Issue body (markdown supported).
        labels:
            Optional list of label names to apply.
        assignees:
            Optional list of GitHub usernames to assign.
        milestone:
            Optional milestone number to attach the issue to.
        """
        payload: dict[str, Any] = {"title": title, "body": body}
        if labels:
            payload["labels"] = labels
        if assignees:
            payload["assignees"] = assignees
        if milestone is not None:
            payload["milestone"] = milestone
        return self._post(f"repos/{owner}/{repo}/issues", json=payload)

    def create_milestone(
        self,
        owner: str,
        repo: str,
        title: str,
        description: str = "",
        due_on: Optional[str] = None,
        state: str = "open",
    ) -> dict:
        """Create a milestone in *repo* and return the raw API response.

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        title:
            Milestone title.
        description:
            Optional description for the milestone.
        due_on:
            Optional ISO 8601 due date string (e.g. ``"2024-12-31T00:00:00Z"``).
        state:
            ``"open"`` (default) or ``"closed"``.
        """
        payload: dict[str, Any] = {"title": title, "state": state}
        if description:
            payload["description"] = description
        if due_on:
            payload["due_on"] = due_on
        return self._post(f"repos/{owner}/{repo}/milestones", json=payload)

    def list_milestones(
        self,
        owner: str,
        repo: str,
        state: str = "open",
    ) -> list[Milestone]:
        """Return milestones for *repo*.

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        state:
            ``"open"`` (default), ``"closed"``, or ``"all"``.
        """
        raw = self._paginate(f"repos/{owner}/{repo}/milestones", state=state)
        milestones: list[Milestone] = []
        for item in raw:
            due_on_str = item.get("due_on")
            milestones.append(
                Milestone(
                    number=item["number"],
                    title=item.get("title", ""),
                    state=item.get("state", ""),
                    description=item.get("description") or "",
                    due_on=isoparse(due_on_str) if due_on_str else None,
                    open_issues=item.get("open_issues", 0),
                    closed_issues=item.get("closed_issues", 0),
                    url=item.get("html_url", ""),
                    repo=f"{owner}/{repo}",
                )
            )
        return milestones


# ------------------------------------------------------------------
# Utility helpers
# ------------------------------------------------------------------


def _to_iso(dt: datetime) -> str:
    """Render *dt* as an ISO-8601 UTC string suitable for GitHub API params."""
    return _ensure_utc(dt).strftime("%Y-%m-%dT%H:%M:%SZ")


def _ensure_utc(dt: datetime) -> datetime:
    """Return *dt* as a timezone-aware UTC datetime."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
