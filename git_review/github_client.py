"""GitHub REST API client for git-review."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional
from urllib.parse import urljoin

import requests
from dateutil.parser import isoparse

from .models import Commit, Issue, PullRequest

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
                )
            )
        return prs


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
