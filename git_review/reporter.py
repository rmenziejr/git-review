"""High-level SDK entry-point: fetch all GitHub activity and render to Markdown.

Example
-------
>>> from datetime import datetime, timedelta, timezone
>>> from git_review import GitHubClient, ReviewReporter
>>>
>>> gh = GitHubClient(token="ghp_xxx")
>>> since = datetime.now(tz=timezone.utc) - timedelta(days=7)
>>> until = datetime.now(tz=timezone.utc)
>>>
>>> # Single repository
>>> summary = ReviewReporter(gh).fetch("owner", since, until, repo="my-repo")
>>>
>>> # All repositories for an owner
>>> summary = ReviewReporter(gh).fetch("owner", since, until)
>>>
>>> # Get the full Markdown report
>>> md = ReviewReporter.to_markdown(summary)
>>> print(md)
"""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional

from .github_client import GitHubClient
from .models import Commit, Contributor, Issue, PullRequest, Release, ReviewSummary

_DAYS_OPEN_BUCKETS: list[tuple[str, int | None]] = [
    ("0–7 days", 7),
    ("8–30 days", 30),
    ("31–90 days", 90),
    ("91+ days", None),
]


class ReviewReporter:
    """Fetch GitHub activity and produce a Markdown report.

    Parameters
    ----------
    gh:
        An authenticated :class:`~git_review.GitHubClient` instance.
    """

    def __init__(self, gh: GitHubClient) -> None:
        self._gh = gh

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------

    def fetch(
        self,
        owner: str,
        since: datetime,
        until: datetime,
        *,
        repo: Optional[str] = None,
        author: Optional[str] = None,
        branch: str = "*",
        include_stats: bool = True,
        include_details: bool = True,
    ) -> ReviewSummary:
        """Fetch all GitHub activity and return a :class:`ReviewSummary`.

        Parameters
        ----------
        owner:
            GitHub username or organisation name.
        since:
            Start of the time window (inclusive).
        until:
            End of the time window (inclusive).
        repo:
            When provided, limit the fetch to this single repository name
            (without the owner prefix, e.g. ``"my-repo"``).  When *None*,
            all non-archived repositories for *owner* are scanned.
        author:
            Optional GitHub username to filter commits by.
        branch:
            Which branch(es) to include when fetching commits.

            * ``"*"`` (default) – every branch, deduplicated by SHA.
            * ``None`` – the repository's default branch only.
            * Any other string – that specific branch name.
        include_stats:
            When *True*, fetch per-commit addition/deletion counts
            (one extra API call per commit).
        include_details:
            When *True*, fetch per-PR addition/deletion/review counts
            (two extra API calls per PR).
        """
        if repo:
            repo_names = [repo]
            repo_label = repo
        else:
            repo_names = self._gh.list_repos(owner)
            repo_label = "*"

        summary = ReviewSummary(
            owner=owner,
            repo=repo_label,
            since=since,
            until=until,
        )

        for repo_name in repo_names:
            try:
                summary.commits += self._gh.get_commits(
                    owner, repo_name, since, until,
                    author=author, include_stats=include_stats, branch=branch,
                )
            except Exception:
                pass

            try:
                summary.issues += self._gh.get_issues(owner, repo_name, since, until)
            except Exception:
                pass

            try:
                summary.pull_requests += self._gh.get_pull_requests(
                    owner, repo_name, since, until,
                    include_details=include_details,
                )
            except Exception:
                pass

            try:
                summary.releases += self._gh.get_releases(owner, repo_name, since, until)
            except Exception:
                pass

            try:
                summary.contributors += self._gh.get_contributors(owner, repo_name)
            except Exception:
                pass

        return summary

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown(summary: ReviewSummary) -> str:
        """Render *summary* as a GFM Markdown report.

        The output mirrors the table sections produced by the CLI but in
        plain Markdown so it can be saved to a file, embedded in a PR
        description, or processed by downstream tools.

        Parameters
        ----------
        summary:
            A :class:`~git_review.models.ReviewSummary` obtained from
            :meth:`fetch` or built manually.

        Returns
        -------
        str
            A multi-section GFM Markdown string.
        """
        show_repo = summary.repo == "*"
        parts: list[str] = []

        parts.append(_md_header(summary.owner, summary.repo, summary.since, summary.until))
        parts.append(_md_commits(summary.commits, show_repo=show_repo))
        parts.append(_md_repo_stats(summary.commits))
        parts.append(_md_issues(summary.issues, show_repo=show_repo))
        parts.append(_md_issue_age(summary.issues))
        parts.append(_md_prs(summary.pull_requests, show_repo=show_repo))
        parts.append(_md_releases(summary.releases, show_repo=show_repo))
        parts.append(_md_contributors(summary.contributors))

        return "\n\n".join(section for section in parts if section)


# ---------------------------------------------------------------------------
# Private markdown builders
# ---------------------------------------------------------------------------


def _md_table(headers: list[str], rows: list[list[str]]) -> str:
    """Return a GFM markdown table string."""
    sep = ["-" * max(len(h), 3) for h in headers]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(sep) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _md_header(owner: str, repo: str, since: datetime, until: datetime) -> str:
    repo_display = f"{owner}/*" if repo == "*" else f"{owner}/{repo}"
    return (
        f"# git-review: {repo_display}\n\n"
        f"**Period:** {since.date()} → {until.date()}"
    )


def _md_commits(commits: list[Commit], *, show_repo: bool = False) -> str:
    if not commits:
        return "## Commits\n\n_No commits found in this time window._"

    headers = ["SHA", "Date", "Author"]
    if show_repo:
        headers.append("Repo")
    headers += ["Message", "+", "-"]

    rows = []
    for c in commits:
        row = [c.sha[:7], str(c.authored_at.date()), c.author]
        if show_repo:
            row.append(c.repo)
        row.append(c.message[:80] + ("…" if len(c.message) > 80 else ""))
        row.append(f"+{c.additions}" if c.additions else "")
        row.append(f"-{c.deletions}" if c.deletions else "")
        rows.append(row)

    return f"## Commits ({len(commits)})\n\n{_md_table(headers, rows)}"


def _md_repo_stats(commits: list[Commit]) -> str:
    if not commits:
        return ""

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"commits": 0, "additions": 0, "deletions": 0}
    )
    for c in commits:
        stats[c.repo]["commits"] += 1
        stats[c.repo]["additions"] += c.additions
        stats[c.repo]["deletions"] += c.deletions

    headers = ["Repo", "Commits", "Additions", "Deletions"]
    rows = [
        [
            repo_name,
            str(data["commits"]),
            f"+{data['additions']}",
            f"-{data['deletions']}",
        ]
        for repo_name, data in sorted(stats.items())
    ]
    return f"## Repo Stats\n\n{_md_table(headers, rows)}"


def _md_issues(issues: list[Issue], *, show_repo: bool = False) -> str:
    if not issues:
        return "## Issues\n\n_No issues found in this time window._"

    headers = ["#", "State", "Author"]
    if show_repo:
        headers.append("Repo")
    headers += ["Title", "Labels"]

    rows = []
    for issue in issues:
        row = [str(issue.number), issue.state, issue.author]
        if show_repo:
            row.append(issue.repo)
        row.append(issue.title[:80] + ("…" if len(issue.title) > 80 else ""))
        row.append(", ".join(issue.labels) if issue.labels else "—")
        rows.append(row)

    return f"## Issues ({len(issues)})\n\n{_md_table(headers, rows)}"


def _md_issue_age(issues: list[Issue]) -> str:
    if not issues:
        return ""

    now = datetime.now(tz=timezone.utc)
    bucket_labels = [label for label, _ in _DAYS_OPEN_BUCKETS]
    sections: list[str] = []

    for subset, title in (
        ([i for i in issues if i.state == "open"], "Open Issue Age"),
        ([i for i in issues if i.state == "closed"], "Closed Issue Age"),
    ):
        if not subset:
            continue

        repo_buckets: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for issue in subset:
            end = issue.closed_at or now
            days = (end - issue.created_at).days
            bucket = _days_open_bucket(days)
            repo_buckets[issue.repo][bucket] += 1

        headers = ["Repo"] + bucket_labels
        rows = [
            [repo_name] + [str(buckets.get(label, 0)) for label in bucket_labels]
            for repo_name, buckets in sorted(repo_buckets.items())
        ]
        sections.append(f"### {title}\n\n{_md_table(headers, rows)}")

    if not sections:
        return ""
    return "## Issue Age\n\n" + "\n\n".join(sections)


def _md_prs(prs: list[PullRequest], *, show_repo: bool = False) -> str:
    if not prs:
        return "## Pull Requests\n\n_No pull requests found in this time window._"

    headers = ["#", "State", "Author"]
    if show_repo:
        headers.append("Repo")
    headers += ["Title", "Merged", "Reviewers (comments)"]

    rows = []
    for pr in prs:
        merged_str = str(pr.merged_at.date()) if pr.merged_at else "—"
        if pr.reviewer_comments:
            reviewers_str = ", ".join(
                f"{login}({count})"
                for login, count in sorted(
                    pr.reviewer_comments.items(), key=lambda item: item[1], reverse=True
                )
            )
        else:
            reviewers_str = "—"
        row = [str(pr.number), pr.state, pr.author]
        if show_repo:
            row.append(pr.repo)
        row += [
            pr.title[:80] + ("…" if len(pr.title) > 80 else ""),
            merged_str,
            reviewers_str,
        ]
        rows.append(row)

    return f"## Pull Requests ({len(prs)})\n\n{_md_table(headers, rows)}"


def _md_releases(releases: list[Release], *, show_repo: bool = False) -> str:
    if not releases:
        return ""

    headers = ["Tag"]
    if show_repo:
        headers.append("Repo")
    headers += ["Name", "Author", "Published", "Pre-release"]

    rows = []
    for release in releases:
        pub_str = str(release.published_at.date()) if release.published_at else "—"
        row = [release.tag]
        if show_repo:
            row.append(release.repo)
        row += [
            release.name[:60] + ("…" if len(release.name) > 60 else ""),
            release.author or "—",
            pub_str,
            "yes" if release.prerelease else "no",
        ]
        rows.append(row)

    return f"## Releases ({len(releases)})\n\n{_md_table(headers, rows)}"


def _md_contributors(contributors: list[Contributor]) -> str:
    if not contributors:
        return ""

    agg: dict[str, int] = defaultdict(int)
    for contributor in contributors:
        agg[contributor.login] += contributor.contributions

    headers = ["Username", "Contributions"]
    rows = [
        [login, str(total)]
        for login, total in sorted(agg.items(), key=lambda item: item[1], reverse=True)
    ]
    return f"## Contributors ({len(agg)})\n\n{_md_table(headers, rows)}"


def _days_open_bucket(days: int) -> str:
    for label, max_days in _DAYS_OPEN_BUCKETS:
        if max_days is None or days <= max_days:
            return label
    return _DAYS_OPEN_BUCKETS[-1][0]
