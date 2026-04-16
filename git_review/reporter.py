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
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from typing import Optional

from .github_client import GitHubClient
from .models import AuthorSummary, Commit, Contributor, Issue, PullRequest, Release, ReviewSummary

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
            with ThreadPoolExecutor(max_workers=5) as executor:
                futures = {
                    executor.submit(
                        self._gh.get_commits,
                        owner, repo_name, since, until,
                        author=author, include_stats=include_stats, branch=branch,
                    ): "commits",
                    executor.submit(
                        self._gh.get_issues, owner, repo_name, since, until
                    ): "issues",
                    executor.submit(
                        self._gh.get_pull_requests,
                        owner, repo_name, since, until,
                        include_details=include_details,
                    ): "pull_requests",
                    executor.submit(
                        self._gh.get_releases, owner, repo_name, since, until
                    ): "releases",
                    executor.submit(
                        self._gh.get_contributors, owner, repo_name
                    ): "contributors",
                }

                for future in as_completed(futures):
                    section = futures[future]
                    try:
                        results = future.result()
                        if section == "commits":
                            summary.commits += results
                        elif section == "issues":
                            summary.issues += results
                        elif section == "pull_requests":
                            summary.pull_requests += results
                        elif section == "releases":
                            summary.releases += results
                        elif section == "contributors":
                            summary.contributors += results
                    except Exception:
                        pass

        return summary

    # ------------------------------------------------------------------
    # Markdown rendering
    # ------------------------------------------------------------------

    @staticmethod
    def to_markdown(
        summary: ReviewSummary,
        *,
        include_author_summaries: bool = True,
    ) -> str:
        """Render *summary* as a GFM Markdown report.

        The output mirrors the table sections produced by the CLI but in
        plain Markdown so it can be saved to a file, embedded in a PR
        description, or processed by downstream tools.

        Parameters
        ----------
        summary:
            A :class:`~git_review.models.ReviewSummary` obtained from
            :meth:`fetch` or built manually.
        include_author_summaries:
            When *True* (default), append a **By Author** section that breaks
            down commits, issues, PRs, and releases per contributor.

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

        if include_author_summaries:
            author_summaries = ReviewReporter.partition_by_author(summary)
            if author_summaries:
                parts.append(ReviewReporter.author_summaries_to_markdown(author_summaries))

        return "\n\n".join(section for section in parts if section)

    @staticmethod
    def partition_by_author(summary: ReviewSummary) -> dict[str, AuthorSummary]:
        """Partition *summary* activity into per-author slices.

        Commits are keyed by :attr:`~git_review.models.Commit.author` (the git
        author name / GitHub login stored on the object).  Issues, PRs, and
        releases are keyed by their ``author`` field (GitHub login).  An author
        who only appears in one category still gets their own entry.

        Parameters
        ----------
        summary:
            The aggregate review data to partition.

        Returns
        -------
        dict[str, AuthorSummary]
            Mapping from author identifier to their
            :class:`~git_review.models.AuthorSummary`, sorted alphabetically
            by author key.
        """
        result: dict[str, AuthorSummary] = {}

        def _get(key: str) -> AuthorSummary:
            if key not in result:
                result[key] = AuthorSummary(author=key)
            return result[key]

        for commit in summary.commits:
            _get(commit.author).commits.append(commit)
        for issue in summary.issues:
            _get(issue.author).issues.append(issue)
        for pr in summary.pull_requests:
            _get(pr.author).pull_requests.append(pr)
        for release in summary.releases:
            if release.author:
                _get(release.author).releases.append(release)

        return dict(sorted(result.items()))

    @staticmethod
    def author_summaries_to_markdown(author_summaries: dict[str, AuthorSummary]) -> str:
        """Render per-author activity tables as a GFM Markdown section.

        Parameters
        ----------
        author_summaries:
            The mapping returned by :meth:`partition_by_author`.

        Returns
        -------
        str
            A ``## By Author`` section containing a subsection for every
            author, each with a stats headline and mini-tables for commits,
            issues, PRs, and releases.
        """
        if not author_summaries:
            return ""

        sections: list[str] = ["## By Author"]
        for author, data in author_summaries.items():
            sections.append(_md_author_section(author, data))

        return "\n\n".join(sections)


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


def _md_author_section(author: str, data: AuthorSummary) -> str:
    """Render a single author's activity as a ``###`` subsection."""
    n_commits = len(data.commits)
    n_open = sum(1 for i in data.issues if i.state == "open")
    n_closed = sum(1 for i in data.issues if i.state == "closed")
    n_merged = sum(1 for pr in data.pull_requests if pr.merged_at)
    n_open_prs = sum(1 for pr in data.pull_requests if pr.state == "open")
    n_releases = len(data.releases)

    headline_parts = []
    if n_commits:
        headline_parts.append(f"**Commits:** {n_commits}")
    if data.issues:
        headline_parts.append(f"**Issues:** {len(data.issues)} ({n_open} open, {n_closed} closed)")
    if data.pull_requests:
        headline_parts.append(
            f"**Pull Requests:** {len(data.pull_requests)} ({n_merged} merged, {n_open_prs} open)"
        )
    if n_releases:
        headline_parts.append(f"**Releases:** {n_releases}")

    parts: list[str] = [f"### {author}", "  \n".join(headline_parts)]

    if data.commits:
        headers = ["SHA", "Date", "Repo", "Message", "+", "-"]
        rows = [
            [
                c.sha[:7],
                str(c.authored_at.date()),
                c.repo,
                c.message[:80] + ("…" if len(c.message) > 80 else ""),
                f"+{c.additions}" if c.additions else "",
                f"-{c.deletions}" if c.deletions else "",
            ]
            for c in data.commits
        ]
        parts.append(f"#### Commits ({n_commits})\n\n{_md_table(headers, rows)}")

    if data.issues:
        headers = ["#", "State", "Repo", "Title", "Labels"]
        rows = [
            [
                str(i.number),
                i.state,
                i.repo,
                i.title[:80] + ("…" if len(i.title) > 80 else ""),
                ", ".join(i.labels) if i.labels else "—",
            ]
            for i in data.issues
        ]
        parts.append(f"#### Issues ({len(data.issues)})\n\n{_md_table(headers, rows)}")

    if data.pull_requests:
        headers = ["#", "State", "Repo", "Title", "Merged"]
        rows = [
            [
                str(pr.number),
                pr.state,
                pr.repo,
                pr.title[:80] + ("…" if len(pr.title) > 80 else ""),
                str(pr.merged_at.date()) if pr.merged_at else "—",
            ]
            for pr in data.pull_requests
        ]
        parts.append(
            f"#### Pull Requests ({len(data.pull_requests)})\n\n{_md_table(headers, rows)}"
        )

    if data.releases:
        headers = ["Tag", "Repo", "Name", "Published", "Pre-release"]
        rows = [
            [
                r.tag,
                r.repo,
                r.name[:60] + ("…" if len(r.name) > 60 else ""),
                str(r.published_at.date()) if r.published_at else "—",
                "yes" if r.prerelease else "no",
            ]
            for r in data.releases
        ]
        parts.append(f"#### Releases ({n_releases})\n\n{_md_table(headers, rows)}")

    return "\n\n".join(parts)
