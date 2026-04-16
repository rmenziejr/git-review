"""Shared Rich table builders for git-review output."""

from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone

from rich.console import Console, RenderableType
from rich.panel import Panel
from rich.table import Table

from .models import Commit, Contributor, Issue, PullRequest, Release, ReviewSummary

_DAYS_OPEN_BUCKETS: list[tuple[str, int | None]] = [
    ("0–7 days", 7),
    ("8–30 days", 30),
    ("31–90 days", 90),
    ("91+ days", None),
]


def build_review_renderables(summary: ReviewSummary) -> list[RenderableType]:
    """Build the same header/panels/tables used by the review CLI output."""
    show_repo = summary.repo == "*"
    renderables: list[RenderableType] = [
        _build_header(summary.owner, summary.repo, summary.since, summary.until),
        *_build_commits_renderables(summary.commits, show_repo=show_repo),
        *_build_repo_stats_renderables(summary.commits),
        *_build_issues_renderables(summary.issues, show_repo=show_repo),
        *_build_issue_days_open_renderables(summary.issues),
        *_build_prs_renderables(summary.pull_requests, show_repo=show_repo),
        *_build_releases_renderables(summary.releases, show_repo=show_repo),
        *_build_contributors_renderables(summary.contributors),
    ]
    return renderables


def render_review_tables(summary: ReviewSummary, console: Console | None = None) -> None:
    """Render review output for Python callers using a Rich console."""
    target_console = console or Console()
    target_console.print()
    for renderable in build_review_renderables(summary):
        target_console.print(renderable)
        target_console.print()


def _build_header(owner: str, repo: str, since: datetime, until: datetime) -> Panel:
    repo_display = f"{owner}/*  (all repos)" if repo == "*" else f"{owner}/{repo}"
    return Panel(
        f"[bold]{repo_display}[/bold]  ·  " f"{since.date()} → {until.date()}",
        title="[bold blue]git-review[/bold blue]",
        border_style="blue",
    )


def _build_commits_renderables(
    commits: list[Commit], *, show_repo: bool = False
) -> list[RenderableType]:
    if not commits:
        return ["[dim]No commits found in this time window.[/dim]"]

    table = Table(title=f"Commits ({len(commits)})", show_lines=False)
    table.add_column("SHA", style="dim", no_wrap=True, width=9)
    table.add_column("Date", no_wrap=True, width=12)
    table.add_column("Author", no_wrap=True)
    if show_repo:
        table.add_column("Repo", no_wrap=True)
    table.add_column("Message")
    table.add_column("+", style="green", justify="right", no_wrap=True, width=8)
    table.add_column("-", style="red", justify="right", no_wrap=True, width=8)

    for commit in commits:
        row = [commit.sha[:7], str(commit.authored_at.date()), commit.author]
        if show_repo:
            row.append(commit.repo)
        row.append(commit.message[:80] + ("…" if len(commit.message) > 80 else ""))
        row.append(f"+{commit.additions}" if commit.additions else "")
        row.append(f"-{commit.deletions}" if commit.deletions else "")
        table.add_row(*row)
    return [table]


def _build_repo_stats_renderables(commits: list[Commit]) -> list[RenderableType]:
    if not commits:
        return []

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"commits": 0, "additions": 0, "deletions": 0}
    )
    for commit in commits:
        stats[commit.repo]["commits"] += 1
        stats[commit.repo]["additions"] += commit.additions
        stats[commit.repo]["deletions"] += commit.deletions

    table = Table(title="Repo Stats", show_lines=False)
    table.add_column("Repo", no_wrap=True)
    table.add_column("Commits", justify="right", width=9)
    table.add_column("Additions", style="green", justify="right", width=11)
    table.add_column("Deletions", style="red", justify="right", width=11)

    for repo_name, data in sorted(stats.items()):
        table.add_row(
            repo_name,
            str(data["commits"]),
            f"+{data['additions']}",
            f"-{data['deletions']}",
        )
    return [table]


def _build_issues_renderables(
    issues: list[Issue], *, show_repo: bool = False
) -> list[RenderableType]:
    if not issues:
        return ["[dim]No issues found in this time window.[/dim]"]

    table = Table(title=f"Issues ({len(issues)})", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=6)
    table.add_column("State", width=8)
    table.add_column("Author", no_wrap=True)
    if show_repo:
        table.add_column("Repo", no_wrap=True)
    table.add_column("Title")
    table.add_column("Labels")

    state_styles = {"open": "green", "closed": "red"}
    for issue in issues:
        style = state_styles.get(issue.state, "white")
        row = [str(issue.number), f"[{style}]{issue.state}[/{style}]", issue.author]
        if show_repo:
            row.append(issue.repo)
        row += [
            issue.title[:80] + ("…" if len(issue.title) > 80 else ""),
            ", ".join(issue.labels) or "—",
        ]
        table.add_row(*row)
    return [table]


def _build_prs_renderables(
    prs: list[PullRequest], *, show_repo: bool = False
) -> list[RenderableType]:
    if not prs:
        return ["[dim]No pull requests found in this time window.[/dim]"]

    table = Table(title=f"Pull Requests ({len(prs)})", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=6)
    table.add_column("State", width=8)
    table.add_column("Author", no_wrap=True)
    if show_repo:
        table.add_column("Repo", no_wrap=True)
    table.add_column("Title")
    table.add_column("Merged", width=12)
    table.add_column("Reviewers (comments)", no_wrap=False)

    for pr in prs:
        merged_str = str(pr.merged_at.date()) if pr.merged_at else "—"
        state_style = "green" if pr.state == "open" else ("magenta" if pr.merged_at else "red")
        if pr.reviewer_comments:
            reviewers_str = ", ".join(
                f"{login}({count})"
                for login, count in sorted(
                    pr.reviewer_comments.items(), key=lambda item: item[1], reverse=True
                )
            )
        else:
            reviewers_str = "—"
        row = [str(pr.number), f"[{state_style}]{pr.state}[/{state_style}]", pr.author]
        if show_repo:
            row.append(pr.repo)
        row += [
            pr.title[:80] + ("…" if len(pr.title) > 80 else ""),
            merged_str,
            reviewers_str,
        ]
        table.add_row(*row)
    return [table]


def _build_releases_renderables(
    releases: list[Release], *, show_repo: bool = False
) -> list[RenderableType]:
    if not releases:
        return []

    table = Table(title=f"Releases ({len(releases)})", show_lines=False)
    table.add_column("Tag", no_wrap=True)
    if show_repo:
        table.add_column("Repo", no_wrap=True)
    table.add_column("Name")
    table.add_column("Author", no_wrap=True)
    table.add_column("Published", width=12)
    table.add_column("Pre-release", width=11)

    for release in releases:
        pub_str = str(release.published_at.date()) if release.published_at else "—"
        pre_str = "[yellow]yes[/yellow]" if release.prerelease else "no"
        row = [release.tag]
        if show_repo:
            row.append(release.repo)
        row += [
            release.name[:60] + ("…" if len(release.name) > 60 else ""),
            release.author or "—",
            pub_str,
            pre_str,
        ]
        table.add_row(*row)
    return [table]


def _days_open_bucket(days: int) -> str:
    for label, max_days in _DAYS_OPEN_BUCKETS:
        if max_days is None or days <= max_days:
            return label
    return _DAYS_OPEN_BUCKETS[-1][0]


def _build_issue_days_open_renderables(issues: list[Issue]) -> list[RenderableType]:
    if not issues:
        return []

    now = datetime.now(tz=timezone.utc)
    bucket_labels = [label for label, _ in _DAYS_OPEN_BUCKETS]
    open_issues = [issue for issue in issues if issue.state == "open"]
    closed_issues = [issue for issue in issues if issue.state == "closed"]
    renderables: list[RenderableType] = []

    for subset, title in (
        (open_issues, "Open Issue Age (Days Open)"),
        (closed_issues, "Closed Issue Age (Days Open)"),
    ):
        if not subset:
            continue

        stats: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for issue in subset:
            end = issue.closed_at or now
            days = (end - issue.created_at).days
            bucket = _days_open_bucket(days)
            stats[issue.repo][bucket] += 1

        table = Table(title=title, show_lines=False)
        table.add_column("Repo", no_wrap=True)
        for label in bucket_labels:
            table.add_column(label, justify="right", width=12)

        for repo_name, buckets in sorted(stats.items()):
            row = [repo_name] + [str(buckets.get(label, 0)) for label in bucket_labels]
            table.add_row(*row)

        renderables.append(table)

    return renderables


def _build_contributors_renderables(contributors: list[Contributor]) -> list[RenderableType]:
    if not contributors:
        return []

    agg: dict[str, int] = defaultdict(int)
    for contributor in contributors:
        agg[contributor.login] += contributor.contributions

    table = Table(title=f"Contributors ({len(agg)})", show_lines=False)
    table.add_column("Username", no_wrap=True)
    table.add_column("Contributions", justify="right", width=14)

    for login, total in sorted(agg.items(), key=lambda item: item[1], reverse=True):
        table.add_row(login, str(total))
    return [table]
