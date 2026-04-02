"""CLI entry-point for git-review.

Usage examples
--------------
# Single repo – last 7 days (default)
git-review review --repo owner/repo --token ghp_xxx --openai-key sk-xxx

# All repos for an owner
git-review review --owner myorg --days 14 --token ghp_xxx --openai-key sk-xxx

# Explicit date range
git-review review --repo owner/repo --since 2024-01-01 --until 2024-01-31

# Delta shortcut
git-review review --repo owner/repo --days 30

# Skip LLM, just show the raw tables
git-review review --repo owner/repo --days 7 --no-summary
"""

from __future__ import annotations

import logging
import os
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .commit_message_generator import CommitMessageGenerator, get_git_diff
from .github_client import GitHubClient
from .issue_factory import IssueFactory, IssueDraft
from .llm_client import LLMClient
from .models import Commit, Contributor, Issue, PullRequest, Release, ReviewSummary
from .prompt_utils import load_prompt_file, validate_prompt_template

console = Console()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_git_root(path: str) -> Optional[str]:
    """Walk up from *path* until a directory containing ``.git`` is found.

    Returns the first matching directory, or *None* if none is found.
    """
    current = os.path.abspath(path)
    while True:
        if os.path.exists(os.path.join(current, ".git")):
            return current
        parent = os.path.dirname(current)
        if parent == current:
            return None
        current = parent


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.version_option(package_name="git-review")
def main() -> None:
    """git-review – summarise GitHub activity with AI."""


# ---------------------------------------------------------------------------
# review command
# ---------------------------------------------------------------------------

@main.command()
@click.option(
    "--repo",
    default=None,
    envvar="GITREVIEW_REPO",
    metavar="OWNER/REPO",
    help="GitHub repository in 'owner/repo' format.",
)
@click.option(
    "--owner",
    default=None,
    metavar="OWNER",
    help="GitHub user or organisation – reviews ALL non-archived repos.",
)
@click.option(
    "--token",
    envvar="GITHUB_TOKEN",
    default=None,
    help="GitHub personal access token (or set GITHUB_TOKEN env var).",
)
@click.option(
    "--since",
    "since_str",
    default=None,
    metavar="YYYY-MM-DD",
    help="Start of the time window (inclusive).",
)
@click.option(
    "--until",
    "until_str",
    default=None,
    metavar="YYYY-MM-DD",
    help="End of the time window (inclusive, defaults to today).",
)
@click.option(
    "--days",
    default=7,
    show_default=True,
    type=int,
    help="How many days back to look (used when --since is not set).",
)
@click.option(
    "--author",
    default=None,
    help="Filter commits by this GitHub username.",
)
@click.option(
    "--openai-key",
    envvar="OPENAI_API_KEY",
    default=None,
    help="OpenAI API key (or set OPENAI_API_KEY env var).",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model to use for summarisation.",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    default=None,
    help="Custom OpenAI-compatible API base URL.",
)
@click.option(
    "--no-summary",
    is_flag=True,
    default=False,
    help="Skip LLM summarisation and only print the data tables.",
)
@click.option(
    "--output",
    "-o",
    "output_file",
    default=None,
    metavar="FILE",
    help="Write the AI summary to a markdown file instead of (or in addition to) stdout.",
)
@click.option(
    "--prompt-file",
    "-p",
    "prompt_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    metavar="FILE",
    help="Path to a Jinja2 template file that overrides the default system prompt. "
         "Available variables: n, n_commits, n_issues, n_prs.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def review(
    repo: Optional[str],
    owner: Optional[str],
    token: Optional[str],
    since_str: Optional[str],
    until_str: Optional[str],
    days: int,
    author: Optional[str],
    openai_key: Optional[str],
    model: str,
    base_url: Optional[str],
    no_summary: bool,
    output_file: Optional[str],
    prompt_file: Optional[str],
    verbose: bool,
) -> None:
    """Fetch GitHub activity and generate an AI summary.

    Provide either --repo OWNER/REPO for a single repository, or
    --owner OWNER to review all non-archived repositories for that user
    or organisation.
    """

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # --- Validate target selection ------------------------------------
    if repo and owner:
        raise click.UsageError("Provide either --repo or --owner, not both.")
    if not repo and not owner:
        raise click.UsageError("Provide one of --repo OWNER/REPO or --owner OWNER.")

    # --- Parse / derive date window -----------------------------------
    now_utc = datetime.now(tz=timezone.utc).replace(hour=23, minute=59, second=59, microsecond=0)

    if since_str:
        try:
            since = datetime.strptime(since_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            raise click.BadParameter(f"Invalid date '{since_str}'. Use YYYY-MM-DD.", param_hint="--since")
    else:
        since = (now_utc - timedelta(days=days)).replace(hour=0, minute=0, second=0, microsecond=0)

    if until_str:
        try:
            until = datetime.strptime(until_str, "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
        except ValueError:
            raise click.BadParameter(f"Invalid date '{until_str}'. Use YYYY-MM-DD.", param_hint="--until")
    else:
        until = now_utc

    if since > until:
        raise click.UsageError("--since must be earlier than --until.")

    # --- Resolve owner + list of repos to scan ------------------------
    gh = GitHubClient(token=token)

    if repo:
        parts = repo.split("/", 1)
        if len(parts) != 2 or not all(parts):
            raise click.BadParameter("Expected format: owner/repo", param_hint="--repo")
        resolved_owner, repo_names = parts[0], [parts[1]]
    else:
        resolved_owner = owner  # type: ignore[assignment]
        with console.status(f"[bold green]Listing repositories for {owner}…"):
            try:
                repo_names = gh.list_repos(resolved_owner)
            except Exception as exc:
                console.print(f"[red]Error listing repositories:[/red] {exc}")
                sys.exit(1)
        if not repo_names:
            console.print(f"[yellow]No repositories found for {owner}.[/yellow]")
            return

    # --- Aggregate data across all target repos -----------------------
    all_repos_mode = len(repo_names) > 1
    repo_label = "*" if all_repos_mode else repo_names[0]
    review_data = ReviewSummary(
        owner=resolved_owner, repo=repo_label, since=since, until=until
    )

    for repo_name in repo_names:
        if all_repos_mode:
            console.print(f"[dim]Scanning {resolved_owner}/{repo_name}…[/dim]")

        with console.status(f"[bold green]Fetching commits for {repo_name}…"):
            try:
                review_data.commits += gh.get_commits(
                    resolved_owner, repo_name, since, until, author=author,
                    include_stats=True,
                )
            except Exception as exc:
                console.print(f"[yellow]  Skipping commits for {repo_name}:[/yellow] {exc}")

        with console.status(f"[bold green]Fetching issues for {repo_name}…"):
            try:
                review_data.issues += gh.get_issues(resolved_owner, repo_name, since, until)
            except Exception as exc:
                console.print(f"[yellow]  Skipping issues for {repo_name}:[/yellow] {exc}")

        with console.status(f"[bold green]Fetching pull requests for {repo_name}…"):
            try:
                review_data.pull_requests += gh.get_pull_requests(
                    resolved_owner, repo_name, since, until, include_details=True
                )
            except Exception as exc:
                console.print(f"[yellow]  Skipping pull requests for {repo_name}:[/yellow] {exc}")

        with console.status(f"[bold green]Fetching releases for {repo_name}…"):
            try:
                review_data.releases += gh.get_releases(resolved_owner, repo_name, since, until)
            except Exception as exc:
                console.print(f"[yellow]  Skipping releases for {repo_name}:[/yellow] {exc}")

        with console.status(f"[bold green]Fetching contributors for {repo_name}…"):
            try:
                review_data.contributors += gh.get_contributors(resolved_owner, repo_name)
            except Exception as exc:
                console.print(f"[yellow]  Skipping contributors for {repo_name}:[/yellow] {exc}")

    # --- Print rich tables ------------------------------------------
    _print_header(resolved_owner, repo_label, since, until)
    _print_commits_table(review_data.commits, show_repo=all_repos_mode)
    _print_repo_stats_table(review_data.commits)
    _print_issues_table(review_data.issues, show_repo=all_repos_mode)
    _print_issue_days_open_stats_table(review_data.issues)
    _print_prs_table(review_data.pull_requests, show_repo=all_repos_mode)
    _print_releases_table(review_data.releases, show_repo=all_repos_mode)
    _print_contributors_table(review_data.contributors, show_repo=all_repos_mode)

    # --- LLM summarisation ------------------------------------------
    if no_summary:
        return

    effective_key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not base_url:
        console.print(
            "\n[yellow]⚠  No OpenAI API key found.[/yellow]  "
            "Pass [bold]--openai-key[/bold] or set [bold]OPENAI_API_KEY[/bold] to enable summarisation.\n"
        )
        return

    custom_prompt: Optional[str] = None
    if prompt_file:
        try:
            custom_prompt = load_prompt_file(prompt_file)
        except OSError as exc:
            console.print(f"[red]Error reading prompt file:[/red] {exc}")
            sys.exit(1)

    with console.status("[bold green]Generating AI summary…"):
        try:
            llm = LLMClient(
                api_key=effective_key,
                model=model,
                base_url=base_url,
                system_prompt=custom_prompt,
            )
            summary_text = llm.summarise(review_data)
        except Exception as exc:
            console.print(f"[red]Error generating summary:[/red] {exc}")
            sys.exit(1)

    console.print()
    console.print(
        Panel(
            Markdown(summary_text),
            title="[bold cyan]AI Summary[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )

    if output_file:
        try:
            with open(output_file, "w", encoding="utf-8") as fh:
                fh.write(summary_text)
            console.print(f"\n[green]✓ Summary written to[/green] {output_file}")
        except OSError as exc:
            console.print(f"[red]Error writing output file:[/red] {exc}")
            sys.exit(1)


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_header(owner: str, repo: str, since: datetime, until: datetime) -> None:
    repo_display = f"{owner}/*  (all repos)" if repo == "*" else f"{owner}/{repo}"
    console.print()
    console.print(
        Panel(
            f"[bold]{repo_display}[/bold]  ·  "
            f"{since.date()} → {until.date()}",
            title="[bold blue]git-review[/bold blue]",
            border_style="blue",
        )
    )
    console.print()


def _print_commits_table(commits: list[Commit], *, show_repo: bool = False) -> None:
    if not commits:
        console.print("[dim]No commits found in this time window.[/dim]\n")
        return

    table = Table(title=f"Commits ({len(commits)})", show_lines=False)
    table.add_column("SHA", style="dim", no_wrap=True, width=9)
    table.add_column("Date", no_wrap=True, width=12)
    table.add_column("Author", no_wrap=True)
    if show_repo:
        table.add_column("Repo", no_wrap=True)
    table.add_column("Message")
    table.add_column("+", style="green", justify="right", no_wrap=True, width=8)
    table.add_column("-", style="red", justify="right", no_wrap=True, width=8)

    for c in commits:
        row = [
            c.sha[:7],
            str(c.authored_at.date()),
            c.author,
        ]
        if show_repo:
            row.append(c.repo)
        row.append(c.message[:80] + ("…" if len(c.message) > 80 else ""))
        row.append(f"+{c.additions}" if c.additions else "")
        row.append(f"-{c.deletions}" if c.deletions else "")
        table.add_row(*row)
    console.print(table)
    console.print()


def _print_repo_stats_table(commits: list[Commit]) -> None:
    if not commits:
        return

    stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"commits": 0, "additions": 0, "deletions": 0}
    )
    for c in commits:
        stats[c.repo]["commits"] += 1
        stats[c.repo]["additions"] += c.additions
        stats[c.repo]["deletions"] += c.deletions

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
    console.print(table)
    console.print()


def _print_issues_table(issues: list[Issue], *, show_repo: bool = False) -> None:
    if not issues:
        console.print("[dim]No issues found in this time window.[/dim]\n")
        return

    table = Table(title=f"Issues ({len(issues)})", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=6)
    table.add_column("State", width=8)
    table.add_column("Author", no_wrap=True)
    if show_repo:
        table.add_column("Repo", no_wrap=True)
    table.add_column("Title")
    table.add_column("Labels")

    state_styles = {"open": "green", "closed": "red"}
    for i in issues:
        style = state_styles.get(i.state, "white")
        row = [
            str(i.number),
            f"[{style}]{i.state}[/{style}]",
            i.author,
        ]
        if show_repo:
            row.append(i.repo)
        row += [
            i.title[:80] + ("…" if len(i.title) > 80 else ""),
            ", ".join(i.labels) or "—",
        ]
        table.add_row(*row)
    console.print(table)
    console.print()


def _print_prs_table(prs: list[PullRequest], *, show_repo: bool = False) -> None:
    if not prs:
        console.print("[dim]No pull requests found in this time window.[/dim]\n")
        return

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
                    pr.reviewer_comments.items(), key=lambda x: x[1], reverse=True
                )
            )
        else:
            reviewers_str = "—"
        row = [
            str(pr.number),
            f"[{state_style}]{pr.state}[/{state_style}]",
            pr.author,
        ]
        if show_repo:
            row.append(pr.repo)
        row += [
            pr.title[:80] + ("…" if len(pr.title) > 80 else ""),
            merged_str,
            reviewers_str,
        ]
        table.add_row(*row)
    console.print(table)
    console.print()


def _print_releases_table(releases: list[Release], *, show_repo: bool = False) -> None:
    if not releases:
        return

    table = Table(title=f"Releases ({len(releases)})", show_lines=False)
    table.add_column("Tag", no_wrap=True)
    if show_repo:
        table.add_column("Repo", no_wrap=True)
    table.add_column("Name")
    table.add_column("Author", no_wrap=True)
    table.add_column("Published", width=12)
    table.add_column("Pre-release", width=11)

    for r in releases:
        pub_str = str(r.published_at.date()) if r.published_at else "—"
        pre_str = "[yellow]yes[/yellow]" if r.prerelease else "no"
        row = [r.tag]
        if show_repo:
            row.append(r.repo)
        row += [
            r.name[:60] + ("…" if len(r.name) > 60 else ""),
            r.author or "—",
            pub_str,
            pre_str,
        ]
        table.add_row(*row)
    console.print(table)
    console.print()


_DAYS_OPEN_BUCKETS: list[tuple[str, int | None]] = [
    ("0–7 days", 7),
    ("8–30 days", 30),
    ("31–90 days", 90),
    ("91+ days", None),
]


def _days_open_bucket(days: int) -> str:
    for label, max_days in _DAYS_OPEN_BUCKETS:
        if max_days is None or days <= max_days:
            return label
    return _DAYS_OPEN_BUCKETS[-1][0]  # pragma: no cover – last bucket has max_days=None


def _print_issue_days_open_stats_table(issues: list[Issue]) -> None:
    if not issues:
        return

    now = datetime.now(tz=timezone.utc)
    bucket_labels = [label for label, _ in _DAYS_OPEN_BUCKETS]

    open_issues = [i for i in issues if i.state == "open"]
    closed_issues = [i for i in issues if i.state == "closed"]

    for subset, title in (
        (open_issues, "Open Issue Age (Days Open)"),
        (closed_issues, "Closed Issue Age (Days Open)"),
    ):
        if not subset:
            continue

        # stats[repo][bucket] = count
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

        console.print(table)
        console.print()


def _print_contributors_table(
    contributors: list[Contributor], *, show_repo: bool = False
) -> None:
    if not contributors:
        return

    # Aggregate contributions across repos when in all-repos mode
    agg: dict[str, int] = defaultdict(int)
    for c in contributors:
        agg[c.login] += c.contributions

    table = Table(title=f"Contributors ({len(agg)})", show_lines=False)
    table.add_column("Username", no_wrap=True)
    table.add_column("Contributions", justify="right", width=14)

    for login, total in sorted(agg.items(), key=lambda x: x[1], reverse=True):
        table.add_row(login, str(total))
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# create-issues command
# ---------------------------------------------------------------------------

@main.command("create-issues")
@click.option(
    "--repo",
    required=True,
    envvar="GITREVIEW_REPO",
    metavar="OWNER/REPO",
    help="Target GitHub repository in 'owner/repo' format.",
)
@click.option(
    "--requirements",
    "requirements_file",
    required=True,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    metavar="FILE",
    help="Path to a markdown file containing requirements.",
)
@click.option(
    "--token",
    envvar="GITHUB_TOKEN",
    default=None,
    help="GitHub personal access token (or set GITHUB_TOKEN env var).",
)
@click.option(
    "--openai-key",
    envvar="OPENAI_API_KEY",
    default=None,
    help="OpenAI API key (or set OPENAI_API_KEY env var).",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model to use for issue generation.",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    default=None,
    help="Custom OpenAI-compatible API base URL.",
)
@click.option(
    "--yes",
    "-y",
    is_flag=True,
    default=False,
    help="Skip interactive approval and push all issues immediately.",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Parse and display issue drafts but do not push to GitHub.",
)
@click.option(
    "--prompt-file",
    "-p",
    "prompt_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    metavar="FILE",
    help="Path to a Jinja2 template file that overrides the default system prompt. "
         "No template variables are available for this command.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def create_issues(
    repo: str,
    requirements_file: str,
    token: Optional[str],
    openai_key: Optional[str],
    model: str,
    base_url: Optional[str],
    yes: bool,
    dry_run: bool,
    prompt_file: Optional[str],
    verbose: bool,
) -> None:
    """Parse a markdown requirements file and create GitHub issues.

    Reads REQUIREMENTS_FILE, uses an LLM to extract individual requirements as
    issue drafts, lets you review them, then creates them in OWNER/REPO.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    parts = repo.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise click.BadParameter("Expected format: owner/repo", param_hint="--repo")
    owner, repo_name = parts

    effective_key = openai_key
    if not effective_key and not base_url:
        raise click.UsageError(
            "An OpenAI API key is required. Pass --openai-key or set OPENAI_API_KEY."
        )

    with open(requirements_file, encoding="utf-8") as fh:
        markdown_text = fh.read()

    custom_prompt: Optional[str] = None
    if prompt_file:
        try:
            custom_prompt = load_prompt_file(prompt_file)
        except OSError as exc:
            console.print(f"[red]Error reading prompt file:[/red] {exc}")
            sys.exit(1)

    gh = GitHubClient(token=token)

    with console.status("[bold green]Parsing requirements with LLM…"):
        try:
            factory = IssueFactory(
                github_client=gh,
                openai_api_key=effective_key,
                model=model,
                base_url=base_url,
                system_prompt=custom_prompt,
            )
            drafts = factory.parse_requirements(markdown_text)
        except Exception as exc:
            console.print(f"[red]Error parsing requirements:[/red] {exc}")
            sys.exit(1)

    if not drafts:
        console.print("[yellow]No issues were extracted from the requirements document.[/yellow]")
        return

    _print_issue_drafts(drafts)

    if dry_run:
        console.print("[dim]Dry-run mode – no issues were created.[/dim]")
        return

    approved: list[IssueDraft] = []
    if yes:
        approved = drafts
    else:
        console.print()
        for i, draft in enumerate(drafts, 1):
            answer = click.prompt(
                f"  Push issue {i}/{len(drafts)} '{draft.title}'? [y/n/q]",
                default="y",
            )
            if answer.lower() == "q":
                console.print("[yellow]Aborted.[/yellow]")
                break
            if answer.lower() == "y":
                approved.append(draft)

    if not approved:
        console.print("[dim]No issues were pushed.[/dim]")
        return

    with console.status(f"[bold green]Creating {len(approved)} issue(s) in {owner}/{repo_name}…"):
        try:
            results = factory.push_issues(owner, repo_name, approved)
        except Exception as exc:
            console.print(f"[red]Error creating issues:[/red] {exc}")
            sys.exit(1)

    console.print(f"\n[green]✓ Created {len(results)} issue(s):[/green]")
    for r in results:
        console.print(f"  #{r.get('number')}  {r.get('html_url', '')}")


def _print_issue_drafts(drafts: list) -> None:
    """Display a summary table of issue drafts."""
    table = Table(title=f"Issue Drafts ({len(drafts)})", show_lines=True)
    table.add_column("#", justify="right", style="dim", width=4)
    table.add_column("Title")
    table.add_column("Labels")
    table.add_column("Assignees")

    for i, draft in enumerate(drafts, 1):
        table.add_row(
            str(i),
            draft.title,
            ", ".join(draft.labels) or "—",
            ", ".join(draft.assignees) or "—",
        )
    console.print()
    console.print(table)
    console.print()


# ---------------------------------------------------------------------------
# commit-message command
# ---------------------------------------------------------------------------

@main.command("commit-message")
@click.option(
    "--repo-path",
    default=None,
    show_default=False,
    metavar="PATH",
    help="Path to the git repository (defaults to the current directory, "
         "walking up to find the .git root).",
)
@click.option(
    "--openai-key",
    envvar="OPENAI_API_KEY",
    default=None,
    help="OpenAI API key (or set OPENAI_API_KEY env var).",
)
@click.option(
    "--model",
    default="gpt-4o-mini",
    show_default=True,
    help="LLM model to use for commit message generation.",
)
@click.option(
    "--base-url",
    envvar="OPENAI_BASE_URL",
    default=None,
    help="Custom OpenAI-compatible API base URL.",
)
@click.option(
    "--prompt-file",
    "-p",
    "prompt_file",
    default=None,
    type=click.Path(exists=True, dir_okay=False, readable=True),
    metavar="FILE",
    help="Path to a Jinja2 template file that overrides the default system prompt. "
         "No template variables are available for this command.",
)
@click.option(
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def commit_message(
    repo_path: Optional[str],
    openai_key: Optional[str],
    model: str,
    base_url: Optional[str],
    prompt_file: Optional[str],
    verbose: bool,
) -> None:
    """Generate a commit message for the current git repository.

    Reads the staged diff (or falls back to the unstaged diff) and asks
    an LLM to produce a Conventional Commit message.
    """
    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Resolve the git root: walk up from cwd (or the given path) to find .git
    start = os.path.abspath(repo_path) if repo_path else os.getcwd()
    resolved_repo_path = _find_git_root(start)
    if resolved_repo_path is None:
        console.print(
            f"[red]Error:[/red] No git repository found at or above '{start}'."
        )
        sys.exit(1)

    effective_key = openai_key or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not base_url:
        raise click.UsageError(
            "An OpenAI API key is required. Pass --openai-key or set OPENAI_API_KEY."
        )

    custom_prompt: Optional[str] = None
    if prompt_file:
        try:
            custom_prompt = load_prompt_file(prompt_file)
        except OSError as exc:
            console.print(f"[red]Error reading prompt file:[/red] {exc}")
            sys.exit(1)

    try:
        diff = get_git_diff(resolved_repo_path)
    except RuntimeError as exc:
        console.print(f"[red]Error reading git diff:[/red] {exc}")
        sys.exit(1)

    if not diff.strip():
        console.print(
            "[yellow]No changes detected.[/yellow] Stage or make some changes first."
        )
        sys.exit(1)

    with console.status("[bold green]Generating commit message…"):
        try:
            generator = CommitMessageGenerator(
                api_key=effective_key,
                model=model,
                base_url=base_url,
                system_prompt=custom_prompt,
            )
            message = generator.generate(diff)
        except Exception as exc:
            console.print(f"[red]Error generating commit message:[/red] {exc}")
            sys.exit(1)

    console.print()
    console.print(
        Panel(
            message,
            title="[bold cyan]Suggested Commit Message[/bold cyan]",
            border_style="cyan",
            padding=(1, 2),
        )
    )
    console.print()

    # --- Edit step ---------------------------------------------------------
    if click.confirm("Edit this message?", default=False):
        edited = click.edit(message)
        if edited is not None:
            message = edited.strip()
            if not message:
                console.print("[red]Commit message is empty after editing. Aborting.[/red]")
                sys.exit(1)
            console.print()
            console.print(
                Panel(
                    message,
                    title="[bold cyan]Edited Commit Message[/bold cyan]",
                    border_style="cyan",
                    padding=(1, 2),
                )
            )
            console.print()

    # --- Commit step -------------------------------------------------------
    if click.confirm("Commit with this message?", default=False):
        # Split on the blank line separating subject from body (Conventional Commits)
        parts = message.split("\n\n", 1)
        subject = parts[0].strip()
        body = parts[1].strip() if len(parts) > 1 else ""

        cmd = ["git", "commit", "-m", subject]
        if body:
            cmd += ["-m", body]

        try:
            subprocess.run(
                cmd,
                cwd=resolved_repo_path,
                check=True,
            )
        except subprocess.CalledProcessError as exc:
            console.print(f"[red]git commit failed:[/red] {exc}")
            sys.exit(1)
        console.print("[bold green]Committed successfully.[/bold green]")
