"""CLI entry-point for git-review.

Usage examples
--------------
# Last 7 days (default)
git-review review --repo owner/repo --token ghp_xxx --openai-key sk-xxx

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
import sys
from datetime import datetime, timedelta, timezone
from typing import Optional

import click
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table

from .github_client import GitHubClient
from .llm_client import LLMClient
from .models import Commit, Issue, PullRequest, ReviewSummary

console = Console()


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
    required=True,
    envvar="GITREVIEW_REPO",
    metavar="OWNER/REPO",
    help="GitHub repository in 'owner/repo' format.",
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
    "--verbose",
    "-v",
    is_flag=True,
    default=False,
    help="Enable debug logging.",
)
def review(
    repo: str,
    token: Optional[str],
    since_str: Optional[str],
    until_str: Optional[str],
    days: int,
    author: Optional[str],
    openai_key: Optional[str],
    model: str,
    base_url: Optional[str],
    no_summary: bool,
    verbose: bool,
) -> None:
    """Fetch GitHub activity and generate an AI summary."""

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

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

    # --- Parse owner/repo -------------------------------------------
    parts = repo.split("/", 1)
    if len(parts) != 2 or not all(parts):
        raise click.BadParameter("Expected format: owner/repo", param_hint="--repo")
    owner, repo_name = parts

    # --- Fetch data from GitHub -------------------------------------
    gh = GitHubClient(token=token)
    review_data = ReviewSummary(owner=owner, repo=repo_name, since=since, until=until)

    with console.status("[bold green]Fetching commits…"):
        try:
            review_data.commits = gh.get_commits(owner, repo_name, since, until, author=author)
        except Exception as exc:
            console.print(f"[red]Error fetching commits:[/red] {exc}")
            sys.exit(1)

    with console.status("[bold green]Fetching issues…"):
        try:
            review_data.issues = gh.get_issues(owner, repo_name, since, until)
        except Exception as exc:
            console.print(f"[red]Error fetching issues:[/red] {exc}")
            sys.exit(1)

    with console.status("[bold green]Fetching pull requests…"):
        try:
            review_data.pull_requests = gh.get_pull_requests(owner, repo_name, since, until)
        except Exception as exc:
            console.print(f"[red]Error fetching pull requests:[/red] {exc}")
            sys.exit(1)

    # --- Print rich tables ------------------------------------------
    _print_header(owner, repo_name, since, until)
    _print_commits_table(review_data.commits)
    _print_issues_table(review_data.issues)
    _print_prs_table(review_data.pull_requests)

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

    with console.status("[bold green]Generating AI summary…"):
        try:
            llm = LLMClient(api_key=effective_key, model=model, base_url=base_url)
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


# ---------------------------------------------------------------------------
# Pretty-print helpers
# ---------------------------------------------------------------------------

def _print_header(owner: str, repo: str, since: datetime, until: datetime) -> None:
    console.print()
    console.print(
        Panel(
            f"[bold]{owner}/{repo}[/bold]  ·  "
            f"{since.date()} → {until.date()}",
            title="[bold blue]git-review[/bold blue]",
            border_style="blue",
        )
    )
    console.print()


def _print_commits_table(commits: list[Commit]) -> None:
    if not commits:
        console.print("[dim]No commits found in this time window.[/dim]\n")
        return

    table = Table(title=f"Commits ({len(commits)})", show_lines=False)
    table.add_column("SHA", style="dim", no_wrap=True, width=9)
    table.add_column("Date", no_wrap=True, width=12)
    table.add_column("Author", no_wrap=True)
    table.add_column("Message")

    for c in commits:
        table.add_row(
            c.sha[:7],
            str(c.authored_at.date()),
            c.author,
            c.message[:80] + ("…" if len(c.message) > 80 else ""),
        )
    console.print(table)
    console.print()


def _print_issues_table(issues: list[Issue]) -> None:
    if not issues:
        console.print("[dim]No issues found in this time window.[/dim]\n")
        return

    table = Table(title=f"Issues ({len(issues)})", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=6)
    table.add_column("State", width=8)
    table.add_column("Author", no_wrap=True)
    table.add_column("Title")
    table.add_column("Labels")

    state_styles = {"open": "green", "closed": "red"}
    for i in issues:
        style = state_styles.get(i.state, "white")
        table.add_row(
            str(i.number),
            f"[{style}]{i.state}[/{style}]",
            i.author,
            i.title[:80] + ("…" if len(i.title) > 80 else ""),
            ", ".join(i.labels) or "—",
        )
    console.print(table)
    console.print()


def _print_prs_table(prs: list[PullRequest]) -> None:
    if not prs:
        console.print("[dim]No pull requests found in this time window.[/dim]\n")
        return

    table = Table(title=f"Pull Requests ({len(prs)})", show_lines=False)
    table.add_column("#", justify="right", style="dim", width=6)
    table.add_column("State", width=8)
    table.add_column("Author", no_wrap=True)
    table.add_column("Title")
    table.add_column("Merged", width=12)

    for pr in prs:
        merged_str = str(pr.merged_at.date()) if pr.merged_at else "—"
        state_style = "green" if pr.state == "open" else ("magenta" if pr.merged_at else "red")
        table.add_row(
            str(pr.number),
            f"[{state_style}]{pr.state}[/{state_style}]",
            pr.author,
            pr.title[:80] + ("…" if len(pr.title) > 80 else ""),
            merged_str,
        )
    console.print(table)
    console.print()
