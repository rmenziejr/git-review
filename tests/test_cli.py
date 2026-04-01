"""Tests for the CLI (git_review.cli)."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from git_review.cli import main
from git_review.models import Commit, Issue, PullRequest, ReviewSummary

OWNER = "acme"
REPO = "app"


def _fake_commits():
    return [
        Commit(
            sha="abc1234567890",
            message="feat: new feature",
            author="Alice",
            authored_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
            url="https://github.com/acme/app/commit/abc1234567890",
            repo="acme/app",
            additions=10,
            deletions=3,
        )
    ]


def _fake_issues():
    return [
        Issue(
            number=42,
            title="Fix the bug",
            state="closed",
            author="bob",
            created_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 9, tzinfo=timezone.utc),
            url="https://github.com/acme/app/issues/42",
            repo="acme/app",
            labels=["bug"],
        )
    ]


def _fake_prs():
    return [
        PullRequest(
            number=7,
            title="Add feature X",
            state="closed",
            author="carol",
            created_at=datetime(2024, 1, 6, tzinfo=timezone.utc),
            merged_at=datetime(2024, 1, 8, tzinfo=timezone.utc),
            url="https://github.com/acme/app/pull/7",
            repo="acme/app",
        )
    ]


def _patch_github(commits=None, issues=None, prs=None, repos=None):
    mock_gh = MagicMock()
    mock_gh.return_value.get_commits.return_value = commits or []
    mock_gh.return_value.get_issues.return_value = issues or []
    mock_gh.return_value.get_pull_requests.return_value = prs or []
    mock_gh.return_value.list_repos.return_value = repos or []
    return patch("git_review.cli.GitHubClient", mock_gh)


# ---------------------------------------------------------------------------
# Happy-path tests
# ---------------------------------------------------------------------------

def test_review_with_days_flag() -> None:
    runner = CliRunner()
    with _patch_github(_fake_commits(), _fake_issues(), _fake_prs()):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "abc1234" in result.output
    assert "Fix the bug" in result.output
    assert "Add feature X" in result.output


def test_review_with_explicit_dates() -> None:
    runner = CliRunner()
    with _patch_github(_fake_commits(), [], []):
        result = runner.invoke(
            main,
            [
                "review",
                "--repo", "acme/app",
                "--since", "2024-01-01",
                "--until", "2024-01-31",
                "--no-summary",
            ],
        )
    assert result.exit_code == 0, result.output
    assert "2024-01-01" in result.output
    assert "2024-01-31" in result.output


def test_review_no_commits_shows_message() -> None:
    runner = CliRunner()
    with _patch_github([], [], []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "1", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "No commits" in result.output


# ---------------------------------------------------------------------------
# Error / validation tests
# ---------------------------------------------------------------------------

def test_review_bad_repo_format() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["review", "--repo", "not-valid"])
    assert result.exit_code != 0


def test_review_since_after_until() -> None:
    runner = CliRunner()
    with _patch_github():
        result = runner.invoke(
            main,
            [
                "review",
                "--repo", "acme/app",
                "--since", "2024-02-01",
                "--until", "2024-01-01",
            ],
        )
    assert result.exit_code != 0


def test_review_invalid_since_date() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["review", "--repo", "acme/app", "--since", "not-a-date"],
    )
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# LLM summary integration
# ---------------------------------------------------------------------------

def test_review_calls_llm_when_key_provided() -> None:
    runner = CliRunner()
    mock_llm = MagicMock()
    mock_llm.return_value.summarise.return_value = "## Highlights\nGreat month!"

    with _patch_github(_fake_commits(), _fake_issues(), _fake_prs()):
        with patch("git_review.cli.LLMClient", mock_llm):
            result = runner.invoke(
                main,
                [
                    "review",
                    "--repo", "acme/app",
                    "--days", "7",
                    "--openai-key", "sk-fake",
                ],
            )

    assert result.exit_code == 0, result.output
    assert mock_llm.return_value.summarise.called
    assert "Highlights" in result.output and "Great month" in result.output


def test_review_warns_when_no_openai_key() -> None:
    runner = CliRunner()
    with _patch_github(_fake_commits(), [], []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7"],
            env={"OPENAI_API_KEY": ""},
        )
    assert result.exit_code == 0
    assert "No OpenAI API key" in result.output


# ---------------------------------------------------------------------------
# --owner (all-repos mode) tests
# ---------------------------------------------------------------------------

def test_review_owner_lists_and_scans_all_repos() -> None:
    runner = CliRunner()
    commits_app = [
        Commit(
            sha="aaa0000000001",
            message="feat: app feature",
            author="Alice",
            authored_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
            url="https://github.com/acme/app/commit/aaa0000000001",
            repo="acme/app",
        )
    ]
    commits_api = [
        Commit(
            sha="bbb0000000002",
            message="fix: api bug",
            author="Bob",
            authored_at=datetime(2024, 1, 12, tzinfo=timezone.utc),
            url="https://github.com/acme/api/commit/bbb0000000002",
            repo="acme/api",
        )
    ]

    def side_effect_commits(owner, repo, since, until, author=None, include_stats=False):
        return commits_app if repo == "app" else commits_api

    mock_gh = MagicMock()
    mock_gh.return_value.list_repos.return_value = ["app", "api"]
    mock_gh.return_value.get_commits.side_effect = side_effect_commits
    mock_gh.return_value.get_issues.return_value = []
    mock_gh.return_value.get_pull_requests.return_value = []

    with patch("git_review.cli.GitHubClient", mock_gh):
        result = runner.invoke(
            main,
            ["review", "--owner", "acme", "--days", "7", "--no-summary"],
        )

    assert result.exit_code == 0, result.output
    # Both repos' commits appear
    assert "aaa0000" in result.output
    assert "bbb0000" in result.output
    # Header shows all-repos indicator
    assert "acme/*" in result.output
    # Repo column is shown
    assert "acme/app" in result.output
    assert "acme/api" in result.output
    # list_repos was called once
    mock_gh.return_value.list_repos.assert_called_once_with("acme")


def test_review_owner_no_repos_found() -> None:
    runner = CliRunner()
    with _patch_github(repos=[]):
        result = runner.invoke(
            main,
            ["review", "--owner", "acme", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0
    assert "No repositories" in result.output


def test_review_owner_and_repo_both_given_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["review", "--owner", "acme", "--repo", "acme/app", "--days", "7"],
    )
    assert result.exit_code != 0


def test_review_neither_owner_nor_repo_errors() -> None:
    runner = CliRunner()
    result = runner.invoke(main, ["review", "--days", "7"])
    assert result.exit_code != 0


# ---------------------------------------------------------------------------
# Stats (additions / deletions) tests
# ---------------------------------------------------------------------------

def test_review_shows_additions_deletions_in_commits_table() -> None:
    runner = CliRunner()
    with _patch_github(_fake_commits(), [], []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "+10" in result.output
    assert "-3" in result.output


def test_review_shows_repo_stats_table() -> None:
    runner = CliRunner()
    with _patch_github(_fake_commits(), [], []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Repo Stats" in result.output
    assert "acme/app" in result.output


def test_review_repo_stats_aggregates_per_repo() -> None:
    """In all-repos mode each repo's +/- totals are aggregated correctly."""
    runner = CliRunner()
    commits_app = [
        Commit(
            sha="aaa0000000001",
            message="feat: app feature",
            author="Alice",
            authored_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
            url="https://github.com/acme/app/commit/aaa0000000001",
            repo="acme/app",
            additions=50,
            deletions=5,
        )
    ]
    commits_api = [
        Commit(
            sha="bbb0000000002",
            message="fix: api bug",
            author="Bob",
            authored_at=datetime(2024, 1, 12, tzinfo=timezone.utc),
            url="https://github.com/acme/api/commit/bbb0000000002",
            repo="acme/api",
            additions=20,
            deletions=10,
        )
    ]

    def side_effect_commits(owner, repo, since, until, author=None, include_stats=False):
        return commits_app if repo == "app" else commits_api

    mock_gh = MagicMock()
    mock_gh.return_value.list_repos.return_value = ["app", "api"]
    mock_gh.return_value.get_commits.side_effect = side_effect_commits
    mock_gh.return_value.get_issues.return_value = []
    mock_gh.return_value.get_pull_requests.return_value = []

    with patch("git_review.cli.GitHubClient", mock_gh):
        result = runner.invoke(
            main,
            ["review", "--owner", "acme", "--days", "7", "--no-summary"],
        )

    assert result.exit_code == 0, result.output
    assert "Repo Stats" in result.output
    assert "+50" in result.output
    assert "-5" in result.output
    assert "+20" in result.output
    assert "-10" in result.output


def test_review_repo_stats_not_shown_when_no_commits() -> None:
    runner = CliRunner()
    with _patch_github([], [], []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Repo Stats" not in result.output

