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


def _patch_github(commits=None, issues=None, prs=None):
    mock_gh = MagicMock()
    mock_gh.return_value.get_commits.return_value = commits or []
    mock_gh.return_value.get_issues.return_value = issues or []
    mock_gh.return_value.get_pull_requests.return_value = prs or []
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
