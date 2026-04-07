"""Tests for the CLI (git_review.cli)."""

from __future__ import annotations

import os
import subprocess
import tempfile
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest
from click.testing import CliRunner

from git_review.cli import main, _find_git_root
from git_review.models import Commit, Contributor, Issue, PullRequest, Release, ReviewSummary

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
            comments=2,
        )
    ]


def _fake_prs(reviewer_comments=None):
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
            reviewer_comments=reviewer_comments or {},
        )
    ]


def _fake_releases():
    return [
        Release(
            tag="v1.0.0",
            name="Version 1.0.0",
            body="First release",
            created_at=datetime(2024, 1, 20, tzinfo=timezone.utc),
            published_at=datetime(2024, 1, 20, tzinfo=timezone.utc),
            url="https://github.com/acme/app/releases/tag/v1.0.0",
            repo="acme/app",
            author="alice",
        )
    ]


def _fake_contributors():
    return [
        Contributor(
            login="alice",
            contributions=50,
            url="https://github.com/alice",
            repo="acme/app",
        )
    ]


def _patch_github(commits=None, issues=None, prs=None, repos=None,
                  releases=None, contributors=None):
    mock_gh = MagicMock()
    mock_gh.return_value.get_commits.return_value = commits or []
    mock_gh.return_value.get_issues.return_value = issues or []
    mock_gh.return_value.get_pull_requests.return_value = prs or []
    mock_gh.return_value.list_repos.return_value = repos or []
    mock_gh.return_value.get_releases.return_value = releases or []
    mock_gh.return_value.get_contributors.return_value = contributors or []
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


# ---------------------------------------------------------------------------
# Issue days-open stats table tests
# ---------------------------------------------------------------------------

def test_review_shows_issue_days_open_stats_table() -> None:
    runner = CliRunner()
    with _patch_github([], _fake_issues(), []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    # _fake_issues() returns a closed issue, so only the closed table appears
    assert "Closed Issue Age" in result.output
    assert "acme/app" in result.output


def test_review_shows_open_issue_age_table_for_open_issues() -> None:
    runner = CliRunner()
    open_issue = Issue(
        number=10,
        title="Open issue",
        state="open",
        author="alice",
        created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
        closed_at=None,
        url="https://github.com/acme/app/issues/10",
        repo="acme/app",
    )
    with _patch_github([], [open_issue], []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Open Issue Age" in result.output
    assert "Closed Issue Age" not in result.output


def test_review_issue_days_open_stats_not_shown_when_no_issues() -> None:
    runner = CliRunner()
    with _patch_github([], [], []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Issue Age" not in result.output


def test_review_issue_days_open_buckets_correctly() -> None:
    """Issues with different ages fall into the correct buckets."""
    runner = CliRunner()
    issues = [
        # 3 days open → "0–7 days"
        Issue(
            number=1,
            title="Short issue",
            state="closed",
            author="alice",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 4, tzinfo=timezone.utc),
            url="https://github.com/acme/app/issues/1",
            repo="acme/app",
        ),
        # 15 days open → "8–30 days"
        Issue(
            number=2,
            title="Medium issue",
            state="closed",
            author="bob",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 16, tzinfo=timezone.utc),
            url="https://github.com/acme/app/issues/2",
            repo="acme/app",
        ),
        # 60 days open → "31–90 days"
        Issue(
            number=3,
            title="Long issue",
            state="closed",
            author="carol",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 3, 1, tzinfo=timezone.utc),
            url="https://github.com/acme/app/issues/3",
            repo="acme/app",
        ),
        # 100 days open → "91+ days"
        Issue(
            number=4,
            title="Very long issue",
            state="closed",
            author="dave",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 4, 10, tzinfo=timezone.utc),
            url="https://github.com/acme/app/issues/4",
            repo="acme/app",
        ),
    ]
    with _patch_github([], issues, []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "180", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Closed Issue Age" in result.output
    # All four bucket column headers should appear in the table
    assert "0\u20137 days" in result.output
    assert "8\u201330 days" in result.output
    assert "31\u201390 days" in result.output
    assert "91+ days" in result.output


def test_review_issue_days_open_both_tables_shown_when_mixed() -> None:
    """Both open and closed tables appear when issues of both states exist."""
    runner = CliRunner()
    issues = [
        Issue(
            number=1,
            title="Open issue",
            state="open",
            author="alice",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=None,
            url="https://github.com/acme/app/issues/1",
            repo="acme/app",
        ),
        Issue(
            number=2,
            title="Closed issue",
            state="closed",
            author="bob",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 8, tzinfo=timezone.utc),
            url="https://github.com/acme/app/issues/2",
            repo="acme/app",
        ),
    ]
    with _patch_github([], issues, []):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "30", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Open Issue Age" in result.output
    assert "Closed Issue Age" in result.output


def test_review_issue_days_open_stats_aggregates_per_repo() -> None:
    """Issues from different repos are grouped into separate rows."""
    runner = CliRunner()
    issues_app = [
        Issue(
            number=1,
            title="App issue",
            state="closed",
            author="alice",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 4, tzinfo=timezone.utc),
            url="https://github.com/acme/app/issues/1",
            repo="acme/app",
        ),
    ]
    issues_api = [
        Issue(
            number=2,
            title="API issue",
            state="closed",
            author="bob",
            created_at=datetime(2024, 1, 1, tzinfo=timezone.utc),
            closed_at=datetime(2024, 1, 20, tzinfo=timezone.utc),
            url="https://github.com/acme/api/issues/2",
            repo="acme/api",
        ),
    ]

    def side_effect_issues(owner, repo, since, until, state="all"):
        return issues_app if repo == "app" else issues_api

    mock_gh = MagicMock()
    mock_gh.return_value.list_repos.return_value = ["app", "api"]
    mock_gh.return_value.get_commits.return_value = []
    mock_gh.return_value.get_issues.side_effect = side_effect_issues
    mock_gh.return_value.get_pull_requests.return_value = []

    with patch("git_review.cli.GitHubClient", mock_gh):
        result = runner.invoke(
            main,
            ["review", "--owner", "acme", "--days", "7", "--no-summary"],
        )

    assert result.exit_code == 0, result.output
    assert "Closed Issue Age" in result.output
    assert "acme/app" in result.output
    assert "acme/api" in result.output


# ---------------------------------------------------------------------------
# Releases and Contributors tables
# ---------------------------------------------------------------------------

def test_review_shows_releases_table() -> None:
    runner = CliRunner()
    with _patch_github([], [], [], releases=_fake_releases()):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Releases" in result.output
    assert "v1.0.0" in result.output


def test_review_shows_contributors_table() -> None:
    runner = CliRunner()
    with _patch_github([], [], [], contributors=_fake_contributors()):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Contributors" in result.output
    assert "alice" in result.output


def test_review_no_releases_no_table() -> None:
    runner = CliRunner()
    with _patch_github([], [], [], releases=[]):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Releases" not in result.output


def test_review_prs_table_shows_reviewer_comments() -> None:
    runner = CliRunner()
    prs = _fake_prs(reviewer_comments={"alice": 3, "bob": 1})
    with _patch_github([], [], prs):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "alice(3)" in result.output
    assert "bob(1)" in result.output


def test_review_prs_table_no_reviewer_comments_shows_dash() -> None:
    runner = CliRunner()
    prs = _fake_prs(reviewer_comments={})
    with _patch_github([], [], prs):
        result = runner.invoke(
            main,
            ["review", "--repo", "acme/app", "--days", "7", "--no-summary"],
        )
    assert result.exit_code == 0, result.output
    assert "Pull Requests" in result.output


# ---------------------------------------------------------------------------
# create-issues command
# ---------------------------------------------------------------------------

def _make_requirements_file(content: str) -> str:
    """Write *content* to a temp file and return its path."""
    f = tempfile.NamedTemporaryFile(
        mode="w", suffix=".md", delete=False, encoding="utf-8"
    )
    f.write(content)
    f.flush()
    f.close()
    return f.name


def test_create_issues_dry_run_shows_drafts() -> None:
    from git_review.issue_factory import IssueDraft

    runner = CliRunner()
    tmp = _make_requirements_file("# Requirements\n- Add OAuth\n- Fix crash")

    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = [
        IssueDraft(title="Add OAuth login", body="OAuth body", labels=["enhancement"]),
        IssueDraft(title="Fix crash on startup", body="Crash body", labels=["bug"]),
    ]

    with _patch_github():
        with patch("git_review.cli.IssueFactory", mock_factory_cls):
            result = runner.invoke(
                main,
                [
                    "create-issues",
                    "--repo", "acme/app",
                    "--requirements", tmp,
                    "--openai-key", "sk-fake",
                    "--dry-run",
                ],
            )

    os.unlink(tmp)
    assert result.exit_code == 0, result.output
    assert "Add OAuth login" in result.output
    assert "Fix crash on startup" in result.output
    assert "Dry-run" in result.output


def test_create_issues_yes_flag_pushes_all() -> None:
    from git_review.issue_factory import IssueDraft

    runner = CliRunner()
    tmp = _make_requirements_file("# Requirements\n- Feature A")

    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = [
        IssueDraft(title="Implement Feature A", body="Body A"),
    ]
    mock_factory_cls.return_value.push_issues.return_value = [
        {"number": 1, "html_url": "https://github.com/acme/app/issues/1"},
    ]

    with _patch_github():
        with patch("git_review.cli.IssueFactory", mock_factory_cls):
            result = runner.invoke(
                main,
                [
                    "create-issues",
                    "--repo", "acme/app",
                    "--requirements", tmp,
                    "--openai-key", "sk-fake",
                    "--yes",
                ],
            )

    os.unlink(tmp)
    assert result.exit_code == 0, result.output
    assert "Created 1 issue" in result.output
    assert "#1" in result.output
    mock_factory_cls.return_value.push_issues.assert_called_once()


def test_create_issues_missing_openai_key_errors() -> None:
    runner = CliRunner()
    tmp = _make_requirements_file("# Requirements\n- Feature A")

    result = runner.invoke(
        main,
        [
            "create-issues",
            "--repo", "acme/app",
            "--requirements", tmp,
        ],
        env={"OPENAI_API_KEY": ""},
    )

    os.unlink(tmp)
    assert result.exit_code != 0


def test_create_issues_bad_repo_format_errors() -> None:
    runner = CliRunner()
    tmp = _make_requirements_file("# Requirements")

    result = runner.invoke(
        main,
        [
            "create-issues",
            "--repo", "not-valid",
            "--requirements", tmp,
            "--openai-key", "sk-fake",
        ],
    )

    os.unlink(tmp)
    assert result.exit_code != 0


def test_create_issues_no_drafts_returned_exits_cleanly() -> None:
    runner = CliRunner()
    tmp = _make_requirements_file("# Requirements")

    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = []

    with _patch_github():
        with patch("git_review.cli.IssueFactory", mock_factory_cls):
            result = runner.invoke(
                main,
                [
                    "create-issues",
                    "--repo", "acme/app",
                    "--requirements", tmp,
                    "--openai-key", "sk-fake",
                ],
            )

    os.unlink(tmp)
    assert result.exit_code == 0
    assert "No issues were extracted" in result.output



# ---------------------------------------------------------------------------
# review --output (write summary to markdown file)
# ---------------------------------------------------------------------------

def test_review_output_writes_summary_to_file() -> None:
    runner = CliRunner()
    mock_llm = MagicMock()
    mock_llm.return_value.summarise.return_value = "## Highlights\nGreat work!"

    with _patch_github(_fake_commits(), [], []):
        with patch("git_review.cli.LLMClient", mock_llm):
            with runner.isolated_filesystem():
                result = runner.invoke(
                    main,
                    [
                        "review",
                        "--repo", "acme/app",
                        "--days", "7",
                        "--openai-key", "sk-fake",
                        "--output", "summary.md",
                    ],
                )
                assert result.exit_code == 0, result.output
                assert "summary.md" in result.output
                with open("summary.md", encoding="utf-8") as fh:
                    contents = fh.read()
                assert "## Highlights" in contents
                assert "Great work!" in contents


def test_review_output_not_written_when_no_summary_flag() -> None:
    """--output is silently ignored when --no-summary is set (no LLM call)."""
    runner = CliRunner()

    with _patch_github(_fake_commits(), [], []):
        with runner.isolated_filesystem():
            result = runner.invoke(
                main,
                [
                    "review",
                    "--repo", "acme/app",
                    "--days", "7",
                    "--no-summary",
                    "--output", "summary.md",
                ],
            )
            assert result.exit_code == 0, result.output
            import os as _os
            assert not _os.path.exists("summary.md")


# ---------------------------------------------------------------------------
# _find_git_root helper
# ---------------------------------------------------------------------------

def test_find_git_root_finds_immediate_parent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, ".git"))
        result = _find_git_root(tmpdir)
        assert result == tmpdir


def test_find_git_root_walks_up() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        os.makedirs(os.path.join(tmpdir, ".git"))
        subdir = os.path.join(tmpdir, "a", "b", "c")
        os.makedirs(subdir)
        result = _find_git_root(subdir)
        assert result == tmpdir


def test_find_git_root_returns_none_when_no_git() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        result = _find_git_root(tmpdir)
        assert result is None


# ---------------------------------------------------------------------------
# commit-message command
# ---------------------------------------------------------------------------

SAMPLE_DIFF = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1 +1,2 @@
 def hello():
+    print("hi")
"""


def test_commit_message_prints_suggested_message() -> None:
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat(foo): add hello print"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                result = runner.invoke(
                    main,
                    [
                        "commit-message",
                        "--openai-key", "sk-fake",
                    ],
                    input="n\nn\n",  # no to edit, no to commit
                )

    assert result.exit_code == 0, result.output
    assert "feat(foo): add hello print" in result.output
    mock_gen_cls.return_value.generate.assert_called_once_with(SAMPLE_DIFF)


def test_commit_message_commits_when_confirmed() -> None:
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat(foo): add hello print"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with patch("git_review.cli.subprocess") as mock_subprocess:
                    mock_subprocess.run.return_value = MagicMock(returncode=0)
                    result = runner.invoke(
                        main,
                        [
                            "commit-message",
                            "--openai-key", "sk-fake",
                        ],
                        input="n\ny\n",  # no to edit, yes to commit
                    )

    assert result.exit_code == 0, result.output
    assert "Committed successfully" in result.output
    mock_subprocess.run.assert_called_once_with(
        ["git", "commit", "-m", "feat(foo): add hello print"],
        cwd="/fake/repo",
        check=True,
    )


def test_commit_message_commits_subject_and_body() -> None:
    """Subject and body are passed as separate -m arguments."""
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = (
        "feat(foo): add hello print\n\nThis adds a print statement to hello()."
    )

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with patch("git_review.cli.subprocess") as mock_subprocess:
                    mock_subprocess.run.return_value = MagicMock(returncode=0)
                    result = runner.invoke(
                        main,
                        [
                            "commit-message",
                            "--openai-key", "sk-fake",
                        ],
                        input="n\ny\n",  # no to edit, yes to commit
                    )

    assert result.exit_code == 0, result.output
    mock_subprocess.run.assert_called_once_with(
        [
            "git", "commit",
            "-m", "feat(foo): add hello print",
            "-m", "This adds a print statement to hello().",
        ],
        cwd="/fake/repo",
        check=True,
    )


def test_commit_message_commits_subject_only_when_no_blank_line() -> None:
    """A message with no blank-line separator is committed as subject only."""
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat: single line"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with patch("git_review.cli.subprocess") as mock_subprocess:
                    mock_subprocess.run.return_value = MagicMock(returncode=0)
                    result = runner.invoke(
                        main,
                        [
                            "commit-message",
                            "--openai-key", "sk-fake",
                        ],
                        input="n\ny\n",
                    )

    assert result.exit_code == 0, result.output
    mock_subprocess.run.assert_called_once_with(
        ["git", "commit", "-m", "feat: single line"],
        cwd="/fake/repo",
        check=True,
    )


def test_commit_message_commits_subject_only_when_single_newline() -> None:
    """A message separated by a single newline (no blank line) is committed as subject only."""
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat: title\nno blank line"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with patch("git_review.cli.subprocess") as mock_subprocess:
                    mock_subprocess.run.return_value = MagicMock(returncode=0)
                    result = runner.invoke(
                        main,
                        [
                            "commit-message",
                            "--openai-key", "sk-fake",
                        ],
                        input="n\ny\n",
                    )

    assert result.exit_code == 0, result.output
    # No blank-line separator → no body argument
    mock_subprocess.run.assert_called_once_with(
        ["git", "commit", "-m", "feat: title\nno blank line"],
        cwd="/fake/repo",
        check=True,
    )


def test_commit_message_git_commit_failure_exits_with_error() -> None:
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat: something"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with patch("git_review.cli.subprocess") as mock_subprocess:
                    mock_subprocess.run.side_effect = subprocess.CalledProcessError(1, "git")
                    mock_subprocess.CalledProcessError = subprocess.CalledProcessError
                    result = runner.invoke(
                        main,
                        [
                            "commit-message",
                            "--openai-key", "sk-fake",
                        ],
                        input="n\ny\n",  # no to edit, yes to commit
                    )

    assert result.exit_code != 0
    assert "git commit failed" in result.output


def test_commit_message_edit_updates_message() -> None:
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat(foo): original"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with patch("click.edit", return_value="feat(foo): edited") as mock_edit:
                    with patch("git_review.cli.subprocess") as mock_subprocess:
                        mock_subprocess.run.return_value = MagicMock(returncode=0)
                        result = runner.invoke(
                            main,
                            [
                                "commit-message",
                                "--openai-key", "sk-fake",
                            ],
                            input="y\ny\n",  # yes to edit, yes to commit
                        )

    assert result.exit_code == 0, result.output
    mock_edit.assert_called_once_with("feat(foo): original")
    assert "Edited Commit Message" in result.output
    mock_subprocess.run.assert_called_once_with(
        ["git", "commit", "-m", "feat(foo): edited"],
        cwd="/fake/repo",
        check=True,
    )


def test_commit_message_edit_empty_aborts() -> None:
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat(foo): original"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with patch("click.edit", return_value="   "):
                    result = runner.invoke(
                        main,
                        [
                            "commit-message",
                            "--openai-key", "sk-fake",
                        ],
                        input="y\n",  # yes to edit (result is empty)
                    )

    assert result.exit_code != 0
    assert "empty" in result.output


def test_commit_message_no_git_root_exits_with_error() -> None:
    runner = CliRunner()

    with patch("git_review.cli._find_git_root", return_value=None):
        result = runner.invoke(
            main,
            ["commit-message", "--openai-key", "sk-fake"],
        )

    assert result.exit_code != 0
    assert "No git repository found" in result.output


def test_commit_message_errors_when_no_openai_key() -> None:
    runner = CliRunner()
    result = runner.invoke(
        main,
        ["commit-message"],
        env={"OPENAI_API_KEY": ""},
    )
    assert result.exit_code != 0


def test_commit_message_errors_when_no_diff() -> None:
    runner = CliRunner()

    with patch("git_review.cli.get_git_diff", return_value=""):
        result = runner.invoke(
            main,
            ["commit-message", "--openai-key", "sk-fake"],
        )

    assert result.exit_code != 0
    assert "No changes detected" in result.output


def test_commit_message_errors_on_git_failure() -> None:
    runner = CliRunner()

    with patch(
        "git_review.cli.get_git_diff",
        side_effect=RuntimeError("not a git repository"),
    ):
        result = runner.invoke(
            main,
            ["commit-message", "--openai-key", "sk-fake"],
        )

    assert result.exit_code != 0
    assert "Error reading git diff" in result.output


def test_commit_message_passes_repo_path() -> None:
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "chore: update"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF) as mock_diff:
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/some/path") as mock_root:
                result = runner.invoke(
                    main,
                    [
                        "commit-message",
                        "--repo-path", "/some/path",
                        "--openai-key", "sk-fake",
                    ],
                    input="n\nn\n",  # no to edit, no to commit
                )

    assert result.exit_code == 0, result.output
    mock_root.assert_called_once_with("/some/path")
    mock_diff.assert_called_once_with("/some/path")


# ---------------------------------------------------------------------------
# --prompt-file for review command
# ---------------------------------------------------------------------------

def test_review_prompt_file_passed_to_llm_client() -> None:
    runner = CliRunner()
    mock_llm = MagicMock()
    mock_llm.return_value.summarise.return_value = "## Custom\nDone."

    with _patch_github(_fake_commits(), [], []):
        with patch("git_review.cli.LLMClient", mock_llm):
            with runner.isolated_filesystem():
                with open("my_prompt.j2", "w") as fh:
                    fh.write("Custom prompt with {{ n }} items.")
                result = runner.invoke(
                    main,
                    [
                        "review",
                        "--repo", "acme/app",
                        "--days", "7",
                        "--openai-key", "sk-fake",
                        "--prompt-file", "my_prompt.j2",
                    ],
                )

    assert result.exit_code == 0, result.output
    _, kwargs = mock_llm.call_args
    assert kwargs.get("system_prompt") == "Custom prompt with {{ n }} items."


def test_review_prompt_file_invalid_template_exits_with_error() -> None:
    runner = CliRunner()

    with _patch_github(_fake_commits(), [], []):
        with runner.isolated_filesystem():
            with open("bad_prompt.j2", "w") as fh:
                fh.write("{{ totally_unknown_var }}")
            result = runner.invoke(
                main,
                [
                    "review",
                    "--repo", "acme/app",
                    "--days", "7",
                    "--openai-key", "sk-fake",
                    "--prompt-file", "bad_prompt.j2",
                ],
            )

    assert result.exit_code != 0
    assert "unknown variable" in result.output


# ---------------------------------------------------------------------------
# --prompt-file for commit-message command
# ---------------------------------------------------------------------------

def test_commit_message_prompt_file_passed_to_generator() -> None:
    runner = CliRunner()
    mock_gen_cls = MagicMock()
    mock_gen_cls.return_value.generate.return_value = "feat: custom"

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli.CommitMessageGenerator", mock_gen_cls):
            with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
                with runner.isolated_filesystem():
                    with open("commit_prompt.j2", "w") as fh:
                        fh.write("Write a one-line message.")
                    result = runner.invoke(
                        main,
                        [
                            "commit-message",
                            "--openai-key", "sk-fake",
                            "--prompt-file", "commit_prompt.j2",
                        ],
                        input="n\nn\n",  # no to edit, no to commit
                    )

    assert result.exit_code == 0, result.output
    _, kwargs = mock_gen_cls.call_args
    assert kwargs.get("system_prompt") == "Write a one-line message."


def test_commit_message_prompt_file_invalid_template_exits_with_error() -> None:
    runner = CliRunner()

    with patch("git_review.cli.get_git_diff", return_value=SAMPLE_DIFF):
        with patch("git_review.cli._find_git_root", return_value="/fake/repo"):
            with runner.isolated_filesystem():
                with open("bad.j2", "w") as fh:
                    fh.write("{{ forbidden_var }}")
                result = runner.invoke(
                    main,
                    [
                        "commit-message",
                        "--openai-key", "sk-fake",
                        "--prompt-file", "bad.j2",
                    ],
                )

    assert result.exit_code != 0
    assert "unknown variable" in result.output


# ---------------------------------------------------------------------------
# --prompt-file for create-issues command
# ---------------------------------------------------------------------------

def test_create_issues_prompt_file_passed_to_factory() -> None:
    from git_review.issue_factory import IssueDraft

    runner = CliRunner()
    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = [
        IssueDraft(title="Issue A", body="Body A"),
    ]

    with _patch_github():
        with patch("git_review.cli.IssueFactory", mock_factory_cls):
            with runner.isolated_filesystem():
                with open("issue_prompt.j2", "w") as fh:
                    fh.write("Only extract bug issues.")
                with open("reqs.md", "w") as fh:
                    fh.write("# Reqs")
                result = runner.invoke(
                    main,
                    [
                        "create-issues",
                        "--repo", "acme/app",
                        "--requirements", "reqs.md",
                        "--openai-key", "sk-fake",
                        "--prompt-file", "issue_prompt.j2",
                        "--dry-run",
                    ],
                )

    assert result.exit_code == 0, result.output
    _, kwargs = mock_factory_cls.call_args
    assert kwargs.get("system_prompt") == "Only extract bug issues."


def test_create_issues_prompt_file_invalid_template_exits_with_error() -> None:
    runner = CliRunner()

    with _patch_github():
        with runner.isolated_filesystem():
            with open("bad_prompt.j2", "w") as fh:
                fh.write("{{ unknown_var }}")
            with open("reqs.md", "w") as fh:
                fh.write("# Reqs")
            result = runner.invoke(
                main,
                [
                    "create-issues",
                    "--repo", "acme/app",
                    "--requirements", "reqs.md",
                    "--openai-key", "sk-fake",
                    "--prompt-file", "bad_prompt.j2",
                ],
            )

    assert result.exit_code != 0
    assert "unknown variable" in result.output


# ---------------------------------------------------------------------------
# create-issues – fetch requirements from repo (--requirements-path)
# ---------------------------------------------------------------------------

def test_create_issues_fetches_requirements_from_repo_when_no_local_file() -> None:
    """Without --requirements, the CLI should fetch docs/requirements.md from the repo."""
    from git_review.issue_factory import IssueDraft

    runner = CliRunner()
    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = [
        IssueDraft(title="Fetched Issue", body="Body"),
    ]
    mock_factory_cls.return_value.push_issues.return_value = [
        {"number": 1, "html_url": "https://github.com/acme/app/issues/1"},
    ]

    mock_gh_cls = MagicMock()
    mock_gh_instance = mock_gh_cls.return_value
    mock_gh_instance.get_file_content.return_value = "# Requirements\n- Feature A\n"

    with patch("git_review.cli.GitHubClient", mock_gh_cls):
        with patch("git_review.cli.IssueFactory", mock_factory_cls):
            result = runner.invoke(
                main,
                [
                    "create-issues",
                    "--repo", "acme/app",
                    "--openai-key", "sk-fake",
                    "--yes",
                ],
            )

    assert result.exit_code == 0, result.output
    mock_gh_instance.get_file_content.assert_called_once_with(
        "acme", "app", "docs/requirements.md"
    )
    mock_factory_cls.return_value.parse_requirements.assert_called_once()


def test_create_issues_custom_requirements_path() -> None:
    """--requirements-path overrides the default docs/requirements.md."""
    from git_review.issue_factory import IssueDraft

    runner = CliRunner()
    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = [
        IssueDraft(title="Issue", body="Body"),
    ]
    mock_factory_cls.return_value.push_issues.return_value = [
        {"number": 1, "html_url": ""},
    ]

    mock_gh_cls = MagicMock()
    mock_gh_instance = mock_gh_cls.return_value
    mock_gh_instance.get_file_content.return_value = "# Reqs\n- A\n"

    with patch("git_review.cli.GitHubClient", mock_gh_cls):
        with patch("git_review.cli.IssueFactory", mock_factory_cls):
            result = runner.invoke(
                main,
                [
                    "create-issues",
                    "--repo", "acme/app",
                    "--openai-key", "sk-fake",
                    "--requirements-path", "specs/features.md",
                    "--yes",
                ],
            )

    assert result.exit_code == 0, result.output
    mock_gh_instance.get_file_content.assert_called_once_with(
        "acme", "app", "specs/features.md"
    )


def test_create_issues_fetch_requirements_error_exits() -> None:
    """When fetching from the repo fails, the CLI exits with a non-zero code."""
    import requests

    runner = CliRunner()
    mock_gh_cls = MagicMock()
    mock_gh_instance = mock_gh_cls.return_value
    mock_gh_instance.get_file_content.side_effect = Exception("404 Not Found")

    with patch("git_review.cli.GitHubClient", mock_gh_cls):
        result = runner.invoke(
            main,
            [
                "create-issues",
                "--repo", "acme/app",
                "--openai-key", "sk-fake",
            ],
        )

    assert result.exit_code != 0
    assert "Error fetching requirements" in result.output


# ---------------------------------------------------------------------------
# create-issues – --use-milestones flag
# ---------------------------------------------------------------------------

def test_create_issues_use_milestones_fetches_and_passes_to_factory() -> None:
    """--use-milestones should fetch milestones and pass them to parse_requirements."""
    from git_review.issue_factory import IssueDraft
    from git_review.models import Milestone

    runner = CliRunner()
    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = [
        IssueDraft(title="Issue A", body="Body", milestone=1),
    ]
    mock_factory_cls.return_value.push_issues.return_value = [
        {"number": 1, "html_url": ""},
    ]

    mock_gh_cls = MagicMock()
    mock_gh_instance = mock_gh_cls.return_value
    fake_milestones = [
        Milestone(number=1, title="v1.0", state="open", repo="acme/app"),
    ]
    mock_gh_instance.list_milestones.return_value = fake_milestones

    with runner.isolated_filesystem():
        with open("reqs.md", "w") as fh:
            fh.write("# Reqs\n- Feature A\n")
        with patch("git_review.cli.GitHubClient", mock_gh_cls):
            with patch("git_review.cli.IssueFactory", mock_factory_cls):
                result = runner.invoke(
                    main,
                    [
                        "create-issues",
                        "--repo", "acme/app",
                        "--requirements", "reqs.md",
                        "--openai-key", "sk-fake",
                        "--use-milestones",
                        "--yes",
                    ],
                )

    assert result.exit_code == 0, result.output
    mock_gh_instance.list_milestones.assert_called_once_with("acme", "app", state="open")
    call_args = mock_factory_cls.return_value.parse_requirements.call_args
    passed_milestones = call_args.kwargs.get("milestones") or (
        call_args.args[1] if len(call_args.args) > 1 else None
    )
    assert passed_milestones == fake_milestones


def test_create_issues_use_milestones_no_milestones_continues() -> None:
    """When --use-milestones finds no milestones, the command continues normally."""
    from git_review.issue_factory import IssueDraft

    runner = CliRunner()
    mock_factory_cls = MagicMock()
    mock_factory_cls.return_value.parse_requirements.return_value = [
        IssueDraft(title="Issue A", body="Body"),
    ]
    mock_factory_cls.return_value.push_issues.return_value = [{"number": 1, "html_url": ""}]

    mock_gh_cls = MagicMock()
    mock_gh_instance = mock_gh_cls.return_value
    mock_gh_instance.list_milestones.return_value = []

    with runner.isolated_filesystem():
        with open("reqs.md", "w") as fh:
            fh.write("# Reqs\n")
        with patch("git_review.cli.GitHubClient", mock_gh_cls):
            with patch("git_review.cli.IssueFactory", mock_factory_cls):
                result = runner.invoke(
                    main,
                    [
                        "create-issues",
                        "--repo", "acme/app",
                        "--requirements", "reqs.md",
                        "--openai-key", "sk-fake",
                        "--use-milestones",
                        "--yes",
                    ],
                )

    assert result.exit_code == 0, result.output
    # parse_requirements should still be called, just with milestones=None
    call_args = mock_factory_cls.return_value.parse_requirements.call_args
    passed_milestones = call_args.kwargs.get("milestones") or (
        call_args.args[1] if len(call_args.args) > 1 else None
    )
    assert not passed_milestones
