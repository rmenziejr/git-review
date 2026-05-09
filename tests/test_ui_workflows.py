"""Tests for shared UI workflow helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from git_review.issue_factory import IssueDraft
from git_review.models import AgilePlanResult, Issue, IssueDependency, PullRequest, SprintRecommendation
from git_review.ui_workflows import (
    fetch_requirements_from_repo,
    parse_requirements,
    run_agile_planner,
    submit_issues,
)

_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _issue(number: int, title: str) -> Issue:
    return Issue(
        number=number,
        title=title,
        state="open",
        author="alice",
        created_at=_DT,
        closed_at=None,
        url=f"https://github.com/acme/app/issues/{number}",
        repo="acme/app",
    )


def test_fetch_requirements_from_repo_returns_content() -> None:
    mock_gh = MagicMock()
    mock_gh.get_file_content.return_value = "# Requirements"

    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        content, status = fetch_requirements_from_repo("ghp_test", "acme/app", "docs/requirements.md")

    assert content == "# Requirements"
    assert "Fetched 'docs/requirements.md' from acme/app" in status
    mock_gh.get_file_content.assert_called_once_with("acme", "app", "docs/requirements.md")


def test_parse_requirements_returns_editable_rows() -> None:
    draft = IssueDraft(
        title="Add dashboard",
        body="Build the dashboard",
        labels=["enhancement"],
        assignees=["alice"],
        milestone=3,
    )
    mock_factory = MagicMock()
    mock_factory.parse_requirements.return_value = [draft]

    with patch("git_review.ui_workflows._make_clients", return_value=(MagicMock(), mock_factory)):
        rows, status = parse_requirements(
            "ghp_test",
            "sk-test",
            "gpt-4o-mini",
            "",
            "## Requirements",
            None,
            False,
            "",
        )

    assert rows == [[1, "Add dashboard", "Build the dashboard", "enhancement", "alice", "3"]]
    assert "Extracted 1 issue draft(s)" in status


def test_submit_issues_pushes_rows_to_github() -> None:
    mock_factory = MagicMock()
    mock_factory.push_issues.return_value = [{"number": 12, "html_url": "https://github.com/acme/app/issues/12"}]

    with patch("git_review.ui_workflows.GitHubClient", return_value=MagicMock()), patch(
        "git_review.ui_workflows.IssueFactory",
        return_value=mock_factory,
    ):
        result = submit_issues(
            "ghp_test",
            "acme/app",
            "",
            [[1, "Add dashboard", "Build the dashboard", "enhancement", "alice", "3"]],
        )

    assert "Created 1 issue(s)" in result
    mock_factory.push_issues.assert_called_once()


def test_run_agile_planner_formats_dependency_and_plan_markdown() -> None:
    result = AgilePlanResult(
        owner="acme",
        repo="app",
        issues=[_issue(1, "Add dashboard"), _issue(2, "Ship API")],
        pull_requests=[
            PullRequest(
                number=10,
                title="Prep release",
                state="open",
                author="bob",
                created_at=_DT,
                merged_at=None,
                url="https://github.com/acme/app/pull/10",
                repo="acme/app",
            )
        ],
        dependencies=[
            IssueDependency(
                from_issue=2,
                to_issue=1,
                dep_type="blocked-by",
                confidence=0.8,
                reason="API depends on the dashboard data model.",
            )
        ],
        sprints=[
            SprintRecommendation(
                sprint_number=1,
                issues=[1],
                theme="Foundation",
                rationale="Start with the core dashboard work.",
                deferred=[2],
            )
        ],
        summary_text="Focus on foundational work first.",
    )
    mock_planner = MagicMock()
    mock_planner.analyse.return_value = result

    with patch("git_review.ui_workflows.AgilePlanner", return_value=mock_planner), patch(
        "git_review.ui_workflows.GitHubClient",
        return_value=MagicMock(),
    ):
        deps_md, plan_md, status = run_agile_planner(
            "ghp_test",
            "sk-test",
            "gpt-4o-mini",
            "",
            "acme/app",
            10,
            3,
            False,
        )

    assert "| #2 | #1 | 80% | llm | API depends on the dashboard data model. |" in deps_md
    assert "### Sprint 1: Foundation" in plan_md
    assert "- #1 Add dashboard" in plan_md
    assert "Focus on foundational work first." in plan_md
    assert "Plan generated: 2 issues, 1 PR, 1 dependencies, 1 sprints." in status


def test_agent_app_imports_with_multi_page_shell() -> None:
    from git_review.agent_app import agent_app

    assert agent_app.app is not None
