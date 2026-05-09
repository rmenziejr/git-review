"""Tests for shared UI workflow helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

from git_review.issue_factory import IssueDraft
from git_review.models import AgilePlanResult, Issue, IssueDependency, PullRequest, SprintRecommendation
from git_review.ui_workflows import (
    append_milestone_to_batch,
    create_project_for_owner,
    create_milestones_batch,
    fetch_requirements_from_repo,
    list_open_issues_for_repo,
    list_projects_for_target,
    list_repositories_for_owner,
    load_default_milestones_text,
    parse_requirements,
    read_agile_project_board,
    run_agile_planner,
    submit_issues,
    update_agile_project_status,
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


def test_load_default_milestones_text_formats_queue() -> None:
    queue_text, status = load_default_milestones_text(
        '[{"title":"Backlog","description":"Shared roadmap","due_on":"2026-06-01","state":"open"}]'
    )

    assert queue_text == "Backlog | 2026-06-01 | open | Shared roadmap"
    assert "Loaded 1 default milestone(s)" in status


def test_append_milestone_to_batch_adds_line() -> None:
    queue_text, status = append_milestone_to_batch("", "Backlog", "Shared roadmap", "2026-06-01", "open")

    assert queue_text == "Backlog | 2026-06-01 | open | Shared roadmap"
    assert "Added 'Backlog'" in status


def test_create_milestones_batch_creates_multiple_entries() -> None:
    mock_gh = MagicMock()
    mock_gh.create_milestone.side_effect = [
        {"number": 1, "title": "Backlog", "html_url": "https://github.com/acme/app/milestone/1"},
        {"number": 2, "title": "MVP", "html_url": "https://github.com/acme/app/milestone/2"},
    ]

    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        status, created = create_milestones_batch(
            "ghp_test",
            "acme/app",
            "Backlog | 2026-06-01 | open | Shared roadmap\nMVP | 2026-07-01 | open | First release",
        )

    assert len(created) == 2
    assert "Created 2 of 2 milestone(s) for acme/app." in status
    assert mock_gh.create_milestone.call_count == 2


def test_create_milestones_batch_accepts_three_part_state_shortcut() -> None:
    mock_gh = MagicMock()
    mock_gh.create_milestone.return_value = {
        "number": 1,
        "title": "Backlog",
        "html_url": "https://github.com/acme/app/milestone/1",
    }

    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        status, created = create_milestones_batch(
            "ghp_test",
            "acme/app",
            "Backlog | 2026-06-01 | closed",
        )

    assert len(created) == 1
    assert "Created 1 of 1 milestone(s) for acme/app." in status
    assert mock_gh.create_milestone.call_args.kwargs["state"] == "closed"


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
    assert "Plan generated: 2 issues, 1 PR, 1 dependency, 1 sprint." in status


def test_agent_app_imports_with_multi_page_shell() -> None:
    from git_review.agent_app import agent_app

    assert agent_app.app is not None


def test_read_agile_project_board_formats_markdown() -> None:
    result = AgilePlanResult(
        owner="acme",
        repo="app",
        issues=[_issue(1, "A"), _issue(2, "B")],
        sprints=[SprintRecommendation(sprint_number=1, issues=[1, 2])],
    )
    mock_gh = MagicMock()
    mock_gh.read_project_status_board.return_value = {
        "project_title": "Sprint Board",
        "items": [
            {"number": 1, "type": "Issue", "status": "In Progress", "title": "A"},
            {"number": 2, "type": "Issue", "status": "Todo", "title": "B"},
        ],
    }
    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        markdown, status = read_agile_project_board(
            "ghp_test",
            "acme/app",
            3,
            "Status",
            1,
            result,
        )
    assert "| #1 | Issue | In Progress | A |" in markdown
    assert "Loaded 2 item(s) from project #3" in status
    mock_gh.read_project_status_board.assert_called_once()


def test_update_agile_project_status_returns_success() -> None:
    mock_gh = MagicMock()
    mock_gh.update_project_item_status.return_value = {
        "project_number": 3,
        "issue_number": 12,
        "status": "Done",
    }
    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        status = update_agile_project_status(
            "ghp_test",
            "acme/app",
            3,
            12,
            "Done",
            "Status",
        )
    assert "Updated #12 in project #3 to 'Done'" in status


def test_list_repositories_for_owner_returns_markdown() -> None:
    mock_gh = MagicMock()
    mock_gh.list_repos.return_value = ["app", "platform"]
    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        markdown, status = list_repositories_for_owner("ghp_test", "acme/app")

    assert "- acme/app" in markdown
    assert "- acme/platform" in markdown
    assert "Found 2 repository(ies) for 'acme'" in status


def test_list_projects_for_target_returns_table() -> None:
    mock_gh = MagicMock()
    mock_gh.list_projects.return_value = [
        {"number": 2, "title": "Sprint 2", "closed": False, "url": "https://example/2"},
        {"number": 3, "title": "Sprint 3", "closed": True, "url": "https://example/3"},
    ]
    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        markdown, status = list_projects_for_target("ghp_test", "acme/app")

    assert "| 2 | Sprint 2 | open | https://example/2 |" in markdown
    assert "| 3 | Sprint 3 | closed | https://example/3 |" in markdown
    assert "Found 2 project(s) for acme/app" in status


def test_create_project_for_owner_calls_client() -> None:
    mock_gh = MagicMock()
    mock_gh.create_project.return_value = {"number": 7, "title": "Sprint Board", "url": "https://example/7"}
    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        status = create_project_for_owner("ghp_test", "acme", "Sprint Board")

    assert "Created project #7 'Sprint Board' for acme" in status
    mock_gh.create_project.assert_called_once_with("acme", "Sprint Board")


def test_list_open_issues_for_repo_returns_table() -> None:
    mock_gh = MagicMock()
    mock_gh.get_open_issues.return_value = [_issue(1, "A"), _issue(2, "B")]
    with patch("git_review.ui_workflows.GitHubClient", return_value=mock_gh):
        markdown, status = list_open_issues_for_repo("ghp_test", "acme/app")

    assert "| #1 | A |" in markdown
    assert "| #2 | B |" in markdown
    assert "Found 2 open issue(s) in acme/app" in status
