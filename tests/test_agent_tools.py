"""Unit tests for git_review.agent_tools.

Covers:
- Read-only tools return correctly shaped data.
- Write tools carry needs_approval=True on their FunctionTool descriptor.
- The AgentContext dataclass stores credentials correctly.
- GitHubClient new write methods are called via the tool wrappers.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from agents.tool_context import ToolContext

from git_review.agent_tools import (
    AgentContext,
    ALL_TOOLS,
    agile_plan,
    create_draft_pr,
    create_issue_draft,
    get_issue,
    list_pull_requests,
    list_repos,
    push_issue_draft,
    ready_pr_for_review,
    search_issues,
    update_issue,
    update_pull_request,
)
from git_review.models import Issue, PullRequest

_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_issue(number: int, title: str = "An issue") -> Issue:
    return Issue(
        number=number,
        title=title,
        state="open",
        author="alice",
        created_at=_DT,
        closed_at=None,
        url=f"https://github.com/a/b/issues/{number}",
        repo="a/b",
        labels=["bug"],
        body="body",
        comments=0,
        assignees=[],
        github_id=1000 + number,
    )


def _make_pr(number: int, title: str = "A PR", draft: bool = False) -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        state="open",
        author="bob",
        created_at=_DT,
        merged_at=None,
        url=f"https://github.com/a/b/pull/{number}",
        repo="a/b",
        draft=draft,
        base_branch="main",
        head_branch="feat/x",
    )


def _make_agent_ctx() -> AgentContext:
    return AgentContext(
        owner="myorg",
        repo="myrepo",
        github_token="ghp_test",
        openai_api_key="sk-test",
        openai_base_url="",
        model="gpt-4o-mini",
    )


def _make_tool_ctx(agent_ctx: AgentContext, tool_name: str, args_json: str) -> ToolContext:
    """Create a minimal ToolContext suitable for unit-testing tool functions."""
    return ToolContext(
        context=agent_ctx,
        tool_name=tool_name,
        tool_call_id="call_test_123",
        tool_arguments=args_json,
    )


# ---------------------------------------------------------------------------
# AgentContext
# ---------------------------------------------------------------------------


def test_agent_context_stores_fields() -> None:
    ctx = AgentContext(
        owner="org",
        repo="repo",
        github_token="ghp_abc",
        openai_api_key="sk-abc",
        openai_base_url="http://localhost:11434/v1",
        model="gpt-4o",
    )
    assert ctx.owner == "org"
    assert ctx.repo == "repo"
    assert ctx.github_token == "ghp_abc"
    assert ctx.openai_api_key == "sk-abc"
    assert ctx.openai_base_url == "http://localhost:11434/v1"
    assert ctx.model == "gpt-4o"


def test_agent_context_defaults() -> None:
    ctx = AgentContext(
        owner="org",
        repo="repo",
        github_token="ghp_abc",
        openai_api_key="sk-abc",
    )
    assert ctx.openai_base_url == ""
    assert ctx.model == "gpt-4o"


# ---------------------------------------------------------------------------
# Write tools have needs_approval=True
# ---------------------------------------------------------------------------


WRITE_TOOLS = [
    push_issue_draft,
    update_issue,
    create_draft_pr,
    update_pull_request,
    ready_pr_for_review,
]

READ_TOOLS = [
    list_repos,
    search_issues,
    get_issue,
    list_pull_requests,
    create_issue_draft,
    agile_plan,
]


@pytest.mark.parametrize("tool", WRITE_TOOLS)
def test_write_tool_needs_approval(tool) -> None:
    """Every write tool must declare needs_approval=True."""
    assert tool.needs_approval is True, (
        f"{tool.name} must have needs_approval=True"
    )


@pytest.mark.parametrize("tool", READ_TOOLS)
def test_read_tool_no_approval(tool) -> None:
    """Read-only tools must not require approval."""
    assert tool.needs_approval is False, (
        f"{tool.name} must have needs_approval=False"
    )


def test_all_tools_list_complete() -> None:
    """ALL_TOOLS contains exactly the expected tools."""
    all_names = {t.name for t in ALL_TOOLS}
    expected = {
        "list_repos",
        "search_issues",
        "get_issue",
        "list_pull_requests",
        "create_issue_draft",
        "agile_plan",
        "push_issue_draft",
        "update_issue",
        "create_draft_pr",
        "update_pull_request",
        "ready_pr_for_review",
    }
    assert all_names == expected


# ---------------------------------------------------------------------------
# Read tool: list_repos
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_repos_returns_json() -> None:
    agent_ctx = _make_agent_ctx()
    args = '{"owner": "myorg"}'
    tc = _make_tool_ctx(agent_ctx, "list_repos", args)
    mock_gh = MagicMock()
    mock_gh.list_repos.return_value = ["repo-a", "repo-b"]

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await list_repos.on_invoke_tool(tc, args)

    repos = json.loads(result)
    assert repos == ["repo-a", "repo-b"]
    mock_gh.list_repos.assert_called_once_with("myorg")


# ---------------------------------------------------------------------------
# Read tool: search_issues
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_search_issues_returns_json() -> None:
    agent_ctx = _make_agent_ctx()
    args = '{"owner": "myorg", "repo": "myrepo"}'
    tc = _make_tool_ctx(agent_ctx, "search_issues", args)
    mock_gh = MagicMock()
    mock_gh.get_open_issues.return_value = [
        _make_issue(1, "Bug: crash"),
        _make_issue(2, "Feature: dark mode"),
    ]

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await search_issues.on_invoke_tool(tc, args)

    issues = json.loads(result)
    assert len(issues) == 2
    assert issues[0]["number"] == 1
    assert issues[0]["title"] == "Bug: crash"
    assert issues[1]["labels"] == ["bug"]


# ---------------------------------------------------------------------------
# Read tool: get_issue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_get_issue_returns_full_details() -> None:
    agent_ctx = _make_agent_ctx()
    args = '{"owner": "myorg", "repo": "myrepo", "issue_number": 42}'
    tc = _make_tool_ctx(agent_ctx, "get_issue", args)
    mock_gh = MagicMock()
    mock_gh.get_issue.return_value = _make_issue(42, "Full details issue")

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await get_issue.on_invoke_tool(tc, args)

    data = json.loads(result)
    assert data["number"] == 42
    assert data["title"] == "Full details issue"
    assert data["body"] == "body"
    mock_gh.get_issue.assert_called_once_with("myorg", "myrepo", 42)


# ---------------------------------------------------------------------------
# Read tool: list_pull_requests
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_list_pull_requests_returns_json() -> None:
    agent_ctx = _make_agent_ctx()
    args = '{"owner": "myorg", "repo": "myrepo"}'
    tc = _make_tool_ctx(agent_ctx, "list_pull_requests", args)
    mock_gh = MagicMock()
    mock_gh.get_open_pull_requests.return_value = [
        _make_pr(10, "feat: new endpoint"),
        _make_pr(11, "fix: null pointer", draft=True),
    ]

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await list_pull_requests.on_invoke_tool(tc, args)

    prs = json.loads(result)
    assert len(prs) == 2
    assert prs[1]["draft"] is True


# ---------------------------------------------------------------------------
# Read tool: create_issue_draft
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_issue_draft_returns_drafts() -> None:
    from git_review.issue_factory import IssueDraft

    agent_ctx = _make_agent_ctx()
    args = '{"requirements": "Add OAuth login. Fix startup crash."}'
    tc = _make_tool_ctx(agent_ctx, "create_issue_draft", args)
    mock_factory = MagicMock()
    mock_factory.parse_requirements.return_value = [
        IssueDraft(title="Add OAuth", body="Implement OAuth", labels=["enhancement"]),
        IssueDraft(title="Fix crash", body="App crashes on start", labels=["bug"]),
    ]

    with patch("git_review.agent_tools.IssueFactory", return_value=mock_factory), \
         patch("git_review.agent_tools.GitHubClient", return_value=MagicMock()):
        result = await create_issue_draft.on_invoke_tool(tc, args)

    drafts = json.loads(result)
    assert len(drafts) == 2
    assert drafts[0]["title"] == "Add OAuth"
    assert drafts[1]["labels"] == ["bug"]


# ---------------------------------------------------------------------------
# Write tool: push_issue_draft
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_push_issue_draft_calls_create_issue() -> None:
    agent_ctx = _make_agent_ctx()
    args = json.dumps({
        "owner": "myorg",
        "repo": "myrepo",
        "title": "Add OAuth",
        "body": "Implement OAuth login",
        "labels": "enhancement, security",
        "assignees": "alice",
    })
    tc = _make_tool_ctx(agent_ctx, "push_issue_draft", args)
    mock_gh = MagicMock()
    mock_gh.create_issue.return_value = {
        "number": 5,
        "html_url": "https://github.com/a/b/issues/5",
    }

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await push_issue_draft.on_invoke_tool(tc, args)

    data = json.loads(result)
    assert data["number"] == 5
    mock_gh.create_issue.assert_called_once_with(
        owner="myorg",
        repo="myrepo",
        title="Add OAuth",
        body="Implement OAuth login",
        labels=["enhancement", "security"],
        assignees=["alice"],
    )


@pytest.mark.asyncio
async def test_push_issue_draft_empty_labels_and_assignees() -> None:
    agent_ctx = _make_agent_ctx()
    args = json.dumps({
        "owner": "myorg",
        "repo": "myrepo",
        "title": "No labels",
        "body": "Body",
        "labels": "",
        "assignees": "",
    })
    tc = _make_tool_ctx(agent_ctx, "push_issue_draft", args)
    mock_gh = MagicMock()
    mock_gh.create_issue.return_value = {"number": 7, "html_url": "..."}

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        await push_issue_draft.on_invoke_tool(tc, args)

    mock_gh.create_issue.assert_called_once_with(
        owner="myorg",
        repo="myrepo",
        title="No labels",
        body="Body",
        labels=None,
        assignees=None,
    )


# ---------------------------------------------------------------------------
# Write tool: update_issue
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_issue_patches_only_provided_fields() -> None:
    agent_ctx = _make_agent_ctx()
    args = json.dumps({
        "owner": "myorg",
        "repo": "myrepo",
        "issue_number": 3,
        "title": "New title",
        "body": "",
        "state": "closed",
        "labels": "bug",
        "assignees": "",
    })
    tc = _make_tool_ctx(agent_ctx, "update_issue", args)
    mock_gh = MagicMock()
    mock_gh.update_issue.return_value = {
        "number": 3,
        "html_url": "https://github.com/a/b/issues/3",
    }

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await update_issue.on_invoke_tool(tc, args)

    data = json.loads(result)
    assert data["number"] == 3
    mock_gh.update_issue.assert_called_once_with(
        "myorg", "myrepo", 3,
        title="New title",
        state="closed",
        labels=["bug"],
    )


# ---------------------------------------------------------------------------
# Write tool: create_draft_pr
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_create_draft_pr_creates_pr() -> None:
    agent_ctx = _make_agent_ctx()
    args = json.dumps({
        "owner": "myorg",
        "repo": "myrepo",
        "title": "feat: add thing",
        "body": "Adds the thing",
        "head": "feat/thing",
        "base": "main",
    })
    tc = _make_tool_ctx(agent_ctx, "create_draft_pr", args)
    mock_gh = MagicMock()
    mock_gh.create_pull_request.return_value = {
        "number": 99,
        "html_url": "https://github.com/a/b/pull/99",
    }

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await create_draft_pr.on_invoke_tool(tc, args)

    data = json.loads(result)
    assert data["number"] == 99
    mock_gh.create_pull_request.assert_called_once_with(
        owner="myorg",
        repo="myrepo",
        title="feat: add thing",
        body="Adds the thing",
        head="feat/thing",
        base="main",
        draft=True,
    )


# ---------------------------------------------------------------------------
# Write tool: update_pull_request
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_update_pull_request_patches_fields() -> None:
    agent_ctx = _make_agent_ctx()
    args = json.dumps({
        "owner": "myorg",
        "repo": "myrepo",
        "pull_number": 10,
        "title": "Updated title",
        "body": "",
        "state": "",
    })
    tc = _make_tool_ctx(agent_ctx, "update_pull_request", args)
    mock_gh = MagicMock()
    mock_gh.update_pull_request.return_value = {
        "number": 10,
        "html_url": "https://github.com/a/b/pull/10",
    }

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await update_pull_request.on_invoke_tool(tc, args)

    data = json.loads(result)
    assert data["number"] == 10
    mock_gh.update_pull_request.assert_called_once_with(
        "myorg", "myrepo", 10, title="Updated title"
    )


# ---------------------------------------------------------------------------
# Write tool: ready_pr_for_review
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_ready_pr_for_review_sets_draft_false() -> None:
    agent_ctx = _make_agent_ctx()
    args = json.dumps({
        "owner": "myorg",
        "repo": "myrepo",
        "pull_number": 15,
    })
    tc = _make_tool_ctx(agent_ctx, "ready_pr_for_review", args)
    mock_gh = MagicMock()
    mock_gh.update_pull_request.return_value = {
        "number": 15,
        "html_url": "https://github.com/a/b/pull/15",
    }

    with patch("git_review.agent_tools.GitHubClient", return_value=mock_gh):
        result = await ready_pr_for_review.on_invoke_tool(tc, args)

    data = json.loads(result)
    assert data["number"] == 15
    mock_gh.update_pull_request.assert_called_once_with(
        "myorg", "myrepo", 15, draft=False
    )

