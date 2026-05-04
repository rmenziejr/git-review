"""OpenAI Agents SDK tool functions for git-review.

Each tool is decorated with ``@function_tool`` from the Agents SDK.
Read-only tools execute immediately; write tools carry ``needs_approval=True``
so the agent run pauses for human-in-the-loop (HITL) approval before any
GitHub mutation is performed.

Tools receive repository credentials via the ``AgentContext`` dataclass that
is passed as the run context to ``Runner.run_streamed()``.

Typical usage::

    from git_review.agent import AgentContext, build_agent
    from agents import Runner, RunConfig

    ctx = AgentContext(
        owner="myorg",
        repo="myrepo",
        github_token="ghp_...",
        openai_api_key="sk-...",
        openai_base_url="",
    )
    agent = build_agent(ctx)
    result = Runner.run_streamed(agent, "List open issues", context=ctx)
    async for event in result.stream_events():
        ...
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

try:
    from agents import RunContextWrapper, function_tool
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'openai-agents' package is required for the agent. "
        "Install it with:  pip install 'git-review[agent]'"
    ) from _exc

from .agile_planner import AgilePlanner
from .github_client import GitHubClient
from .issue_factory import IssueDraft, IssueFactory

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Context dataclass – passed as `context` to Runner.run_streamed()
# ---------------------------------------------------------------------------


@dataclass
class AgentContext:
    """Runtime credentials and repository coordinates for agent tools."""

    owner: str
    repo: str
    github_token: str
    openai_api_key: str
    openai_base_url: str = ""
    model: str = "gpt-4o"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _make_gh(ctx: AgentContext) -> GitHubClient:
    return GitHubClient(token=ctx.github_token or None)


def _make_factory(ctx: AgentContext) -> IssueFactory:
    gh = _make_gh(ctx)
    return IssueFactory(
        github_client=gh,
        openai_api_key=ctx.openai_api_key or None,
        model=ctx.model,
        base_url=ctx.openai_base_url or None,
    )


def _draft_to_dict(draft: IssueDraft) -> dict:
    return {
        "title": draft.title,
        "body": draft.body,
        "labels": draft.labels,
        "assignees": draft.assignees,
        "milestone": draft.milestone,
    }


# ---------------------------------------------------------------------------
# Read-only tools (needs_approval=False)
# ---------------------------------------------------------------------------


@function_tool(needs_approval=False)
async def list_repos(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
) -> str:
    """List all non-archived repository names for a GitHub user or organisation.

    Args:
        owner: GitHub username or organisation name.
    """
    gh = _make_gh(ctx.context)
    repos = gh.list_repos(owner)
    return json.dumps(repos)


@function_tool(needs_approval=False)
async def search_issues(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
) -> str:
    """Return all currently open issues for a repository.

    Args:
        owner: Repository owner (user or org).
        repo: Repository name.
    """
    gh = _make_gh(ctx.context)
    issues = gh.get_open_issues(owner, repo)
    result = [
        {
            "number": i.number,
            "title": i.title,
            "state": i.state,
            "author": i.author,
            "labels": i.labels,
            "assignees": i.assignees,
            "milestone": i.milestone,
            "comments": i.comments,
            "url": i.url,
            "body_preview": (i.body or "")[:200],
        }
        for i in issues
    ]
    return json.dumps(result)


@function_tool(needs_approval=False)
async def get_issue(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
    issue_number: int,
) -> str:
    """Fetch full details for a single GitHub issue.

    Args:
        owner: Repository owner.
        repo: Repository name.
        issue_number: The issue number.
    """
    gh = _make_gh(ctx.context)
    issue = gh.get_issue(owner, repo, issue_number)
    return json.dumps(
        {
            "number": issue.number,
            "title": issue.title,
            "state": issue.state,
            "author": issue.author,
            "labels": issue.labels,
            "assignees": issue.assignees,
            "milestone": issue.milestone,
            "comments": issue.comments,
            "url": issue.url,
            "body": issue.body,
        }
    )


@function_tool(needs_approval=False)
async def list_pull_requests(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
) -> str:
    """Return all currently open pull requests for a repository.

    Args:
        owner: Repository owner.
        repo: Repository name.
    """
    gh = _make_gh(ctx.context)
    prs = gh.get_open_pull_requests(owner, repo)
    result = [
        {
            "number": pr.number,
            "title": pr.title,
            "state": pr.state,
            "draft": pr.draft,
            "author": pr.author,
            "labels": pr.labels,
            "base_branch": pr.base_branch,
            "head_branch": pr.head_branch,
            "url": pr.url,
            "body_preview": (pr.body or "")[:200],
        }
        for pr in prs
    ]
    return json.dumps(result)


@function_tool(needs_approval=False)
async def create_issue_draft(
    ctx: RunContextWrapper[AgentContext],
    requirements: str,
) -> str:
    """Parse a plain-text requirements description and return structured issue drafts for review.

    This tool does NOT write to GitHub. Call push_issue_draft after the user
    approves the generated drafts.

    Args:
        requirements: Plain-text description of the feature or work to turn into issues.
    """
    factory = _make_factory(ctx.context)
    drafts = factory.parse_requirements(requirements)
    return json.dumps([_draft_to_dict(d) for d in drafts])


@function_tool(needs_approval=False)
async def agile_plan(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
    sprint_capacity: int,
    num_sprints: int,
) -> str:
    """Fetch all open issues and PRs, infer dependencies, and produce a sprint plan.

    Args:
        owner: Repository owner.
        repo: Repository name.
        sprint_capacity: Number of issues per sprint.
        num_sprints: Number of sprints to plan.
    """
    gh = _make_gh(ctx.context)
    planner = AgilePlanner(
        github_client=gh,
        openai_api_key=ctx.context.openai_api_key or None,
        model=ctx.context.model,
        base_url=ctx.context.openai_base_url or None,
        sprint_capacity=sprint_capacity,
        num_sprints=num_sprints,
    )
    result = planner.analyse(owner, repo)
    sprints = [
        {
            "sprint_number": s.sprint_number,
            "theme": s.theme,
            "issues": s.issues,
            "rationale": s.rationale,
            "deferred": s.deferred,
        }
        for s in result.sprints
    ]
    return json.dumps(
        {
            "summary": result.summary_text,
            "sprints": sprints,
            "dependencies": [
                {
                    "from_issue": d.from_issue,
                    "to_issue": d.to_issue,
                    "dep_type": d.dep_type,
                    "confidence": d.confidence,
                    "reason": d.reason,
                }
                for d in result.dependencies
            ],
        }
    )


# ---------------------------------------------------------------------------
# Write tools (needs_approval=True) – paused for HITL before execution
# ---------------------------------------------------------------------------


@function_tool(needs_approval=True)
async def push_issue_draft(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
    title: str,
    body: str,
    labels: str,
    assignees: str,
) -> str:
    """Create a single GitHub issue after human approval.

    Args:
        owner: Repository owner.
        repo: Repository name.
        title: Issue title.
        body: Issue body in GitHub Markdown.
        labels: Comma-separated label names (empty string for none).
        assignees: Comma-separated GitHub usernames (empty string for none).
    """
    gh = _make_gh(ctx.context)
    label_list = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
    assignee_list = [a.strip() for a in assignees.split(",") if a.strip()]
    result = gh.create_issue(
        owner=owner,
        repo=repo,
        title=title,
        body=body,
        labels=label_list or None,
        assignees=assignee_list or None,
    )
    return json.dumps({"number": result.get("number"), "url": result.get("html_url")})


@function_tool(needs_approval=True)
async def update_issue(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
    issue_number: int,
    title: str,
    body: str,
    state: str,
    labels: str,
    assignees: str,
) -> str:
    """Update an existing GitHub issue after human approval.

    Pass empty strings to leave a field unchanged (title, body, labels, assignees).
    Pass empty string for state to leave it unchanged; use 'open' or 'closed' to change it.

    Args:
        owner: Repository owner.
        repo: Repository name.
        issue_number: Issue number to update.
        title: New title, or empty string to leave unchanged.
        body: New body, or empty string to leave unchanged.
        state: New state ('open', 'closed'), or empty string to leave unchanged.
        labels: Comma-separated replacement labels, or empty string to leave unchanged.
        assignees: Comma-separated replacement assignees, or empty string to leave unchanged.
    """
    gh = _make_gh(ctx.context)
    kwargs: dict = {}
    if title:
        kwargs["title"] = title
    if body:
        kwargs["body"] = body
    if state:
        kwargs["state"] = state
    if labels:
        kwargs["labels"] = [lbl.strip() for lbl in labels.split(",") if lbl.strip()]
    if assignees:
        kwargs["assignees"] = [a.strip() for a in assignees.split(",") if a.strip()]
    result = gh.update_issue(owner, repo, issue_number, **kwargs)
    return json.dumps({"number": result.get("number"), "url": result.get("html_url")})


@function_tool(needs_approval=True)
async def create_draft_pr(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
    title: str,
    body: str,
    head: str,
    base: str,
) -> str:
    """Create a draft pull request after human approval.

    Args:
        owner: Repository owner.
        repo: Repository name.
        title: PR title.
        body: PR body in GitHub Markdown.
        head: The source branch with your changes.
        base: The target branch to merge into.
    """
    gh = _make_gh(ctx.context)
    result = gh.create_pull_request(
        owner=owner,
        repo=repo,
        title=title,
        body=body,
        head=head,
        base=base,
        draft=True,
    )
    return json.dumps({"number": result.get("number"), "url": result.get("html_url")})


@function_tool(needs_approval=True)
async def update_pull_request(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
    pull_number: int,
    title: str,
    body: str,
    state: str,
) -> str:
    """Update the title, body, or state of a pull request after human approval.

    Pass empty strings to leave a field unchanged.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pull_number: PR number to update.
        title: New title, or empty string to leave unchanged.
        body: New body, or empty string to leave unchanged.
        state: New state ('open' or 'closed'), or empty string to leave unchanged.
    """
    gh = _make_gh(ctx.context)
    kwargs: dict = {}
    if title:
        kwargs["title"] = title
    if body:
        kwargs["body"] = body
    if state:
        kwargs["state"] = state
    result = gh.update_pull_request(owner, repo, pull_number, **kwargs)
    return json.dumps({"number": result.get("number"), "url": result.get("html_url")})


@function_tool(needs_approval=True)
async def ready_pr_for_review(
    ctx: RunContextWrapper[AgentContext],
    owner: str,
    repo: str,
    pull_number: int,
) -> str:
    """Convert a draft pull request to ready-for-review status after human approval.

    Args:
        owner: Repository owner.
        repo: Repository name.
        pull_number: The draft PR number to mark as ready.
    """
    gh = _make_gh(ctx.context)
    result = gh.update_pull_request(owner, repo, pull_number, draft=False)
    return json.dumps({"number": result.get("number"), "url": result.get("html_url")})


# ---------------------------------------------------------------------------
# Exported tool list
# ---------------------------------------------------------------------------

ALL_TOOLS = [
    list_repos,
    search_issues,
    get_issue,
    list_pull_requests,
    create_issue_draft,
    agile_plan,
    push_issue_draft,
    update_issue,
    create_draft_pr,
    update_pull_request,
    ready_pr_for_review,
]
