"""Conversational GitHub planning agent built on the OpenAI Agents SDK.

The agent is async and fully streaming: callers iterate over
``RunResultStreaming.stream_events()`` to receive text deltas, tool-call
events, reasoning items, and HITL interruption signals in real time.

Quick start::

    import asyncio
    from git_review.agent import AgentContext, build_agent, run_agent_streaming
    from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent

    ctx = AgentContext(
        owner="myorg",
        repo="myrepo",
        github_token="ghp_...",
        openai_api_key="sk-...",
        openai_base_url="",
        model="gpt-4o",
    )

    async def main():
        result = run_agent_streaming(ctx, "List open issues")
        async for event in result.stream_events():
            if isinstance(event, RawResponsesStreamEvent):
                if getattr(event.data, "type", "") == "response.output_text.delta":
                    print(event.data.delta, end="", flush=True)

        # Check for HITL interruptions (write-tool approvals)
        if result.interruptions:
            run_state = result.to_state()
            run_state.approve(result.interruptions[0])
            result2 = run_agent_streaming(ctx, run_state)
            async for event in result2.stream_events():
                ...

    asyncio.run(main())
"""

from __future__ import annotations

import logging
from typing import Union

try:
    from agents import Agent, RunConfig, Runner
    from agents.models.openai_provider import OpenAIProvider
    from agents.run import RunResultStreaming
    from agents.run_state import RunState
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'openai-agents' package is required. "
        "Install it with:  pip install 'git-review[agent]'"
    ) from _exc

from .agent_tools import (
    AgentContext,
    ALL_TOOLS,
)

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are a GitHub planning assistant integrated with the git-review toolkit.

You can help the user:
- List, search, and read GitHub issues and pull requests
- Generate structured issue drafts from plain-text requirements (via create_issue_draft)
- Push approved issue drafts to GitHub (via push_issue_draft)
- Update existing issues (via update_issue)
- Create draft pull requests (via create_draft_pr)
- Update or mark PRs ready for review (via update_pull_request, ready_pr_for_review)
- Generate agile sprint plans (via agile_plan)
- List repositories (via list_repos)

Current context
---------------
Owner : {owner}
Repo  : {repo}

Always use the above owner and repo as default arguments unless the user
explicitly specifies different ones.

IMPORTANT – write operations:
Any tool that modifies GitHub data (push_issue_draft, update_issue,
create_draft_pr, update_pull_request, ready_pr_for_review) requires explicit
human approval before it executes.  The system will automatically pause and
ask the user to approve or deny the operation.  Do NOT attempt to bypass this
or ask the user to confirm in chat — the approval UI handles it.

When the user asks to "create issues", "push issues", or similar, call
create_issue_draft first to produce drafts, present them to the user, and
only call push_issue_draft after the user confirms.
"""


def build_agent(ctx: AgentContext) -> Agent[AgentContext]:
    """Return an ``Agent`` configured with all git-review tools.

    Parameters
    ----------
    ctx:
        The :class:`AgentContext` used to populate the system prompt with the
        current owner and repo.  The same context must be passed as
        ``context=`` to :func:`run_agent_streaming`.
    """
    system_prompt = _SYSTEM_PROMPT_TEMPLATE.format(
        owner=ctx.owner or "<not set>",
        repo=ctx.repo or "<not set>",
    )
    return Agent(
        name="GitReviewAgent",
        instructions=system_prompt,
        tools=list(ALL_TOOLS),
    )


def _make_run_config(ctx: AgentContext) -> RunConfig:
    """Build a ``RunConfig`` using the credentials from *ctx*."""
    provider = OpenAIProvider(
        api_key=ctx.openai_api_key or None,
        base_url=ctx.openai_base_url or None,
    )
    return RunConfig(
        model=ctx.model,
        model_provider=provider,
        tracing_disabled=True,
    )


def run_agent_streaming(
    ctx: AgentContext,
    input: Union[str, RunState],  # noqa: A002 – mirrors SDK signature
    history: list | None = None,
) -> RunResultStreaming:
    """Start an async streaming agent run and return a :class:`RunResultStreaming`.

    The caller should iterate over ``result.stream_events()`` in an ``async
    for`` loop to receive events as they arrive, then check
    ``result.interruptions`` for any pending HITL approvals.

    Parameters
    ----------
    ctx:
        Agent credentials and repository context.
    input:
        Either a plain user message string or a :class:`RunState` object to
        resume an interrupted run (after calling ``state.approve()`` /
        ``state.reject()`` on the interruption items).
    history:
        Optional list of prior conversation turns in the format returned by
        ``RunResultStreaming.to_input_list()``.  Ignored when *input* is a
        :class:`RunState` (the history is already embedded in the state).

    Returns
    -------
    RunResultStreaming
        An object whose ``stream_events()`` async-iterator yields streaming
        events, and whose ``interruptions`` attribute lists any pending
        tool-approval requests after the stream completes.
    """
    agent = build_agent(ctx)
    run_config = _make_run_config(ctx)

    if isinstance(input, RunState):
        return Runner.run_streamed(
            agent,
            input,
            context=ctx,
            run_config=run_config,
        )

    effective_input: list | str
    if history:
        effective_input = list(history) + [{"role": "user", "content": input}]
    else:
        effective_input = input

    return Runner.run_streamed(
        agent,
        effective_input,
        context=ctx,
        run_config=run_config,
    )
