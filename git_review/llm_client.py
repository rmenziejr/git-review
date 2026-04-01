"""LLM summarisation client for git-review.

Delegates to the OpenAI chat-completions API (or any API that is
OpenAI-compatible, e.g. Azure OpenAI, local Ollama, Groq, …).
"""

from __future__ import annotations

import logging
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

from .models import ReviewSummary

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

_SYSTEM_PROMPT = """\
You are a concise engineering manager summarising GitHub activity.
Given a structured list of commits, issues, and pull requests from a
repository over a specific time window, produce a clear and well-organised
plain-text summary.

Format your response with the following sections (omit a section if there
is nothing to report):

## Highlights
A brief (2–4 sentence) executive summary of the most significant activity.

## Commits ({n})
A short narrative covering the main themes of the commits.

## Issues ({n})
Key issues opened or closed during the period.

## Pull Requests ({n})
Notable pull requests and their current status.

Be factual, professional, and concise.  Do not invent information.
"""


def _build_user_message(summary: ReviewSummary) -> str:
    repo_label = "all repositories" if summary.repo == "*" else f"{summary.owner}/{summary.repo}"
    lines: list[str] = [
        f"Repository: {repo_label}",
        f"Period: {summary.since.date()} → {summary.until.date()}",
        "",
    ]

    lines.append(f"### Commits ({len(summary.commits)})")
    for c in summary.commits:
        lines.append(f"- [{c.sha[:7]}] {c.message}  (by {c.author} on {c.authored_at.date()})")

    lines.append("")
    lines.append(f"### Issues ({len(summary.issues)})")
    for i in summary.issues:
        label_str = ", ".join(i.labels) if i.labels else "—"
        lines.append(
            f"- #{i.number} [{i.state}] {i.title}  (by {i.author}, labels: {label_str})"
        )

    lines.append("")
    lines.append(f"### Pull Requests ({len(summary.pull_requests)})")
    for pr in summary.pull_requests:
        merged = "merged" if pr.merged_at else pr.state
        lines.append(f"- #{pr.number} [{merged}] {pr.title}  (by {pr.author})")

    return "\n".join(lines)


class LLMClient:
    """Summarise a :class:`~git_review.models.ReviewSummary` using an LLM.

    Parameters
    ----------
    api_key:
        OpenAI (or compatible) API key.  When *None* the library uses the
        ``OPENAI_API_KEY`` environment variable automatically.
    model:
        Model identifier, e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``.
    base_url:
        Custom endpoint for non-OpenAI providers (Ollama, Azure, Groq …).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for LLM summarisation. "
                "Install it with:  pip install openai"
            )

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model = model

    def summarise(self, summary: ReviewSummary) -> str:
        """Return a markdown-formatted summary string for *summary*.

        The result is also stored in ``summary.summary_text`` for convenience.
        """
        n_commits = len(summary.commits)
        n_issues = len(summary.issues)
        n_prs = len(summary.pull_requests)

        system_prompt = _SYSTEM_PROMPT.format(
            n=f"{n_commits} commits, {n_issues} issues, {n_prs} PRs"
        )
        user_message = _build_user_message(summary)

        logger.debug("Sending %d tokens (approx) to %s", len(user_message) // 4, self._model)

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
        )
        text: str = response.choices[0].message.content or ""
        summary.summary_text = text
        return text
