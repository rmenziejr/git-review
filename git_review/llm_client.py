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
from .prompt_utils import render_prompt, validate_prompt_template

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

_PROMPT_VARS: set[str] = {"n", "n_commits", "n_issues", "n_prs"}

_DEFAULT_SYSTEM_PROMPT = """\
You are an experienced engineering manager summarizing GitHub activity across one or more repositories.

Given a structured list of commits, issues, pull requests, releases, and contributors over a defined time window, produce a clear, structured, and professional plain-text summary.

Your summary should emphasize:
- What was accomplished
- Why it mattered (objective)
- What outcomes or progress were achieved (results)

Organize the output as follows:

## Highlights
Provide a 4–8 sentence executive summary that:
- Synthesizes the most important work across all repositories
- Describes the underlying objectives or themes (e.g., performance improvements, new features, stability, infrastructure)
- Highlights measurable or observable outcomes (e.g., reduced latency, improved reliability, new capabilities delivered, technical debt reduced)
- Avoids listing items; instead, tell a cohesive story of progress

## Repository Breakdown

For each repository, include a subsection:

### <Repository Name>

#### Summary
A concise 2–4 sentence overview of:
- The primary focus of work in this repository
- The objective of the changes
- The resulting impact or progress

#### Commits ({{ n }})
- Summarize key themes (not individual commits)
- Group related changes (e.g., refactoring, feature additions, bug fixes, infra changes)

#### Issues ({{ n }})
- Highlight important issues opened or resolved
- Focus on blockers, bugs, or notable discussions that influenced progress

#### Pull Requests ({{ n }})
- Summarize major PRs and their purpose
- Include status where relevant (merged, open, in review)
- Emphasize what functionality or improvement each delivered

#### Releases
- Summarize any releases and their significance
- Highlight major features, fixes, or version changes

---

### Writing Guidelines

- Be factual, professional, and concise
- Do NOT invent or infer details not present in the data
- Avoid listing every item; prioritize signal over noise
- Group related work into themes instead of enumerating raw activity
- Focus on outcomes and impact, not just actions
- Use clean, readable formatting suitable for leadership review

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
        stats_str = f"  [+{c.additions}/-{c.deletions}]" if (c.additions or c.deletions) else ""
        lines.append(
            f"- [{c.sha[:7]}] {c.message}  (by {c.author} on {c.authored_at.date()}){stats_str}"
        )

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
        draft_str = " [DRAFT]" if pr.draft else ""
        if pr.reviewer_comments:
            reviewer_str = "; reviewers: " + ", ".join(
                f"{login}({count} comment{'s' if count != 1 else ''})"
                for login, count in sorted(
                    pr.reviewer_comments.items(), key=lambda x: x[1], reverse=True
                )
            )
        else:
            reviewer_str = ""
        lines.append(
            f"- #{pr.number} [{merged}]{draft_str} {pr.title}  (by {pr.author}{reviewer_str})"
        )

    lines.append("")
    lines.append(f"### Releases ({len(summary.releases)})")
    for r in summary.releases:
        pub_str = str(r.published_at.date()) if r.published_at else "unpublished"
        pre_str = " [pre-release]" if r.prerelease else ""
        lines.append(f"- {r.tag}{pre_str}: {r.name}  (published {pub_str} by {r.author or 'unknown'})")

    lines.append("")
    lines.append(f"### Contributors ({len(summary.contributors)})")
    for c in summary.contributors:
        lines.append(f"- {c.login}: {c.contributions} contributions")

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
    system_prompt:
        Optional Jinja2 template string to replace the built-in system
        prompt.  The following variables are available for use inside the
        template: ``n``, ``n_commits``, ``n_issues``, ``n_prs``.
        A :exc:`ValueError` is raised at construction time if the template
        references any other variable.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for LLM summarisation. "
                "Install it with:  pip install openai"
            )

        if system_prompt is not None:
            validate_prompt_template(
                system_prompt,
                _PROMPT_VARS,
                label="system_prompt",
            )

        kwargs: dict = {}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url

        self._client = OpenAI(**kwargs)
        self._model = model
        self._system_prompt_template = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def summarise(self, summary: ReviewSummary) -> str:
        """Return a markdown-formatted summary string for *summary*.

        The result is also stored in ``summary.summary_text`` for convenience.
        """
        n_commits = len(summary.commits)
        n_issues = len(summary.issues)
        n_prs = len(summary.pull_requests)

        system_prompt = render_prompt(
            self._system_prompt_template,
            n=f"{n_commits} commits, {n_issues} issues, {n_prs} PRs",
            n_commits=n_commits,
            n_issues=n_issues,
            n_prs=n_prs,
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
