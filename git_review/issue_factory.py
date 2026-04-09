"""Convert a markdown requirements document into GitHub issues.

Workflow
--------
1.  Parse the markdown with :meth:`IssueFactory.parse_requirements`.
    The LLM returns a structured list of :class:`IssueDraft` objects that you
    can inspect and edit before committing.
2.  Review the drafts interactively (e.g. via the CLI).
3.  Call :meth:`IssueFactory.push_issues` to create the approved drafts as
    real GitHub issues via :class:`~git_review.github_client.GitHubClient`.

Example
-------
::

    from git_review.issue_factory import IssueFactory, IssueDraft
    from git_review.github_client import GitHubClient

    gh = GitHubClient(token="ghp_...")
    factory = IssueFactory(openai_api_key="sk-...", github_client=gh)

    with open("requirements.md") as f:
        drafts = factory.parse_requirements(f.read())

    # inspect / approve drafts, then push:
    factory.push_issues("myorg", "myrepo", drafts)
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence

from pydantic import BaseModel, Field

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

from .github_client import GitHubClient
from .models import Milestone
from .prompt_utils import validate_prompt_template

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

# No template variables are rendered into the issue-factory system prompt.
_PROMPT_VARS: set[str] = set()

_DEFAULT_SYSTEM_PROMPT = """\
You are a senior software engineer helping to convert a product requirements \
document into well-structured GitHub issues.

Analyze all requirements and user stories holistically, then produce the \
minimal set of clear, objective, and achievable issues needed to fully satisfy \
all requirements and user story criteria. Each issue should represent a \
coherent, independently deliverable unit of work.

Each issue must have:
- A concise, action-oriented title (start with a verb, e.g. "Add", "Fix", \
"Implement").
- A body written in GitHub Markdown that includes:
  - A short description of the work to be done.
  - A "Sub-tasks" section with a checklist of concrete, independently \
completable steps required to deliver the issue.
- A list of relevant labels from: bug, enhancement, documentation, \
question, help wanted, good first issue.
- An optional list of GitHub usernames to assign (leave empty if none are \
mentioned in the document).

Return ONLY the structured JSON. Do not include any prose outside the JSON.
"""


class IssueDraft(BaseModel):
    """A single GitHub issue to be created, generated from a requirements doc."""

    title: str = Field(..., description="Short, action-oriented issue title.")
    body: str = Field(
        ...,
        description=(
            "Full issue body in GitHub Markdown, including description and "
            "sub-tasks checklist."
        ),
    )
    labels: list[str] = Field(
        default_factory=list,
        description="Label names to apply (e.g. 'enhancement', 'bug').",
    )
    assignees: list[str] = Field(
        default_factory=list,
        description="GitHub usernames to assign the issue to.",
    )
    milestone: Optional[int] = Field(
        default=None,
        description="Milestone number to attach the issue to.",
    )


class IssueList(BaseModel):
    """Container returned by the LLM – a list of issue drafts."""

    issues: list[IssueDraft] = Field(
        ...,
        description="All issues extracted from the requirements document.",
    )


class IssueFactory:
    """Parse a markdown requirements document and create GitHub issues.

    Parameters
    ----------
    github_client:
        An authenticated :class:`~git_review.github_client.GitHubClient`.
    openai_api_key:
        OpenAI API key.  Falls back to the ``OPENAI_API_KEY`` env variable.
    model:
        LLM model identifier (default: ``"gpt-4o-mini"``).
    base_url:
        Custom OpenAI-compatible API base URL (Ollama, Groq, Azure, …).
    system_prompt:
        Optional Jinja2 template string to replace the built-in system
        prompt.  The issue-factory prompt has no template variables, so
        any ``{{ var }}`` expression will raise a :exc:`ValueError` at
        construction time.
    """

    def __init__(
        self,
        github_client: GitHubClient,
        openai_api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
        system_prompt: Optional[str] = None,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for issue generation. "
                "Install it with:  pip install openai"
            )

        if system_prompt is not None:
            validate_prompt_template(
                system_prompt,
                _PROMPT_VARS,
                label="system_prompt",
            )

        kwargs: dict = {}
        if openai_api_key:
            kwargs["api_key"] = openai_api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._gh = github_client
        self._system_prompt = system_prompt or _DEFAULT_SYSTEM_PROMPT

    def parse_requirements(
        self,
        markdown_text: str,
        milestones: Optional[Sequence[Milestone]] = None,
    ) -> list[IssueDraft]:
        """Parse *markdown_text* and return a list of :class:`IssueDraft` objects.

        The LLM is asked to return structured JSON matching the
        :class:`IssueList` schema so each draft can be inspected before
        being pushed.

        Parameters
        ----------
        markdown_text:
            The full content of the requirements markdown document.
        milestones:
            Optional list of :class:`~git_review.models.Milestone` objects
            fetched from the target repository.  When provided, their titles,
            numbers, and descriptions are appended to the user message so the
            LLM can assign each issue to the most relevant milestone by
            setting the ``milestone`` field to the milestone's number.
        """
        logger.debug(
            "Parsing requirements document (%d chars) with %s",
            len(markdown_text),
            self._model,
        )

        user_content = markdown_text
        if milestones:
            lines = [
                "\n\n---\n**Available milestones** – assign each issue to the most "
                "appropriate milestone by setting its `milestone` field to the "
                "milestone number.  Leave `milestone` as null if no milestone fits.\n",
            ]
            for m in milestones:
                desc = f" — {m.description}" if m.description else ""
                lines.append(f"- #{m.number}: \"{m.title}\"{desc}")
            user_content = markdown_text + "\n".join(lines)

        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": self._system_prompt},
                {"role": "user", "content": user_content},
            ],
            response_format=IssueList,
        )
        parsed: IssueList = response.choices[0].message.parsed
        logger.debug("LLM returned %d issue drafts", len(parsed.issues))
        return parsed.issues

    def push_issues(
        self,
        owner: str,
        repo: str,
        drafts: list[IssueDraft],
        milestone: Optional[int] = None,
    ) -> list[dict]:
        """Create *drafts* as real GitHub issues in *owner/repo*.

        Parameters
        ----------
        owner, repo:
            Target repository coordinates.
        drafts:
            The approved :class:`IssueDraft` objects to push.
        milestone:
            Optional milestone number to attach to every issue.  When provided
            this overrides the ``milestone`` field on individual drafts.

        Returns
        -------
        list[dict]
            Raw GitHub API responses for each created issue (contain the issue
            number, URL, etc.).
        """
        results: list[dict] = []
        for draft in drafts:
            logger.debug("Creating issue: %s", draft.title)
            effective_milestone = milestone if milestone is not None else draft.milestone
            result = self._gh.create_issue(
                owner=owner,
                repo=repo,
                title=draft.title,
                body=draft.body,
                labels=draft.labels or None,
                assignees=draft.assignees or None,
                milestone=effective_milestone,
            )
            results.append(result)
        return results
