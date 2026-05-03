"""Agile sprint planner for git-review.

Fetches all open issues and pull requests for a repository or organisation,
then uses an LLM to produce:

- A dependency graph (blocks / blocked-by relationships between issues).
- A prioritised sprint plan covering a configurable number of future sprints.
- Label recommendations that can optionally be written back to GitHub.

Example
-------
::

    from git_review.github_client import GitHubClient
    from git_review.agile_planner import AgilePlanner

    gh = GitHubClient(token="ghp_…")
    planner = AgilePlanner(
        github_client=gh,
        openai_api_key="sk-…",
        sprint_capacity=10,
        num_sprints=3,
    )

    result = planner.analyse("myorg", "myrepo")
    for sprint in result.sprints:
        print(sprint.sprint_number, sprint.theme, sprint.issues)
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment,misc]

from pydantic import BaseModel, Field

from .github_client import GitHubClient
from .models import AgilePlanResult, Issue, IssueDependency, PullRequest, SprintRecommendation

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "gpt-4o-mini"

# ---------------------------------------------------------------------------
# Regex patterns for explicit cross-references in issue / PR bodies
# ---------------------------------------------------------------------------

_CLOSES_RE = re.compile(
    r"(?:closes|close|closed|fixes|fix|fixed|resolves|resolve|resolved)\s+#(\d+)",
    re.IGNORECASE,
)
_BLOCKS_RE = re.compile(r"blocks\s+#(\d+)", re.IGNORECASE)
_BLOCKED_BY_RE = re.compile(r"blocked\s+(?:by\s+)?#(\d+)", re.IGNORECASE)
_DEPENDS_ON_RE = re.compile(r"depends\s+on\s+#(\d+)", re.IGNORECASE)
_LINKED_ISSUE_RE = re.compile(r"#(\d+)")

# ---------------------------------------------------------------------------
# Pydantic models used for structured LLM output
# ---------------------------------------------------------------------------


class _LLMDependency(BaseModel):
    from_issue: int
    to_issue: int
    type: str = Field(description="'blocks' or 'blocked-by'")
    confidence: float = Field(ge=0.0, le=1.0)
    reason: str


class _LLMSprint(BaseModel):
    sprint_number: int
    issues: list[int]
    theme: str
    rationale: str
    deferred: list[int] = Field(default_factory=list)


class _LLMLabelRec(BaseModel):
    issue_number: int
    labels: list[str]


class _LLMAgileResponse(BaseModel):
    dependencies: list[_LLMDependency] = Field(default_factory=list)
    sprint_plan: list[_LLMSprint] = Field(default_factory=list)
    label_recommendations: list[_LLMLabelRec] = Field(default_factory=list)
    summary: str = ""


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are an experienced agile coach and engineering manager.

You will receive a list of open GitHub issues and open pull requests for one or
more repositories.  Your job is to produce a structured sprint plan in JSON.

Return ONLY valid JSON with the following top-level keys. Do not include prose
outside the JSON object.

{
  "dependencies": [
    {
      "from_issue": <int>,
      "to_issue": <int>,
      "type": "blocks" | "blocked-by",
      "confidence": <float 0-1>,
      "reason": "<one sentence>"
    }
  ],
  "sprint_plan": [
    {
      "sprint_number": <int starting at 1>,
      "issues": [<issue numbers for this sprint>],
      "theme": "<short theme label>",
      "rationale": "<2-3 sentences>",
      "deferred": [<issue numbers deferred from this sprint>]
    }
  ],
  "label_recommendations": [
    {
      "issue_number": <int>,
      "labels": ["<label>", ...]
    }
  ],
  "summary": "<3-5 sentence executive summary of the plan>"
}

Guidelines
----------
- Fill `dependencies` with BOTH explicit relationships stated in issue bodies AND
  semantically inferred relationships (e.g. "implement auth" must precede
  "build user dashboard").  Explicit relationships should have confidence >= 0.95.
- For `sprint_plan`, respect the sprint capacity limit stated in the user message.
  Prioritise: critical bugs > security issues > blockers for other work > features
  by estimated value > nice-to-haves.
- Include ALL open issues exactly once: either in a sprint's `issues` list OR in a
  sprint's `deferred` list (use the last sprint's `deferred` for issues that do not
  fit any sprint).
- For `label_recommendations`, suggest labels like "priority: critical",
  "priority: high", "priority: medium", "priority: low", "blocked", "good first issue"
  for issues that would benefit from them.  Do NOT suggest removing existing labels.
- Be factual. Do not invent requirements not present in the issue data.
"""

# ---------------------------------------------------------------------------
# Public class
# ---------------------------------------------------------------------------


class AgilePlanner:
    """Fetch open issues/PRs and generate an LLM-powered sprint plan.

    Parameters
    ----------
    github_client:
        An authenticated :class:`~git_review.github_client.GitHubClient`.
    openai_api_key:
        OpenAI (or compatible) API key.  Falls back to ``OPENAI_API_KEY`` env var.
    model:
        LLM model identifier (default: ``"gpt-4o-mini"``).
    base_url:
        Custom OpenAI-compatible API base URL.
    sprint_capacity:
        Maximum number of issues per sprint (default: 10).
    num_sprints:
        Number of future sprints to plan (default: 3).
    """

    def __init__(
        self,
        github_client: GitHubClient,
        openai_api_key: Optional[str] = None,
        model: str = _DEFAULT_MODEL,
        base_url: Optional[str] = None,
        sprint_capacity: int = 10,
        num_sprints: int = 3,
    ) -> None:
        if OpenAI is None:  # pragma: no cover
            raise ImportError(
                "The 'openai' package is required for agile planning. "
                "Install it with:  pip install openai"
            )
        kwargs: dict = {}
        if openai_api_key:
            kwargs["api_key"] = openai_api_key
        if base_url:
            kwargs["base_url"] = base_url
        self._client = OpenAI(**kwargs)
        self._model = model
        self._gh = github_client
        self._sprint_capacity = sprint_capacity
        self._num_sprints = num_sprints

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def analyse(self, owner: str, repo: str) -> AgilePlanResult:
        """Fetch open issues/PRs for *owner/repo* and return a sprint plan.

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        """
        logger.debug("Fetching open issues for %s/%s", owner, repo)
        issues = self._gh.get_open_issues(owner, repo)
        prs = self._gh.get_open_pull_requests(owner, repo)
        return self._plan(owner, repo, issues, prs)

    def analyse_org(self, owner: str) -> AgilePlanResult:
        """Aggregate open issues/PRs across all repos for *owner* and plan sprints.

        Parameters
        ----------
        owner:
            GitHub username or organisation name.
        """
        repo_names = self._gh.list_repos(owner)
        all_issues: list[Issue] = []
        all_prs: list[PullRequest] = []
        for repo_name in repo_names:
            logger.debug("Fetching open issues/PRs for %s/%s", owner, repo_name)
            all_issues.extend(self._gh.get_open_issues(owner, repo_name))
            all_prs.extend(self._gh.get_open_pull_requests(owner, repo_name))
        return self._plan(owner, "*", all_issues, all_prs)

    def apply_labels(
        self,
        owner: str,
        repo: str,
        result: AgilePlanResult,
        dry_run: bool = True,
    ) -> list[dict]:
        """Apply label recommendations from *result* back to GitHub issues.

        Parameters
        ----------
        owner, repo:
            Repository coordinates (ignored when ``result.repo == "*"``).
        result:
            The :class:`~git_review.models.AgilePlanResult` returned by
            :meth:`analyse` or :meth:`analyse_org`.
        dry_run:
            When *True* (default), log what *would* happen but make no API calls.

        Returns
        -------
        list[dict]
            GitHub API responses for each updated issue (empty list when
            ``dry_run=True``).
        """
        responses: list[dict] = []
        for issue in result.issues:
            new_labels = result.label_recommendations.get(issue.number)
            if not new_labels:
                continue
            merged = sorted(set(issue.labels) | set(new_labels))
            if set(merged) == set(issue.labels):
                continue
            issue_repo = issue.repo
            if "/" in issue_repo:
                parts = issue_repo.split("/", 1)
                issue_owner, issue_repo_name = parts[0], parts[1]
            else:
                issue_owner, issue_repo_name = owner, repo
            if dry_run:
                logger.info(
                    "[dry-run] Would set labels on %s#%d: %s",
                    issue.repo,
                    issue.number,
                    merged,
                )
            else:
                resp = self._gh.update_issue_labels(
                    issue_owner, issue_repo_name, issue.number, merged
                )
                responses.append(resp)
        return responses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _plan(
        self,
        owner: str,
        repo: str,
        issues: list[Issue],
        prs: list[PullRequest],
    ) -> AgilePlanResult:
        """Run the LLM analysis and return an :class:`AgilePlanResult`."""
        explicit_deps = _extract_explicit_dependencies(issues, prs)
        user_message = self._build_context_message(issues, prs)
        llm_result = self._call_llm(user_message)

        # Merge explicit deps (they take precedence / higher confidence)
        dep_keys: set[tuple[int, int, str]] = set()
        all_deps: list[IssueDependency] = []
        for dep in explicit_deps:
            key = (dep.from_issue, dep.to_issue, dep.dep_type)
            dep_keys.add(key)
            all_deps.append(dep)

        for ld in llm_result.dependencies:
            key = (ld.from_issue, ld.to_issue, ld.type)
            if key not in dep_keys:
                all_deps.append(
                    IssueDependency(
                        from_issue=ld.from_issue,
                        to_issue=ld.to_issue,
                        dep_type=ld.type,
                        confidence=ld.confidence,
                        reason=ld.reason,
                        source="llm",
                    )
                )

        sprints = [
            SprintRecommendation(
                sprint_number=s.sprint_number,
                issues=s.issues,
                theme=s.theme,
                rationale=s.rationale,
                deferred=s.deferred,
            )
            for s in llm_result.sprint_plan
        ]

        label_recs: dict[int, list[str]] = {
            lr.issue_number: lr.labels for lr in llm_result.label_recommendations
        }

        return AgilePlanResult(
            owner=owner,
            repo=repo,
            issues=issues,
            pull_requests=prs,
            dependencies=all_deps,
            sprints=sprints,
            summary_text=llm_result.summary,
            label_recommendations=label_recs,
        )

    def _build_context_message(
        self,
        issues: list[Issue],
        prs: list[PullRequest],
    ) -> str:
        """Format open issues and PRs as a structured text blob for the LLM."""
        issue_map = {i.number: i for i in issues}

        # Map issue number → list of PR titles/numbers that reference it
        pr_refs: dict[int, list[str]] = {}
        for pr in prs:
            for num in _extract_issue_refs(pr.body):
                pr_refs.setdefault(num, []).append(f"#{pr.number} {pr.title}")

        lines: list[str] = [
            f"Sprint capacity: {self._sprint_capacity} issues per sprint",
            f"Number of sprints to plan: {self._num_sprints}",
            f"Total open issues: {len(issues)}",
            f"Total open PRs: {len(prs)}",
            "",
            "## Open Issues",
        ]

        for issue in issues:
            assignees = ", ".join(issue.assignees) if issue.assignees else "—"
            labels = ", ".join(issue.labels) if issue.labels else "—"
            milestone = issue.milestone or "—"
            body_snippet = _trim(issue.body, 200)
            linked_prs = "; ".join(pr_refs.get(issue.number, [])) or "—"
            lines.append(
                f"### #{issue.number}: {issue.title}\n"
                f"- labels: {labels}\n"
                f"- assignees: {assignees}\n"
                f"- milestone: {milestone}\n"
                f"- comments: {issue.comments}\n"
                f"- linked PRs: {linked_prs}\n"
                f"- body: {body_snippet}"
            )

        lines += ["", "## Open Pull Requests"]
        for pr in prs:
            linked_issues = ", ".join(
                f"#{n}" for n in _extract_issue_refs(pr.body)
                if n in issue_map
            ) or "—"
            draft_str = " [DRAFT]" if pr.draft else ""
            labels = ", ".join(pr.labels) if pr.labels else "—"
            reviewers = ", ".join(pr.requested_reviewers) if pr.requested_reviewers else "—"
            body_snippet = _trim(pr.body, 150)
            lines.append(
                f"### #{pr.number}{draft_str}: {pr.title}\n"
                f"- author: {pr.author}\n"
                f"- labels: {labels}\n"
                f"- requested reviewers: {reviewers}\n"
                f"- linked issues: {linked_issues}\n"
                f"- body: {body_snippet}"
            )

        return "\n".join(lines)

    def _call_llm(self, user_message: str) -> _LLMAgileResponse:
        """Send *user_message* to the LLM and parse the structured JSON response."""
        logger.debug(
            "Sending agile planner context (~%d chars) to %s",
            len(user_message),
            self._model,
        )
        response = self._client.beta.chat.completions.parse(
            model=self._model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            response_format=_LLMAgileResponse,
        )
        parsed = response.choices[0].message.parsed
        if parsed is None:
            # Fall back to raw JSON parsing if structured output is unavailable
            raw_text = response.choices[0].message.content or "{}"
            try:
                data = json.loads(raw_text)
                parsed = _LLMAgileResponse(**data)
            except Exception:
                parsed = _LLMAgileResponse()
        return parsed


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------


def _extract_explicit_dependencies(
    issues: list[Issue],
    prs: list[PullRequest],
) -> list[IssueDependency]:
    """Scan issue and PR bodies for explicit cross-references."""
    issue_numbers = {i.number for i in issues}
    deps: list[IssueDependency] = []

    def _add(from_num: int, to_num: int, dep_type: str, reason: str) -> None:
        if from_num != to_num and to_num in issue_numbers:
            deps.append(
                IssueDependency(
                    from_issue=from_num,
                    to_issue=to_num,
                    dep_type=dep_type,
                    confidence=1.0,
                    reason=reason,
                    source="explicit",
                )
            )

    for item in [*issues, *prs]:
        num = item.number
        body = item.body or ""
        for m in _BLOCKS_RE.finditer(body):
            _add(num, int(m.group(1)), "blocks", f"#{num} explicitly states it blocks #{m.group(1)}")
        for m in _BLOCKED_BY_RE.finditer(body):
            _add(num, int(m.group(1)), "blocked-by", f"#{num} is blocked by #{m.group(1)}")
        for m in _DEPENDS_ON_RE.finditer(body):
            _add(num, int(m.group(1)), "blocked-by", f"#{num} depends on #{m.group(1)}")
        for m in _CLOSES_RE.finditer(body):
            ref = int(m.group(1))
            if ref in issue_numbers:
                _add(num, ref, "blocks", f"PR #{num} closes issue #{ref}")

    return deps


def _extract_issue_refs(body: str) -> list[int]:
    """Return all #N issue references found in *body*."""
    return [int(m.group(1)) for m in _LINKED_ISSUE_RE.finditer(body or "")]


def _trim(text: str, max_len: int = 200) -> str:
    """Normalise whitespace and truncate *text* to *max_len* characters."""
    cleaned = re.sub(r"\s+", " ", (text or "").strip())
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[: max_len - 1] + "…"
