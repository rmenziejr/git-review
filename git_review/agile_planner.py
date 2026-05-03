"""Agile sprint planner for git-review.

Fetches all open issues and pull requests for a repository or organisation,
reads their existing GitHub blocking/blocked-by relationships, then uses an
LLM to produce:

- A full dependency graph (including both existing GitHub relationships and
  newly inferred ones).
- A prioritised sprint plan covering a configurable number of future sprints.
- Priority label recommendations that can optionally be written back to GitHub.

New dependency relationships inferred by the LLM are written back to GitHub
via the native issue-dependencies REST API (not labels), using:
  POST /repos/{owner}/{repo}/issues/{issue_number}/dependencies/blocked_by

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

    # Write new inferred blocking relationships back to GitHub
    planner.apply_relationships("myorg", "myrepo", result, dry_run=False)
"""

from __future__ import annotations

import json
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
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
# Pydantic models for structured LLM output
# ---------------------------------------------------------------------------


class _LLMDependency(BaseModel):
    from_issue: int
    to_issue: int
    type: str = Field(description="'blocked-by': from_issue is blocked by to_issue")
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
more repositories.  Some issues already have existing blocking/blocked-by
relationships recorded in GitHub — these are labelled "source: github" in the
input and should NOT be repeated in your output.

Your job is to produce a structured sprint plan in JSON.

Return ONLY valid JSON with the following top-level keys. Do not include prose
outside the JSON object.

{
  "dependencies": [
    {
      "from_issue": <int>,
      "to_issue": <int>,
      "type": "blocked-by",
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

Guidelines for `dependencies`
------------------------------
- Express ALL dependency relationships using the single type "blocked-by":
  `from_issue` is blocked by `to_issue` (i.e. `to_issue` must be done first).
- Include ONLY semantically inferred new relationships — do NOT repeat
  relationships already recorded in GitHub (labelled "source: github" in the
  input).
- Explicit textual references in issue bodies may have high confidence (>= 0.9).
- Inferred semantic relationships typically have confidence 0.5 – 0.85.

Guidelines for `sprint_plan`
------------------------------
- Respect the sprint capacity limit stated in the user message.
- Prioritise: critical bugs > security issues > issues that unblock others >
  high-value features > nice-to-haves.
- An issue that is blocked by another open issue should NOT be placed in a
  sprint before the issue that blocks it.
- Include ALL open issues exactly once: either in a sprint's `issues` list OR
  in a sprint's `deferred` list.  Use the last sprint's `deferred` for issues
  that do not fit any sprint.

Guidelines for `label_recommendations`
----------------------------------------
- Suggest ONLY priority / status labels such as "priority: critical",
  "priority: high", "priority: medium", "priority: low", "good first issue".
- Do NOT suggest blocking/dependency labels — those are handled natively by
  GitHub's dependency API.
- Do NOT suggest removing existing labels.

Be factual. Do not invent requirements not present in the issue data.
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

        Also reads existing GitHub blocking/blocked-by relationships so the
        LLM receives the full dependency picture and avoids duplicating them.

        Parameters
        ----------
        owner, repo:
            Repository coordinates.
        """
        logger.debug("Fetching open issues for %s/%s", owner, repo)
        issues = self._gh.get_open_issues(owner, repo)
        prs = self._gh.get_open_pull_requests(owner, repo)
        existing_deps = self._fetch_existing_deps(owner, repo, issues)
        return self._plan(owner, repo, issues, prs, existing_deps)

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
        all_existing_deps: list[IssueDependency] = []
        for repo_name in repo_names:
            logger.debug("Fetching open issues/PRs for %s/%s", owner, repo_name)
            issues = self._gh.get_open_issues(owner, repo_name)
            prs = self._gh.get_open_pull_requests(owner, repo_name)
            existing_deps = self._fetch_existing_deps(owner, repo_name, issues)
            all_issues.extend(issues)
            all_prs.extend(prs)
            all_existing_deps.extend(existing_deps)
        return self._plan(owner, "*", all_issues, all_prs, all_existing_deps)

    def apply_relationships(
        self,
        owner: str,
        repo: str,
        result: AgilePlanResult,
        dry_run: bool = True,
    ) -> list[dict]:
        """Write new inferred blocking relationships back to GitHub's dependency API.

        Only processes dependencies with ``source == "llm"`` — relationships
        already sourced from GitHub (``source == "explicit"`` or
        ``source == "github"``) are skipped to avoid duplicates.

        Uses ``POST /repos/{owner}/{repo}/issues/{issue_number}/dependencies/blocked_by``
        with ``{"issue_id": <github_internal_id>}``.

        Parameters
        ----------
        owner, repo:
            Repository coordinates (used as the fallback when an issue's
            ``repo`` field cannot be parsed).
        result:
            The :class:`~git_review.models.AgilePlanResult` from
            :meth:`analyse` or :meth:`analyse_org`.
        dry_run:
            When *True* (default), log what *would* happen but make no API
            calls and return an empty list.

        Returns
        -------
        list[dict]
            GitHub API responses for each new relationship created.  Empty
            when ``dry_run=True``.
        """
        # Build number → (owner, repo_name, github_id) lookup from result issues
        issue_info: dict[int, tuple[str, str, int]] = {}
        for issue in result.issues:
            if issue.github_id is None:
                continue
            parts = issue.repo.split("/", 1)
            if len(parts) == 2:
                issue_info[issue.number] = (parts[0], parts[1], issue.github_id)

        responses: list[dict] = []
        for dep in result.dependencies:
            if dep.source not in ("llm",):
                continue
            # dep_type is always "blocked-by": from_issue is blocked by to_issue
            blocked_num = dep.from_issue
            blocker_num = dep.to_issue

            blocker_info = issue_info.get(blocker_num)
            blocked_info = issue_info.get(blocked_num)
            if blocker_info is None or blocked_info is None:
                logger.warning(
                    "Skipping dependency #%d blocked-by #%d — issue info missing",
                    blocked_num,
                    blocker_num,
                )
                continue

            blocked_owner, blocked_repo_name, _ = blocked_info
            _, _, blocker_github_id = blocker_info

            if dry_run:
                logger.info(
                    "[dry-run] Would record: #%d (%s/%s) is blocked by #%d (github_id=%d)",
                    blocked_num,
                    blocked_owner,
                    blocked_repo_name,
                    blocker_num,
                    blocker_github_id,
                )
            else:
                try:
                    resp = self._gh.add_issue_blocked_by(
                        blocked_owner,
                        blocked_repo_name,
                        blocked_num,
                        blocker_github_id,
                    )
                    responses.append(resp)
                except Exception as exc:
                    logger.warning(
                        "Failed to record #%d blocked-by #%d: %s",
                        blocked_num,
                        blocker_num,
                        exc,
                    )
        return responses

    def apply_labels(
        self,
        owner: str,
        repo: str,
        result: AgilePlanResult,
        dry_run: bool = True,
    ) -> list[dict]:
        """Apply priority label recommendations from *result* back to GitHub.

        This only applies priority/status labels (e.g. ``"priority: high"``).
        Blocking relationships are handled separately by :meth:`apply_relationships`.

        Parameters
        ----------
        owner, repo:
            Repository coordinates (fallback when an issue's ``repo`` field
            cannot be parsed).
        result:
            The :class:`~git_review.models.AgilePlanResult` to read
            ``label_recommendations`` from.
        dry_run:
            When *True* (default), log changes but make no API calls.
        """
        responses: list[dict] = []
        for issue in result.issues:
            new_labels = result.label_recommendations.get(issue.number)
            if not new_labels:
                continue
            merged = sorted(set(issue.labels) | set(new_labels))
            if set(merged) == set(issue.labels):
                continue
            parts = issue.repo.split("/", 1)
            issue_owner = parts[0] if len(parts) == 2 else owner
            issue_repo_name = parts[1] if len(parts) == 2 else repo
            if dry_run:
                logger.info(
                    "[dry-run] Would set labels on %s#%d: %s",
                    issue.repo,
                    issue.number,
                    merged,
                )
            else:
                try:
                    resp = self._gh.update_issue_labels(
                        issue_owner, issue_repo_name, issue.number, merged
                    )
                    responses.append(resp)
                except Exception as exc:
                    logger.warning(
                        "Failed to update labels for #%d: %s", issue.number, exc
                    )
        return responses

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_existing_deps(
        self,
        owner: str,
        repo: str,
        issues: list[Issue],
    ) -> list[IssueDependency]:
        """Fetch existing GitHub blocking/blocked-by relationships for *issues*.

        Makes two API calls per issue (blocked_by + blocking) concurrently.
        Errors are silently swallowed so that missing permissions or a repo
        that hasn't adopted the dependency feature yet doesn't break the
        planning flow.
        """
        issue_numbers = {i.number for i in issues}
        deps: list[IssueDependency] = []

        def _fetch_blocked_by(number: int) -> list[IssueDependency]:
            result: list[IssueDependency] = []
            try:
                raw = self._gh.get_issue_blocked_by(owner, repo, number)
                for item in raw:
                    blocker_num = item.get("number")
                    if blocker_num and blocker_num in issue_numbers:
                        result.append(
                            IssueDependency(
                                from_issue=number,
                                to_issue=blocker_num,
                                dep_type="blocked-by",
                                confidence=1.0,
                                reason=f"#{number} is blocked by #{blocker_num} (recorded in GitHub)",
                                source="github",
                            )
                        )
            except Exception as exc:
                logger.debug("Could not fetch blocked_by for #%d: %s", number, exc)
            return result

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_fetch_blocked_by, i.number): i.number for i in issues}
            for future in as_completed(futures):
                try:
                    deps.extend(future.result())
                except Exception:
                    pass

        return deps

    def _plan(
        self,
        owner: str,
        repo: str,
        issues: list[Issue],
        prs: list[PullRequest],
        existing_deps: list[IssueDependency],
    ) -> AgilePlanResult:
        """Run the full analysis pipeline and return an :class:`AgilePlanResult`."""
        # Collect explicit text-based dependencies (regex scan of bodies)
        text_deps = _extract_explicit_dependencies(issues, prs)

        # De-duplicate: existing GitHub deps take precedence
        seen: set[tuple[int, int]] = {
            (d.from_issue, d.to_issue) for d in existing_deps
        }
        for dep in text_deps:
            key = (dep.from_issue, dep.to_issue)
            if key not in seen:
                seen.add(key)
                existing_deps.append(dep)

        # Ask the LLM for inferred deps and sprint plan
        user_message = self._build_context_message(issues, prs, existing_deps)
        llm_result = self._call_llm(user_message)

        # Merge LLM deps (skip any already in GitHub or text)
        all_deps: list[IssueDependency] = list(existing_deps)
        for ld in llm_result.dependencies:
            key = (ld.from_issue, ld.to_issue)
            if key not in seen:
                seen.add(key)
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
        existing_deps: list[IssueDependency],
    ) -> str:
        """Format open issues, PRs, and existing dependencies as a text blob for the LLM."""
        issue_map = {i.number: i for i in issues}

        # Map issue number → PR titles referencing it
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
        ]

        if existing_deps:
            lines.append("## Existing Blocking Relationships (already recorded in GitHub)")
            for dep in existing_deps:
                lines.append(
                    f"- #{dep.from_issue} is blocked-by #{dep.to_issue}"
                    f"  (source: {dep.source}, reason: {dep.reason})"
                )
            lines.append("")

        lines.append("## Open Issues")
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
    """Scan issue and PR bodies for explicit cross-references.

    All relationships are normalised to the ``"blocked-by"`` type:
    ``from_issue`` is blocked by ``to_issue``.
    """
    issue_numbers = {i.number for i in issues}
    deps: list[IssueDependency] = []

    def _add(from_num: int, to_num: int, reason: str) -> None:
        if from_num != to_num and to_num in issue_numbers:
            deps.append(
                IssueDependency(
                    from_issue=from_num,
                    to_issue=to_num,
                    dep_type="blocked-by",
                    confidence=1.0,
                    reason=reason,
                    source="explicit",
                )
            )

    for item in [*issues, *prs]:
        num = item.number
        body = item.body or ""
        for m in _BLOCKS_RE.finditer(body):
            # "#{num} blocks #{ref}" → ref is blocked by num
            ref = int(m.group(1))
            _add(ref, num, f"#{num} explicitly states it blocks #{ref}")
        for m in _BLOCKED_BY_RE.finditer(body):
            _add(num, int(m.group(1)), f"#{num} states it is blocked by #{m.group(1)}")
        for m in _DEPENDS_ON_RE.finditer(body):
            _add(num, int(m.group(1)), f"#{num} depends on #{m.group(1)}")
        for m in _CLOSES_RE.finditer(body):
            ref = int(m.group(1))
            if ref in issue_numbers:
                # PR closes issue → PR is blocked by the issue being open
                # But more usefully, note the PR addresses the issue
                _add(num, ref, f"PR #{num} closes issue #{ref}")

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

