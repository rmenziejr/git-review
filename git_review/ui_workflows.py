"""Shared workflow helpers used by the Gradio and Reflex UIs."""

from __future__ import annotations

import json
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from .agile_planner import AgilePlanner
from .agile_planner import resolve_agile_target as _resolve_agile_target_from_input
from .github_client import GitHubClient
from .github_source_sync import (
    GitHubSourceOfTruthSync,
    ServiceNowClient,
    get_repo_cursor,
    load_sync_cursor,
    save_sync_cursor,
    set_repo_cursor,
)
from .issue_factory import IssueDraft, IssueFactory
from .llm_client import LLMClient
from .models import ReviewSummary


def _make_clients(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
) -> tuple[GitHubClient, IssueFactory]:
    gh = GitHubClient(token=github_token or None)
    factory = IssueFactory(
        github_client=gh,
        openai_api_key=openai_key or None,
        model=model or "gpt-4o-mini",
        base_url=base_url or None,
    )
    return gh, factory


def _drafts_to_table(drafts: list[IssueDraft]) -> list[list[Any]]:
    return [
        [
            i + 1,
            d.title,
            d.body,
            ", ".join(d.labels),
            ", ".join(d.assignees),
            str(d.milestone) if d.milestone is not None else "",
        ]
        for i, d in enumerate(drafts)
    ]


def _table_to_drafts(rows: list[list[Any]]) -> list[IssueDraft]:
    drafts: list[IssueDraft] = []
    for row in rows:
        title = str(row[1]).strip()
        body = str(row[2]).strip()
        if not title:
            continue
        labels_raw = str(row[3]) if len(row) > 3 else ""
        assignees_raw = str(row[4]) if len(row) > 4 else ""
        milestone_raw = str(row[5]).strip() if len(row) > 5 else ""
        labels = [lbl.strip() for lbl in labels_raw.split(",") if lbl.strip()]
        assignees = [a.strip() for a in assignees_raw.split(",") if a.strip()]
        milestone: Optional[int] = None
        if milestone_raw:
            try:
                milestone = int(milestone_raw)
            except ValueError:
                pass
        drafts.append(
            IssueDraft(
                title=title,
                body=body,
                labels=labels,
                assignees=assignees,
                milestone=milestone,
            )
        )
    return drafts


@dataclass
class MilestoneSeed:
    """A milestone definition that can be queued before GitHub creation."""

    title: str
    description: str = ""
    due_on: str = ""
    state: str = "open"


def _normalize_milestone_state(state: str) -> str:
    value = (state or "").strip().lower() or "open"
    if value not in {"open", "closed"}:
        raise ValueError(f"Invalid milestone state '{state}'. Use 'open' or 'closed'.")
    return value


def _to_due_on_iso(due_on: str) -> Optional[str]:
    if not (due_on or "").strip():
        return None
    try:
        due_dt = datetime.strptime(due_on.strip(), "%Y-%m-%d").replace(
            hour=0,
            minute=0,
            second=0,
            tzinfo=timezone.utc,
        )
    except ValueError as exc:
        raise ValueError(f"Invalid due date '{due_on}'. Use YYYY-MM-DD format.") from exc
    return due_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _validate_due_on(due_on: str) -> None:
    _to_due_on_iso(due_on)


def serialize_milestone_seeds(seeds: list[MilestoneSeed]) -> str:
    return "\n".join(
        " | ".join(
            [
                seed.title.strip(),
                seed.due_on.strip(),
                seed.state.strip() or "open",
                " ".join(seed.description.splitlines()).strip(),
            ]
        ).rstrip()
        for seed in seeds
    )


def parse_default_milestones_json(raw_json: str) -> list[MilestoneSeed]:
    payload = (raw_json or "").strip()
    if not payload:
        return []
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as exc:
        raise ValueError("DEFAULT_MILESTONES_JSON must contain valid JSON.") from exc

    if not isinstance(data, list):
        raise ValueError("DEFAULT_MILESTONES_JSON must be a JSON array.")

    seeds: list[MilestoneSeed] = []
    for index, item in enumerate(data, start=1):
        if not isinstance(item, dict):
            raise ValueError(
                f"DEFAULT_MILESTONES_JSON item {index} must be an object with milestone fields."
            )
        title = str(item.get("title", "")).strip()
        if not title:
            raise ValueError(f"DEFAULT_MILESTONES_JSON item {index} is missing a title.")
        due_on = str(item.get("due_on", "")).strip()
        if due_on:
            _validate_due_on(due_on)
        seeds.append(
            MilestoneSeed(
                title=title,
                description=str(item.get("description", "")).strip(),
                due_on=due_on,
                state=_normalize_milestone_state(str(item.get("state", "open"))),
            )
        )
    return seeds


def load_default_milestones_text(raw_json: str) -> tuple[str, str]:
    seeds = parse_default_milestones_json(raw_json)
    if not seeds:
        return "", "ℹ️  No default milestones configured in DEFAULT_MILESTONES_JSON."
    return (
        serialize_milestone_seeds(seeds),
        f"✅  Loaded {len(seeds)} default milestone(s) from DEFAULT_MILESTONES_JSON.",
    )


def append_milestone_to_batch(
    batch_text: str,
    title: str,
    description: str,
    due_on: str,
    state: str,
) -> tuple[str, str]:
    title_value = (title or "").strip()
    if not title_value:
        return batch_text, "❌  Milestone title is required before adding it to the queue."

    due_on_value = (due_on or "").strip()
    try:
        if due_on_value:
            _to_due_on_iso(due_on_value)
        seed = MilestoneSeed(
            title=title_value,
            description=(description or "").strip(),
            due_on=due_on_value,
            state=_normalize_milestone_state(state),
        )
    except ValueError as exc:
        return batch_text, f"❌  {exc}"

    line = serialize_milestone_seeds([seed])
    next_batch = f"{batch_text.rstrip()}\n{line}".strip() if batch_text.strip() else line
    return next_batch, f"✅  Added '{title_value}' to the milestone queue."


def parse_milestone_batch_text(batch_text: str) -> list[MilestoneSeed]:
    seeds: list[MilestoneSeed] = []
    for line_number, raw_line in enumerate((batch_text or "").splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue

        parts = [part.strip() for part in line.split("|")]
        title = ""
        description = ""
        due_on = ""
        state = "open"

        if len(parts) == 1:
            title = parts[0]
        elif len(parts) == 2:
            title, description = parts
        elif len(parts) == 3:
            title, due_on, third = parts
            if third.strip().lower() in {"open", "closed"}:
                state = third
            else:
                description = third
        else:
            title = parts[0]
            due_on = parts[1]
            state = parts[2] or "open"
            description = " | ".join(parts[3:]).strip()

        title = title.strip()
        if not title:
            raise ValueError(f"Line {line_number}: milestone title is required.")
        if due_on:
            _validate_due_on(due_on)
        seeds.append(
            MilestoneSeed(
                title=title,
                description=description.strip(),
                due_on=due_on.strip(),
                state=_normalize_milestone_state(state),
            )
        )

    return seeds


def create_milestones_batch(
    github_token: str,
    repo: str,
    batch_text: str,
) -> tuple[str, list[dict[str, Any]]]:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format.", []

    try:
        seeds = parse_milestone_batch_text(batch_text)
    except ValueError as exc:
        return f"❌  {exc}", []

    if not seeds:
        return "❌  Add at least one milestone to the queue before creating them.", []

    owner, repo_name = repo.strip().split("/", 1)
    gh = GitHubClient(token=github_token or None)
    created: list[dict[str, Any]] = []
    errors: list[str] = []

    for seed in seeds:
        try:
            result = gh.create_milestone(
                owner=owner,
                repo=repo_name,
                title=seed.title,
                description=seed.description,
                due_on=_to_due_on_iso(seed.due_on),
                state=seed.state,
            )
            created.append(result)
        except Exception as exc:
            errors.append(f"- {seed.title}: {exc}")

    if created and not errors:
        status_prefix = "✅"
    elif created:
        status_prefix = "⚠️"
    else:
        status_prefix = "❌"

    lines = [
        f"{status_prefix}  Created {len(created)} of {len(seeds)} milestone(s) for {owner}/{repo_name}."
    ]
    for result in created:
        lines.append(
            f"- #{result.get('number', '?')} {result.get('title', '')} {result.get('html_url', '')}".rstrip()
        )
    if errors:
        lines.extend(["", "Errors:"])
        lines.extend(errors)
    return "\n".join(lines), created


def summarize_activity(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
    repo: str,
    days: int,
    since_str: str,
    until_str: str,
    author: str,
    system_prompt: str,
    all_repos: bool = False,
) -> tuple[str, str]:
    repo = (repo or "").strip()
    if not repo:
        label = "owner name" if all_repos else "'owner/repo' format"
        return "", f"❌  Please enter the repository in {label}."

    if all_repos:
        owner = repo.split("/", 1)[0].strip()
        repo_label = "*"
    else:
        if "/" not in repo:
            return "", "❌  Please enter the repository in 'owner/repo' format."
        owner, repo_label = repo.split("/", 1)

    effective_key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not (base_url or "").strip():
        return "", "❌  An OpenAI API key is required. Enter it above or set OPENAI_API_KEY."

    now_utc = datetime.now(tz=timezone.utc).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    try:
        since = (
            datetime.strptime(since_str.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if since_str.strip()
            else (now_utc - timedelta(days=max(1, int(days)))).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        )
        until = (
            datetime.strptime(until_str.strip(), "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            if until_str.strip()
            else now_utc
        )
    except ValueError as exc:
        return "", f"❌  Invalid date: {exc}"

    if since > until:
        return "", "❌  'Since' date must be earlier than 'Until' date."

    gh = GitHubClient(token=github_token or None)
    review_data = ReviewSummary(owner=owner, repo=repo_label, since=since, until=until)
    errors: list[str] = []
    author_filter = author.strip() or None
    section_to_attr = {
        "commits": "commits",
        "issues": "issues",
        "pull_requests": "pull_requests",
        "releases": "releases",
        "contributors": "contributors",
    }

    if all_repos:
        try:
            repo_names = gh.list_repos(owner)
        except Exception as exc:
            return "", f"❌  Could not list repositories for '{owner}': {exc}"
        if not repo_names:
            return "", f"⚠️  No non-archived repositories found for '{owner}'."
    else:
        repo_names = [repo_label]

    for repo_name in repo_names:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    gh.get_commits,
                    owner,
                    repo_name,
                    since,
                    until,
                    author=author_filter,
                    include_stats=True,
                ): "commits",
                executor.submit(gh.get_issues, owner, repo_name, since, until): "issues",
                executor.submit(
                    gh.get_pull_requests,
                    owner,
                    repo_name,
                    since,
                    until,
                    include_details=True,
                ): "pull_requests",
                executor.submit(gh.get_releases, owner, repo_name, since, until): "releases",
                executor.submit(gh.get_contributors, owner, repo_name): "contributors",
            }

            for future in as_completed(futures):
                data_label = futures[future]
                try:
                    getattr(review_data, section_to_attr[data_label]).extend(future.result())
                except Exception as exc:
                    display_label = data_label.replace("_", " ")
                    errors.append(
                        f"⚠️  Could not fetch {display_label} for {owner}/{repo_name}: {exc}"
                    )

    try:
        llm = LLMClient(
            api_key=effective_key or None,
            model=(model or "").strip() or "gpt-4o-mini",
            base_url=(base_url or "").strip() or None,
            system_prompt=system_prompt.strip() or None,
        )
        summary_text = llm.summarise(review_data)
    except Exception as exc:
        return "", f"❌  Error generating summary: {exc}"

    repo_display = f"{owner}/* ({len(repo_names)} repos)" if all_repos else f"{owner}/{repo_label}"
    status = f"✅  Summary generated for {repo_display} ({since.date()} → {until.date()})."
    if errors:
        status += "\n" + "\n".join(errors)
    return summary_text, status


def create_milestone(
    github_token: str,
    repo: str,
    title: str,
    description: str,
    due_on: str,
    state: str,
) -> str:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."
    if not title.strip():
        return "❌  Milestone title is required."

    owner, repo_name = repo.strip().split("/", 1)
    try:
        due_on_iso = _to_due_on_iso(due_on)
        state_value = _normalize_milestone_state(state)
    except ValueError as exc:
        return f"❌  {exc}"

    gh = GitHubClient(token=github_token or None)
    try:
        result = gh.create_milestone(
            owner=owner,
            repo=repo_name,
            title=title.strip(),
            description=description.strip(),
            due_on=due_on_iso,
            state=state_value,
        )
    except Exception as exc:
        return f"❌  Error creating milestone: {exc}"

    number = result.get("number", "?")
    url = result.get("html_url", "")
    return f"✅  Milestone #{number} '{title.strip()}' created successfully.\n{url}"


def list_milestones(github_token: str, repo: str, state: str) -> str:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."

    owner, repo_name = repo.strip().split("/", 1)
    gh = GitHubClient(token=github_token or None)
    try:
        milestones = gh.list_milestones(owner, repo_name, state=state)
    except Exception as exc:
        return f"❌  Error listing milestones: {exc}"

    if not milestones:
        return f"No milestones found (state={state})."

    lines = [f"Found {len(milestones)} milestone(s):\n"]
    for milestone in milestones:
        due = str(milestone.due_on.date()) if milestone.due_on else "—"
        lines.append(
            f"  #{milestone.number}  {milestone.title}  (state={milestone.state}, due={due}, "
            f"open={milestone.open_issues}, closed={milestone.closed_issues})"
        )
    return "\n".join(lines)


def sync_servicenow(
    github_token: str,
    repo: str,
    servicenow_url: str,
    servicenow_user: str,
    servicenow_password: str,
    servicenow_token: str,
    milestone_table: str,
    issue_table: str,
    cursor_path: str,
    dry_run: bool,
    allow_back_sync_fields: str,
) -> str:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."
    if not servicenow_url.strip():
        return "❌  ServiceNow URL is required."
    if not (
        servicenow_token.strip()
        or (servicenow_user.strip() and servicenow_password.strip())
    ):
        return "❌  ServiceNow auth required: token or username/password."

    owner, repo_name = repo.strip().split("/", 1)
    allowed = {"labels", "assignees"}
    back_sync_fields = tuple(
        sorted(
            {
                field.strip().lower()
                for field in (allow_back_sync_fields or "").split(",")
                if field.strip().lower() in allowed
            }
        )
    )

    gh = GitHubClient(token=github_token or None)
    try:
        snow = ServiceNowClient(
            servicenow_url.strip(),
            user=servicenow_user.strip() or None,
            password=servicenow_password or None,
            token=servicenow_token.strip() or None,
        )
        syncer = GitHubSourceOfTruthSync(
            gh,
            snow,
            milestone_table=(milestone_table or "").strip() or "u_github_milestone",
            issue_table=(issue_table or "").strip() or "u_github_issue",
        )
        repo_key = f"{owner}/{repo_name}"
        cursor_file = os.path.realpath(
            (cursor_path or "").strip() or ".git-review-sync-cursor.json"
        )
        cwd = os.path.realpath(os.getcwd())
        if os.path.commonpath([cwd, cursor_file]) != cwd:
            return "❌  Cursor file path must be inside the current working directory."
        cursor_data = load_sync_cursor(cursor_file)
        since = get_repo_cursor(cursor_data, repo_key)
        report, next_cursor = syncer.sync_repo(
            owner,
            repo_name,
            since=since,
            dry_run=bool(dry_run),
            allow_back_sync_fields=back_sync_fields,
        )
        if not dry_run:
            set_repo_cursor(cursor_data, repo_key, next_cursor)
            save_sync_cursor(cursor_file, cursor_data)
    except Exception as exc:
        return f"❌  Error syncing ServiceNow: {exc}"

    lines = [
        f"✅  Sync {'previewed' if dry_run else 'applied'} for {owner}/{repo_name}.",
        f"Milestones: scanned={report.milestones_scanned}, created={report.milestones_created}, updated={report.milestones_updated}",
        f"Issues: scanned={report.issues_scanned}, created={report.issues_created}, updated={report.issues_updated}",
        f"Back-sync updates: {report.back_sync_updates}",
        f"Conflicts: {len(report.conflicts)}",
        f"Next cursor: {next_cursor.isoformat()}",
    ]
    if report.conflicts:
        lines.extend(["", "Conflict samples:"])
        for conflict in report.conflicts[:10]:
            lines.append(
                f"- {conflict.entity} {conflict.github_key} {conflict.field}: "
                f"ServiceNow='{conflict.servicenow_value}' vs GitHub='{conflict.github_value}'"
            )
        if len(report.conflicts) > 10:
            lines.append(f"... and {len(report.conflicts) - 10} more")
    if dry_run:
        lines.extend(["", "Dry-run enabled: no ServiceNow writes and cursor not advanced."])
    return "\n".join(lines)


_DRAFT_COLUMNS = ["#", "Title", "Body", "Labels", "Assignees", "Milestone #"]
_EMPTY_TABLE: list[list[Any]] = []


def fetch_requirements_from_repo(
    github_token: str,
    repo: str,
    path: str,
) -> tuple[str, str]:
    if not repo or "/" not in repo:
        return "", "❌  Please enter the repository in 'owner/repo' format."
    path = (path or "docs/requirements.md").strip()
    if not path:
        return "", "❌  Please enter a file path."

    owner, repo_name = repo.strip().split("/", 1)
    gh = GitHubClient(token=github_token or None)
    try:
        content = gh.get_file_content(owner, repo_name, path)
    except Exception as exc:
        return "", f"❌  Error fetching file: {exc}"
    return content, f"✅  Fetched '{path}' from {owner}/{repo_name} ({len(content)} chars)."


def parse_requirements(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
    requirements_text: str,
    requirements_file: Any = None,
    use_milestones: bool = False,
    milestones_repo: str = "",
) -> tuple[list[list[Any]], str]:
    if requirements_file is not None:
        try:
            real_file = os.path.realpath(str(requirements_file))
            real_tmp = os.path.realpath(tempfile.gettempdir())
            if not real_file.startswith(real_tmp + os.sep):
                return _EMPTY_TABLE, "❌  Uploaded file path is outside the allowed directory."
            with open(real_file, encoding="utf-8") as file_handle:
                requirements_text = file_handle.read()
        except Exception as exc:
            return _EMPTY_TABLE, f"❌  Error reading uploaded file: {exc}"

    if not requirements_text.strip():
        return _EMPTY_TABLE, "❌  Provide requirements text or upload a markdown file."

    effective_key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not (base_url or "").strip():
        return _EMPTY_TABLE, "❌  An OpenAI API key is required. Enter it above or set OPENAI_API_KEY."

    milestones = None
    if use_milestones and milestones_repo and "/" in milestones_repo:
        try:
            ms_owner, ms_repo_name = milestones_repo.strip().split("/", 1)
            gh_ms = GitHubClient(token=github_token or None)
            milestones = gh_ms.list_milestones(ms_owner, ms_repo_name, state="open") or None
        except Exception:
            milestones = None

    try:
        _, factory = _make_clients(github_token, effective_key or "", model, base_url)
        drafts = factory.parse_requirements(requirements_text, milestones=milestones)
    except Exception as exc:
        return _EMPTY_TABLE, f"❌  Error parsing requirements: {exc}"

    if not drafts:
        return _EMPTY_TABLE, "⚠️  No issues were extracted from the requirements document."

    milestone_note = (
        f" (with {len(milestones)} milestone(s) as context)"
        if milestones is not None and len(milestones) > 0
        else ""
    )
    return (
        _drafts_to_table(drafts),
        f"✅  Extracted {len(drafts)} issue draft(s){milestone_note}. Review and edit below.",
    )


def submit_issues(
    github_token: str,
    repo: str,
    milestone_override: str,
    draft_rows: Any,
) -> str:
    try:
        import pandas as pd

        if isinstance(draft_rows, pd.DataFrame):
            draft_rows = draft_rows.values.tolist()
    except ImportError:
        pass

    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."
    if not draft_rows:
        return "❌  No issue drafts to submit. Parse requirements first."

    owner, repo_name = repo.strip().split("/", 1)
    milestone_num: Optional[int] = None
    if str(milestone_override).strip():
        try:
            milestone_num = int(str(milestone_override).strip())
        except ValueError:
            return f"❌  Invalid milestone number '{milestone_override}'."

    drafts = _table_to_drafts(draft_rows)
    if not drafts:
        return "❌  All rows appear empty. Nothing to submit."

    gh = GitHubClient(token=github_token or None)
    factory = IssueFactory(github_client=gh, openai_api_key=None, model="gpt-4o-mini")
    lines = [f"Creating {len(drafts)} issue(s) in {owner}/{repo_name}…\n"]
    try:
        results = factory.push_issues(owner, repo_name, drafts, milestone=milestone_num)
    except Exception as exc:
        return f"❌  Error creating issues: {exc}"

    lines.append(f"✅  Created {len(results)} issue(s):\n")
    for result in results:
        lines.append(f"  #{result.get('number')}  {result.get('html_url', '')}")
    return "\n".join(lines)


def run_agile_planner(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
    repo: str,
    sprint_capacity: int,
    num_sprints: int,
    all_repos: bool,
) -> tuple[str, str, str]:
    try:
        owner, repo_name, org_mode = _resolve_agile_target_from_input(repo)
    except ValueError as exc:
        return "", "", f"❌  {exc}"

    effective_key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not (base_url or "").strip():
        return "", "", "❌  An OpenAI API key is required. Enter it above or set OPENAI_API_KEY."

    planner = AgilePlanner(
        github_client=GitHubClient(token=github_token or None),
        openai_api_key=effective_key or None,
        model=(model or "gpt-4o-mini").strip(),
        base_url=(base_url or "").strip() or None,
        sprint_capacity=max(1, int(sprint_capacity or 10)),
        num_sprints=max(1, int(num_sprints or 3)),
    )

    try:
        result = planner.analyse_org(owner) if org_mode else planner.analyse(owner, repo_name)
    except Exception as exc:
        return "", "", f"❌  Error running agile analysis: {exc}"

    deps_lines: list[str] = []
    if result.dependencies:
        deps_lines.extend(
            [
                "| Blocked Issue | Blocked By | Confidence | Source | Reason |",
                "|---------------|------------|------------|--------|--------|",
            ]
        )
        for dep in result.dependencies:
            dep_source = dep.source or "llm"
            deps_lines.append(
                f"| #{dep.from_issue} | #{dep.to_issue} | {dep.confidence:.0%} | {dep_source} | {dep.reason} |"
            )
    deps_md = "\n".join(deps_lines) if deps_lines else "_No dependencies identified._"

    plan_lines: list[str] = []
    issue_titles = {issue.number: issue.title for issue in result.issues}
    for sprint in result.sprints:
        plan_lines.extend(
            [
                f"### Sprint {sprint.sprint_number}: {sprint.theme}",
                sprint.rationale,
                "",
                "**Issues:**",
            ]
        )
        for number in sprint.issues:
            plan_lines.append(f"- #{number} {issue_titles.get(number, '')}")
        if sprint.deferred:
            plan_lines.extend(["", "**Deferred:**"])
            for number in sprint.deferred:
                plan_lines.append(f"- #{number} {issue_titles.get(number, '')}")
        plan_lines.append("")

    if result.summary_text:
        plan_lines.extend(["---", "**Summary**", "", result.summary_text])
    plan_md = "\n".join(plan_lines) if plan_lines else "_No sprint plan generated._"

    pr_label = "PR" if len(result.pull_requests) == 1 else "PRs"
    dependency_label = "dependency" if len(result.dependencies) == 1 else "dependencies"
    sprint_label = "sprint" if len(result.sprints) == 1 else "sprints"
    status = (
        f"✅  Plan generated: {len(result.issues)} issues, "
        f"{len(result.pull_requests)} {pr_label}, "
        f"{len(result.dependencies)} {dependency_label}, "
        f"{len(result.sprints)} {sprint_label}."
    )
    return deps_md, plan_md, status


def run_agile_planner_state(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
    repo: str,
    sprint_capacity: int,
    num_sprints: int,
    all_repos: bool,
) -> Any:
    try:
        owner, repo_name, org_mode = _resolve_agile_target_from_input(repo)
    except ValueError:
        return None

    effective_key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not (base_url or "").strip():
        return None

    planner = AgilePlanner(
        github_client=GitHubClient(token=github_token or None),
        openai_api_key=effective_key or None,
        model=(model or "gpt-4o-mini").strip(),
        base_url=(base_url or "").strip() or None,
        sprint_capacity=max(1, int(sprint_capacity or 10)),
        num_sprints=max(1, int(num_sprints or 3)),
    )

    try:
        return planner.analyse_org(owner) if org_mode else planner.analyse(owner, repo_name)
    except Exception:
        return None


def apply_agile_relationships(
    github_token: str,
    repo: str,
    all_repos: bool,
    result: Any,
) -> str:
    if result is None:
        return "❌  No plan available. Generate a plan first."

    try:
        owner, repo_name, _ = _resolve_agile_target_from_input(repo)
    except ValueError as exc:
        return f"❌  {exc}"

    new_relationships = [dependency for dependency in result.dependencies if dependency.source == "llm"]
    if not new_relationships:
        return "ℹ️  No new inferred relationships to record."

    planner = AgilePlanner(
        github_client=GitHubClient(token=github_token or None),
        openai_api_key="unused",
    )
    try:
        responses = planner.apply_relationships(owner, repo_name, result, dry_run=False)
    except Exception as exc:
        return f"❌  Error recording relationships: {exc}"

    return f"✅  Recorded {len(responses)} blocking relationship(s) in GitHub."


def apply_agile_labels(
    github_token: str,
    repo: str,
    all_repos: bool,
    result: Any,
) -> str:
    if result is None:
        return "❌  No plan available. Generate a plan first."

    try:
        owner, repo_name, _ = _resolve_agile_target_from_input(repo)
    except ValueError as exc:
        return f"❌  {exc}"

    label_issues = [issue for issue in result.issues if result.label_recommendations.get(issue.number)]
    if not label_issues:
        return "ℹ️  No priority label recommendations to apply."

    planner = AgilePlanner(
        github_client=GitHubClient(token=github_token or None),
        openai_api_key="unused",
    )
    try:
        responses = planner.apply_labels(owner, repo_name, result, dry_run=False)
    except Exception as exc:
        return f"❌  Error applying labels: {exc}"

    return f"✅  Updated labels on {len(responses)} issue(s)."
