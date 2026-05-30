"""Reflex application state for the git-review agent frontend.

State design
------------
* Reactive vars (sent to the browser): ``messages``, ``streaming_text``,
  ``is_thinking``, ``pending_hitl``, plus all settings fields.
* Backend-only vars (``_`` prefix, never serialised): ``_pending_result``
  stores the :class:`RunResultStreaming` so that HITL approve / deny can
  resume the run in the same server process.

Streaming flow
--------------
1.  ``send_message`` appends the user message, sets ``is_thinking=True``,
    and ``yield``\\s to push the update to the browser immediately.
2.  It calls :func:`~git_review.agent.run_agent_streaming` and iterates
    ``stream_events()`` in an ``async for`` loop.
3.  Answer text deltas are accumulated in ``streaming_text``; reasoning is
    accumulated separately and attached to the completed assistant message.
    Tool-call events are appended to ``messages``; each ``yield`` inside the
    loop pushes the incremental update live.
4.  After the stream ends the final assistant message is moved from
    ``streaming_text`` into ``messages``.
5.  If ``result.interruptions`` is non-empty, the HITL details are stored
    in ``pending_hitl`` and the raw result is kept in ``_pending_result``.

HITL flow
---------
* ``approve_hitl(idx)`` calls ``state.approve(item)`` on the stored result
  and re-runs the agent from the :class:`RunState`.
* ``deny_hitl(idx)`` calls ``state.reject(item)`` and appends a denied
  message, clearing the pending state.
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field, replace
from typing import Any, Optional

import reflex as rx

from git_review.agent import AgentContext, run_agent_streaming
from git_review.config import AppSettings
from git_review.ui_workflows import (
    apply_agile_labels,
    apply_agile_relationships,
    append_milestone_to_batch,
    create_project_for_owner,
    create_milestone,
    create_milestones_batch,
    fetch_requirements_from_repo,
    list_open_issues_for_repo,
    list_projects_for_target,
    list_repositories_for_owner,
    list_milestones,
    load_default_milestones_text,
    parse_requirements,
    read_agile_project_board,
    run_agile_planner,
    run_agile_planner_state,
    submit_issues,
    summarize_activity,
    sync_servicenow,
    update_agile_project_status,
)

try:
    from agents.items import ToolApprovalItem
    from agents.run import RunResultStreaming
    from agents.stream_events import RawResponsesStreamEvent, RunItemStreamEvent
except ImportError:  # pragma: no cover
    pass

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model types stored in reactive state
# ---------------------------------------------------------------------------


@dataclass
class ChatMessage:
    """A single entry in the chat history."""

    id: str
    role: str
    """'user' | 'assistant' | 'reasoning' | 'tool_call' | 'tool_result' | 'tool_error' | 'error'"""
    content: str
    reasoning_text: str = ""
    tool_name: str = ""
    args_json: str = ""
    is_error: bool = False
    tool_events: list["ToolEvent"] = field(default_factory=list)


@dataclass
class ToolEvent:
    """A lifecycle event attached to the active assistant bubble."""

    id: str
    kind: str
    tool_name: str
    args_json: str = ""
    content: str = ""
    is_error: bool = False


@dataclass
class HITLRequest:
    """A pending write-tool approval request surfaced in the HITL panel."""

    id: str
    tool_name: str
    args_json: str
    description: str


@dataclass
class RequirementDraft:
    """Editable issue draft shown in the requirements workflow."""

    title: str
    body: str
    labels: str = ""
    assignees: str = ""
    milestone: str = ""


_REQUIREMENT_DRAFT_FIELDS = frozenset(RequirementDraft.__annotations__)


# ---------------------------------------------------------------------------
# Application state
# ---------------------------------------------------------------------------


class AppState(rx.State):
    """Central reactive state for the agent chat UI."""

    # ---- Chat ----
    messages: list[ChatMessage] = []
    streaming_text: str = ""
    reasoning_text: str = ""
    is_thinking: bool = False
    processing_tool: bool = False

    # ---- HITL ----
    pending_hitl: list[HITLRequest] = []

    # ---- Input ----
    input_value: str = ""
    input_disabled: bool = False

    # ---- Settings ----
    github_token: str = ""
    openai_key: str = ""
    openai_base_url: str = ""
    agent_model: str = "gpt-4o"
    owner: str = ""
    repo: str = ""
    servicenow_enabled: bool = False
    servicenow_url: str = ""
    servicenow_user: str = ""
    servicenow_password: str = ""
    servicenow_token: str = ""
    servicenow_milestone_table: str = "u_github_milestone"
    servicenow_issue_table: str = "u_github_issue"
    servicenow_cursor_path: str = ".git-review-sync-cursor.json"
    settings_open: bool = False
    settings_repo_open: bool = True
    settings_credentials_open: bool = True
    settings_model_open: bool = False
    settings_servicenow_open: bool = False
    sidebar_collapsed: bool = False

    # ---- Workflow pages ----
    summary_repo: str = ""
    summary_all_repos: bool = False
    summary_author: str = ""
    summary_days: str = "7"
    summary_since: str = ""
    summary_until: str = ""
    summary_prompt: str = ""
    summary_status: str = ""
    summary_output: str = ""

    milestone_repo: str = ""
    milestone_title: str = ""
    milestone_description: str = ""
    milestone_due_on: str = ""
    milestone_state: str = "open"
    milestone_create_result: str = ""
    milestone_queue_text: str = ""
    milestone_defaults_status: str = ""
    milestone_list_repo: str = ""
    milestone_list_state: str = "open"
    milestone_list_output: str = ""

    sync_repo: str = ""
    sync_dry_run: bool = True
    sync_back_sync_fields: str = ""
    sync_result: str = ""

    requirements_repo: str = ""
    requirements_path: str = "docs/requirements.md"
    requirements_text: str = ""
    requirements_fetch_status: str = ""
    requirements_use_milestones: bool = False
    requirements_milestones_repo: str = ""
    requirements_status: str = ""
    requirements_milestone_status: str = ""
    requirement_drafts: list[RequirementDraft] = []
    submit_repo: str = ""
    submit_milestone_override: str = ""
    submit_status: str = ""
    submit_open_issues_markdown: str = ""
    submit_open_issues_status: str = ""

    agile_repo: str = ""
    agile_capacity: str = "10"
    agile_sprints: str = "3"
    agile_status: str = ""
    agile_dependencies_markdown: str = ""
    agile_plan_markdown: str = ""
    agile_apply_status: str = ""
    agile_project_number: str = ""
    agile_project_status_field: str = "Status"
    agile_project_sprint_number: str = ""
    agile_project_issue_number: str = ""
    agile_project_status_value: str = ""
    agile_project_board_markdown: str = ""
    agile_project_board_status: str = ""
    agile_repos_markdown: str = ""
    agile_projects_markdown: str = ""
    agile_context_status: str = ""
    agile_new_project_title: str = ""
    agile_open_issues_markdown: str = ""
    agile_open_issues_status: str = ""

    # ---- Backend-only (not sent to frontend) ----
    _pending_result: Any = None
    _conversation_history: list = []
    _agile_result: Any = None
    _tool_args_by_call_id: dict[str, str] = {}
    _active_tool_calls: set[str] = set()
    _tool_event_index_by_call_id: dict[str, int] = {}

    # ------------------------------------------------------------------ #
    # Lifecycle
    # ------------------------------------------------------------------ #

    def on_load(self) -> None:
        """Populate settings from environment / .env on first load."""
        settings = AppSettings()
        if settings.github_token:
            self.github_token = settings.github_token
        if settings.openai_api_key:
            self.openai_key = settings.openai_api_key
        if settings.openai_base_url:
            self.openai_base_url = settings.openai_base_url
        if settings.agent_model:
            self.agent_model = settings.agent_model
        self.servicenow_enabled = bool(settings.servicenow_enabled)
        if settings.servicenow_url:
            self.servicenow_url = settings.servicenow_url
        if settings.servicenow_user:
            self.servicenow_user = settings.servicenow_user
        if settings.servicenow_password:
            self.servicenow_password = settings.servicenow_password
        if settings.servicenow_token:
            self.servicenow_token = settings.servicenow_token
        if settings.servicenow_milestone_table:
            self.servicenow_milestone_table = settings.servicenow_milestone_table
        if settings.servicenow_issue_table:
            self.servicenow_issue_table = settings.servicenow_issue_table
        if settings.servicenow_cursor_path:
            self.servicenow_cursor_path = settings.servicenow_cursor_path
        if settings.default_milestones_json and not self.milestone_queue_text.strip():
            try:
                queue_text, status = load_default_milestones_text(
                    settings.default_milestones_json
                )
                self.milestone_queue_text = queue_text
                self.milestone_defaults_status = status
            except ValueError as exc:
                self.milestone_defaults_status = f"❌  {exc}"
        repo_value = self._settings_repo_value()
        if repo_value:
            for field_name in (
                "summary_repo",
                "milestone_repo",
                "milestone_list_repo",
                "sync_repo",
                "requirements_repo",
                "submit_repo",
                "agile_repo",
            ):
                if not getattr(self, field_name):
                    setattr(self, field_name, repo_value)
            if not self.requirements_milestones_repo:
                self.requirements_milestones_repo = repo_value

    # ------------------------------------------------------------------ #
    # Settings sidebar
    # ------------------------------------------------------------------ #

    def toggle_settings(self) -> None:
        self.settings_open = not self.settings_open

    def toggle_settings_section(self, section: str) -> None:
        section_name = section.strip().lower()
        field_name = f"settings_{section_name}_open"
        if hasattr(self, field_name):
            setattr(self, field_name, not bool(getattr(self, field_name)))

    def toggle_sidebar(self) -> None:
        self.sidebar_collapsed = not self.sidebar_collapsed

    def set_github_token(self, value: str) -> None:
        self.github_token = value

    def set_openai_key(self, value: str) -> None:
        self.openai_key = value

    def set_openai_base_url(self, value: str) -> None:
        self.openai_base_url = value

    def set_agent_model(self, value: str) -> None:
        self.agent_model = value

    def set_owner(self, value: str) -> None:
        self.owner = value

    def set_repo(self, value: str) -> None:
        self.repo = value

    def toggle_servicenow_enabled(self) -> None:
        self.servicenow_enabled = not self.servicenow_enabled

    def set_servicenow_url(self, value: str) -> None:
        self.servicenow_url = value

    def set_servicenow_user(self, value: str) -> None:
        self.servicenow_user = value

    def set_servicenow_password(self, value: str) -> None:
        self.servicenow_password = value

    def set_servicenow_token(self, value: str) -> None:
        self.servicenow_token = value

    def set_servicenow_milestone_table(self, value: str) -> None:
        self.servicenow_milestone_table = value

    def set_servicenow_issue_table(self, value: str) -> None:
        self.servicenow_issue_table = value

    def set_servicenow_cursor_path(self, value: str) -> None:
        self.servicenow_cursor_path = value

    def set_input_value(self, value: str) -> None:
        self.input_value = value

    # ------------------------------------------------------------------ #
    # Workflow helpers
    # ------------------------------------------------------------------ #

    def _settings_repo_value(self) -> str:
        owner = self.owner.strip()
        repo = self.repo.strip()
        if owner and repo:
            return f"{owner}/{repo}"
        return ""

    def set_workflow_field(self, field_name: str, value: Any) -> None:
        setattr(self, field_name, value)

    def toggle_workflow_flag(self, field_name: str) -> None:
        setattr(self, field_name, not bool(getattr(self, field_name)))

    def use_settings_repo(self, field_name: str) -> None:
        repo_value = self._settings_repo_value()
        if repo_value:
            setattr(self, field_name, repo_value)

    def sync_submit_repo_from_requirements(self) -> None:
        if self.requirements_repo.strip():
            self.submit_repo = self.requirements_repo.strip()

    def update_requirement_draft(self, value: str, idx: int, field_name: str) -> None:
        if field_name not in _REQUIREMENT_DRAFT_FIELDS:
            logger.warning(
                "Ignoring requirement draft update for unknown field: %s", field_name
            )
            return
        drafts = list(self.requirement_drafts)
        if not 0 <= idx < len(drafts):
            logger.warning(
                "Ignoring requirement draft update for out-of-range index %s (draft count=%s)",
                idx,
                len(drafts),
            )
            return
        updated = replace(drafts[idx])
        setattr(updated, field_name, value)
        drafts[idx] = updated
        self.requirement_drafts = drafts

    def clear_requirement_drafts(self) -> None:
        self.requirement_drafts = []
        self.requirements_status = ""
        self.submit_status = ""

    def load_default_milestones(self) -> None:
        settings = AppSettings()
        try:
            queue_text, status = load_default_milestones_text(
                settings.default_milestones_json
            )
        except ValueError as exc:
            self.milestone_defaults_status = f"❌  {exc}"
            return
        self.milestone_queue_text = queue_text
        self.milestone_defaults_status = status

    def queue_current_milestone(self) -> None:
        next_queue, status = append_milestone_to_batch(
            self.milestone_queue_text,
            self.milestone_title,
            self.milestone_description,
            self.milestone_due_on,
            self.milestone_state,
        )
        self.milestone_queue_text = next_queue
        self.milestone_create_result = status
        if status.startswith("✅"):
            self.milestone_title = ""
            self.milestone_description = ""
            self.milestone_due_on = ""

    def clear_milestone_queue(self) -> None:
        self.milestone_queue_text = ""
        self.milestone_create_result = ""

    def use_milestones_for_requirements(self) -> None:
        repo_value = (
            self.requirements_milestones_repo or self.requirements_repo
        ).strip()
        if repo_value:
            self.requirements_milestones_repo = repo_value
            self.requirements_use_milestones = True

    def _draft_rows(self) -> list[list[str]]:
        return [
            [
                str(index + 1),
                draft.title,
                draft.body,
                draft.labels,
                draft.assignees,
                draft.milestone,
            ]
            for index, draft in enumerate(self.requirement_drafts)
        ]

    def _hydrate_requirement_drafts(self, rows: list[list[Any]]) -> None:
        self.requirement_drafts = [
            RequirementDraft(
                title=str(row[1]) if len(row) > 1 else "",
                body=str(row[2]) if len(row) > 2 else "",
                labels=str(row[3]) if len(row) > 3 else "",
                assignees=str(row[4]) if len(row) > 4 else "",
                milestone=str(row[5]) if len(row) > 5 else "",
            )
            for row in rows
        ]

    def _safe_int(self, raw: str, default: int, min_value: int = 1) -> int:
        try:
            return max(min_value, int((raw or "").strip() or str(default)))
        except ValueError:
            return default

    async def generate_summary(self) -> None:
        self.summary_status = "Working…"
        self.summary_output = ""
        yield
        output, status = summarize_activity(
            self.github_token,
            self.openai_key,
            self.agent_model,
            self.openai_base_url,
            self.summary_repo,
            self._safe_int(self.summary_days, 7),
            self.summary_since,
            self.summary_until,
            self.summary_author,
            self.summary_prompt,
            self.summary_all_repos,
        )
        self.summary_output = output
        self.summary_status = status
        yield

    async def create_milestone_workflow(self) -> None:
        self.milestone_create_result = "Working…"
        yield
        self.milestone_create_result = create_milestone(
            self.github_token,
            self.milestone_repo,
            self.milestone_title,
            self.milestone_description,
            self.milestone_due_on,
            self.milestone_state,
        )
        yield

    async def create_queued_milestones_workflow(self) -> None:
        self.milestone_create_result = "Working…"
        yield
        self.milestone_create_result, _ = create_milestones_batch(
            self.github_token,
            self.milestone_repo,
            self.milestone_queue_text,
        )
        yield

    async def list_milestones_workflow(self) -> None:
        self.milestone_list_output = "Working…"
        yield
        self.milestone_list_output = list_milestones(
            self.github_token,
            self.milestone_list_repo,
            self.milestone_list_state,
        )
        yield

    async def run_servicenow_sync(self) -> None:
        self.sync_result = "Working…"
        yield
        self.sync_result = sync_servicenow(
            self.github_token,
            self.sync_repo,
            self.servicenow_url,
            self.servicenow_user,
            self.servicenow_password,
            self.servicenow_token,
            self.servicenow_milestone_table,
            self.servicenow_issue_table,
            self.servicenow_cursor_path,
            self.sync_dry_run,
            self.sync_back_sync_fields,
        )
        yield

    async def fetch_requirements_text(self) -> None:
        self.requirements_fetch_status = "Working…"
        yield
        text, status = fetch_requirements_from_repo(
            self.github_token,
            self.requirements_repo,
            self.requirements_path,
        )
        self.requirements_text = text
        self.requirements_fetch_status = status
        if self.requirements_repo.strip() and not self.submit_repo.strip():
            self.submit_repo = self.requirements_repo.strip()
        if (
            self.requirements_repo.strip()
            and not self.requirements_milestones_repo.strip()
        ):
            self.requirements_milestones_repo = self.requirements_repo.strip()
        yield

    async def parse_requirements_workflow(self) -> None:
        self.requirements_status = "Working…"
        self.submit_status = ""
        yield
        rows, status = parse_requirements(
            self.github_token,
            self.openai_key,
            self.agent_model,
            self.openai_base_url,
            self.requirements_text,
            None,
            self.requirements_use_milestones,
            self.requirements_milestones_repo or self.requirements_repo,
        )
        self.requirements_status = status
        self._hydrate_requirement_drafts(rows)
        if self.requirements_repo.strip():
            self.submit_repo = self.requirements_repo.strip()
        yield

    async def seed_requirements_milestones(self) -> None:
        self.requirements_milestone_status = "Working…"
        yield
        target_repo = (
            self.requirements_milestones_repo or self.requirements_repo
        ).strip()
        status, _ = create_milestones_batch(
            self.github_token,
            target_repo,
            self.milestone_queue_text,
        )
        self.requirements_milestone_status = status
        if status.startswith("✅") or status.startswith("⚠️"):
            self.requirements_use_milestones = True
            if target_repo:
                self.requirements_milestones_repo = target_repo
        yield

    async def submit_requirement_drafts(self) -> None:
        self.submit_status = "Working…"
        yield
        self.submit_status = submit_issues(
            self.github_token,
            self.submit_repo,
            self.submit_milestone_override,
            self._draft_rows(),
        )
        yield

    async def list_submit_open_issues_workflow(self) -> None:
        self.submit_open_issues_status = "Working…"
        self.submit_open_issues_markdown = ""
        yield
        markdown, status = list_open_issues_for_repo(
            self.github_token,
            self.submit_repo,
        )
        self.submit_open_issues_markdown = markdown
        self.submit_open_issues_status = status
        yield

    async def run_agile_workflow(self) -> None:
        self.agile_status = "Working…"
        self.agile_apply_status = ""
        self.agile_dependencies_markdown = ""
        self.agile_plan_markdown = ""
        self.agile_open_issues_markdown = ""
        self.agile_open_issues_status = ""
        self.agile_repos_markdown = ""
        self.agile_projects_markdown = ""
        self.agile_context_status = ""
        self.agile_project_board_markdown = ""
        self.agile_project_board_status = ""
        self._agile_result = None
        yield
        deps_md, plan_md, status = run_agile_planner(
            self.github_token,
            self.openai_key,
            self.agent_model,
            self.openai_base_url,
            self.agile_repo,
            self._safe_int(self.agile_capacity, 10),
            self._safe_int(self.agile_sprints, 3),
            False,
        )
        self.agile_dependencies_markdown = deps_md
        self.agile_plan_markdown = plan_md
        self.agile_status = status
        self._agile_result = run_agile_planner_state(
            self.github_token,
            self.openai_key,
            self.agent_model,
            self.openai_base_url,
            self.agile_repo,
            self._safe_int(self.agile_capacity, 10),
            self._safe_int(self.agile_sprints, 3),
            False,
        )
        yield

    async def apply_agile_relationships_workflow(self) -> None:
        self.agile_apply_status = "Working…"
        yield
        self.agile_apply_status = apply_agile_relationships(
            self.github_token,
            self.agile_repo,
            False,
            self._agile_result,
        )
        yield

    async def apply_agile_labels_workflow(self) -> None:
        self.agile_apply_status = "Working…"
        yield
        self.agile_apply_status = apply_agile_labels(
            self.github_token,
            self.agile_repo,
            False,
            self._agile_result,
        )
        yield

    async def read_agile_project_board_workflow(self) -> None:
        self.agile_project_board_status = "Working…"
        self.agile_project_board_markdown = ""
        yield
        markdown, status = read_agile_project_board(
            self.github_token,
            self.agile_repo,
            self._safe_int(self.agile_project_number, 0, min_value=0),
            self.agile_project_status_field,
            self._safe_int(self.agile_project_sprint_number, 0, min_value=0),
            self._agile_result,
        )
        self.agile_project_board_markdown = markdown
        self.agile_project_board_status = status
        yield

    async def update_agile_project_status_workflow(self) -> None:
        self.agile_project_board_status = "Working…"
        yield
        self.agile_project_board_status = update_agile_project_status(
            self.github_token,
            self.agile_repo,
            self._safe_int(self.agile_project_number, 0, min_value=0),
            self._safe_int(self.agile_project_issue_number, 0, min_value=0),
            self.agile_project_status_value,
            self.agile_project_status_field,
        )
        yield

    async def list_agile_repositories_workflow(self) -> None:
        self.agile_context_status = "Working…"
        self.agile_repos_markdown = ""
        yield
        markdown, status = list_repositories_for_owner(
            self.github_token,
            self.agile_repo,
        )
        self.agile_repos_markdown = markdown
        self.agile_context_status = status
        yield

    async def list_agile_projects_workflow(self) -> None:
        self.agile_context_status = "Working…"
        self.agile_projects_markdown = ""
        yield
        markdown, status = list_projects_for_target(
            self.github_token,
            self.agile_repo,
        )
        self.agile_projects_markdown = markdown
        self.agile_context_status = status
        yield

    async def create_agile_project_workflow(self) -> None:
        self.agile_context_status = "Working…"
        yield
        self.agile_context_status = create_project_for_owner(
            self.github_token,
            self.agile_repo,
            self.agile_new_project_title,
        )
        yield

    async def list_agile_open_issues_workflow(self) -> None:
        self.agile_open_issues_status = "Working…"
        self.agile_open_issues_markdown = ""
        yield
        markdown, status = list_open_issues_for_repo(
            self.github_token,
            self.agile_repo,
        )
        self.agile_open_issues_markdown = markdown
        self.agile_open_issues_status = status
        yield

    # ------------------------------------------------------------------ #
    # Chat helpers
    # ------------------------------------------------------------------ #

    def _make_ctx(self) -> AgentContext:
        return AgentContext(
            owner=self.owner,
            repo=self.repo,
            github_token=self.github_token,
            openai_api_key=self.openai_key,
            openai_base_url=self.openai_base_url,
            model=self.agent_model,
            servicenow_enabled=self.servicenow_enabled,
            servicenow_url=self.servicenow_url,
            servicenow_user=self.servicenow_user,
            servicenow_password=self.servicenow_password,
            servicenow_token=self.servicenow_token,
            servicenow_milestone_table=self.servicenow_milestone_table,
            servicenow_issue_table=self.servicenow_issue_table,
            servicenow_cursor_path=self.servicenow_cursor_path,
        )

    def _append(self, msg: ChatMessage) -> None:
        self.messages = [*self.messages, msg]

    def _append_reasoning(self, text: str) -> None:
        if text:
            self.reasoning_text += text

    @staticmethod
    def _to_json_string(value: Any) -> str:
        if value is None:
            return ""
        if isinstance(value, str):
            return value
        try:
            return json.dumps(value)
        except TypeError:
            return str(value)

    @staticmethod
    def _pretty_json(value: Any) -> str:
        raw = AppState._to_json_string(value)
        if not raw:
            return ""
        try:
            return json.dumps(json.loads(raw), indent=2)
        except (json.JSONDecodeError, TypeError):
            return raw

    @staticmethod
    def _event_attr(obj: Any, key: str, default: Any = None) -> Any:
        if obj is None:
            return default
        if isinstance(obj, dict):
            return obj.get(key, default)
        return getattr(obj, key, default)

    @staticmethod
    def _event_type(event: Any) -> str:
        event_type = AppState._event_attr(event, "type", "")
        if event_type:
            return str(event_type)
        data = AppState._event_attr(event, "data", None)
        data_type = AppState._event_attr(data, "type", "")
        return str(data_type or "")

    def _ensure_assistant_message(self) -> None:
        if not self.messages or self.messages[-1].role != "assistant":
            self._append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    role="assistant",
                    content="",
                )
            )

    def _current_assistant_message(self) -> ChatMessage | None:
        if self.messages and self.messages[-1].role == "assistant":
            return self.messages[-1]
        return None

    def _replace_current_assistant(self, updated: ChatMessage) -> None:
        if self.messages and self.messages[-1].role == "assistant":
            self.messages[-1] = updated

    def _upsert_tool_call_message(
        self, call_id: str, tool_name: str, args_str: str
    ) -> None:
        args_pretty = self._pretty_json(args_str)
        self._ensure_assistant_message()
        assistant = self._current_assistant_message()
        if assistant is None:
            return
        event_index = self._tool_event_index_by_call_id.get(call_id)
        if event_index is None or not (0 <= event_index < len(assistant.tool_events)):
            self._tool_event_index_by_call_id[call_id] = len(assistant.tool_events)
            assistant.tool_events.append(
                ToolEvent(
                    id=call_id,
                    kind="tool_call",
                    tool_name=tool_name or "unknown",
                    args_json=args_pretty,
                )
            )
        else:
            assistant.tool_events[event_index] = replace(
                assistant.tool_events[event_index],
                tool_name=tool_name or assistant.tool_events[event_index].tool_name,
                args_json=args_pretty,
            )
        self._replace_current_assistant(
            replace(assistant, tool_events=assistant.tool_events)
        )

    def _append_tool_result(self, tool_name: str, output: Any) -> None:
        self._ensure_assistant_message()
        assistant = self._current_assistant_message()
        if assistant is None:
            return
        assistant.tool_events.append(
            ToolEvent(
                id=str(uuid.uuid4()),
                kind="tool_result",
                tool_name=tool_name or "unknown",
                content=self._pretty_json(output),
            )
        )
        self._replace_current_assistant(
            replace(assistant, tool_events=assistant.tool_events)
        )

    def _append_tool_error(self, tool_name: str, error: Any) -> None:
        self._ensure_assistant_message()
        assistant = self._current_assistant_message()
        if assistant is None:
            return
        assistant.tool_events.append(
            ToolEvent(
                id=str(uuid.uuid4()),
                kind="tool_error",
                tool_name=tool_name or "unknown",
                content=self._pretty_json(error),
                is_error=True,
            )
        )
        self._replace_current_assistant(
            replace(assistant, tool_events=assistant.tool_events)
        )

    def _mark_tool_started(self, call_id: str, tool_name: str, args_str: Any) -> None:
        args_text = self._to_json_string(args_str)
        if not call_id:
            call_id = str(uuid.uuid4())
        self._tool_args_by_call_id[call_id] = args_text
        self._active_tool_calls.add(call_id)
        self.processing_tool = True
        self._upsert_tool_call_message(call_id, tool_name, args_text)

    def _append_tool_args_delta(
        self, call_id: str, delta: str, tool_name: str = ""
    ) -> None:
        if not call_id:
            return
        existing = self._tool_args_by_call_id.get(call_id, "")
        updated = f"{existing}{delta or ''}"
        self._tool_args_by_call_id[call_id] = updated
        self._active_tool_calls.add(call_id)
        self.processing_tool = True
        self._upsert_tool_call_message(call_id, tool_name, updated)

    def _mark_tool_finished(
        self, call_id: str, tool_name: str, output: Any, is_error: bool = False
    ) -> None:
        if call_id and call_id in self._active_tool_calls:
            self._active_tool_calls.remove(call_id)
        if call_id and call_id in self._tool_args_by_call_id:
            self._tool_args_by_call_id.pop(call_id, None)
        self.processing_tool = bool(self._active_tool_calls)
        if is_error:
            self._append_tool_error(tool_name, output)
        else:
            self._append_tool_result(tool_name, output)

    def _extract_reasoning_text(self, item: Any) -> str:
        parts = self._event_attr(item, "content", []) or []
        if isinstance(parts, str):
            return parts
        chunks: list[str] = []
        for part in parts:
            text = self._event_attr(part, "text", "")
            if text:
                chunks.append(str(text))
            else:
                maybe_text = self._event_attr(part, "reasoning", "")
                if maybe_text:
                    chunks.append(str(maybe_text))
        return "".join(chunks)

    def _handle_stream_event(self, event: Any) -> bool:
        """Normalize stream events from Agents SDK and plain OpenAI Responses streams."""
        if isinstance(event, RawResponsesStreamEvent):
            event_type = getattr(event.data, "type", "")
            if event_type == "response.output_text.delta":
                self.streaming_text += self._event_attr(event.data, "delta", "")
                return True
            if event_type == "response.reasoning_text.delta":
                self._append_reasoning(self._event_attr(event.data, "delta", ""))
                return True

        if isinstance(event, RunItemStreamEvent):
            if event.name == "reasoning_item_created":
                text = self._extract_reasoning_text(
                    self._event_attr(event.item, "raw_item", None)
                )
                if text:
                    self._append_reasoning(text)
                    return True

            if event.name == "tool_called":
                raw = event.item.raw_item
                self._mark_tool_started(
                    call_id=self._event_attr(raw, "call_id", "") or str(uuid.uuid4()),
                    tool_name=self._event_attr(raw, "name", "unknown"),
                    args_str=self._event_attr(raw, "arguments", "{}"),
                )
                return True

            if event.name == "tool_output":
                raw = self._event_attr(event.item, "raw_item", None)
                tool_name = self._event_attr(raw, "name", "") or self._event_attr(
                    event.item, "tool_name", "unknown"
                )
                call_id = self._event_attr(raw, "call_id", "")
                self._mark_tool_finished(
                    call_id=call_id,
                    tool_name=tool_name,
                    output=self._event_attr(event.item, "output", ""),
                    is_error=False,
                )
                return True

        event_type = self._event_type(event)
        data = self._event_attr(event, "data", None)

        if event_type == "response.output_text.delta":
            self.streaming_text += self._event_attr(data, "delta", "")
            return True

        if event_type == "response.reasoning_text.delta":
            self._append_reasoning(self._event_attr(data, "delta", ""))
            return True

        if event_type in (
            "response.output_item.added",
            "response.output_item.delta",
            "response.output_item.done",
        ):
            item = self._event_attr(data, "item", None) or self._event_attr(
                event, "item", None
            )
            item_type = self._event_attr(item, "type", "")
            call_id = self._event_attr(item, "call_id", "") or self._event_attr(
                item, "id", ""
            )
            tool_name = self._event_attr(item, "name", "unknown")

            if item_type in ("function_call", "tool_call"):
                args = self._event_attr(item, "arguments", "")
                self._mark_tool_started(
                    call_id=call_id, tool_name=tool_name, args_str=args
                )
                return True

            if item_type in ("function_call_output", "tool_result"):
                output = self._event_attr(item, "output", "")
                self._mark_tool_finished(
                    call_id=call_id, tool_name=tool_name, output=output
                )
                return True

        if event_type in (
            "response.function_call_arguments.delta",
            "response.tool_call.arguments.delta",
        ):
            call_id = self._event_attr(data, "call_id", "")
            delta = self._event_attr(data, "delta", "")
            tool_name = self._event_attr(data, "name", "")
            self._append_tool_args_delta(
                call_id=call_id, delta=delta, tool_name=tool_name
            )
            return True

        if event_type in (
            "response.function_call_arguments.done",
            "response.tool_call.arguments.done",
        ):
            call_id = self._event_attr(data, "call_id", "")
            args = self._event_attr(data, "arguments", "")
            tool_name = self._event_attr(data, "name", "")
            self._mark_tool_started(call_id=call_id, tool_name=tool_name, args_str=args)
            return True

        if event_type in (
            "response.tool_error",
            "response.function_call.error",
        ):
            call_id = self._event_attr(data, "call_id", "")
            tool_name = self._event_attr(data, "name", "unknown")
            error = self._event_attr(data, "error", "Tool call failed")
            self._mark_tool_finished(
                call_id=call_id, tool_name=tool_name, output=error, is_error=True
            )
            return True

        return False

    def _flush_assistant_message(self) -> None:
        self._ensure_assistant_message()
        assistant = self._current_assistant_message()
        if assistant is not None:
            self._replace_current_assistant(
                replace(
                    assistant,
                    content=self.streaming_text or assistant.content,
                    reasoning_text=self.reasoning_text or assistant.reasoning_text,
                )
            )
            self.streaming_text = ""
            self.reasoning_text = ""
        self.processing_tool = bool(self._active_tool_calls)

    # ------------------------------------------------------------------ #
    # Primary chat handler
    # ------------------------------------------------------------------ #

    async def send_message(self) -> None:
        """Handle the user submitting a chat message (async streaming)."""
        message = self.input_value.strip()
        if not message or self.is_thinking:
            return

        self.input_value = ""
        self.input_disabled = True
        self.is_thinking = True
        self.processing_tool = False
        self.streaming_text = ""
        self.reasoning_text = ""
        self._tool_args_by_call_id = {}
        self._active_tool_calls = set()
        self._tool_event_index_by_call_id = {}
        self._append(
            ChatMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=message,
            )
        )
        self._append(
            ChatMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content="",
            )
        )
        yield

        ctx = self._make_ctx()
        history = self._conversation_history or None

        try:
            result = run_agent_streaming(ctx, message, history=history)
            async for event in result.stream_events():
                if self._handle_stream_event(event):
                    yield

            # ---- Stream finished ----------------------------------------
            # Persist streaming text as the final assistant message
            self._flush_assistant_message()
            self.reasoning_text = ""

            # Update conversation history for multi-turn context
            try:
                self._conversation_history = list(result.to_input_list())
            except Exception:
                self._conversation_history = []

            # ---- Check for HITL interruptions ---------------------------
            interruptions: list[ToolApprovalItem] = result.interruptions
            if interruptions:
                self._pending_result = result
                hitl_items: list[HITLRequest] = []
                for item in interruptions:
                    raw = item.raw_item
                    tool_name = item.tool_name or getattr(raw, "name", "unknown")
                    args_str = getattr(raw, "arguments", "{}")
                    try:
                        args_pretty = json.dumps(json.loads(args_str), indent=2)
                    except (json.JSONDecodeError, TypeError):
                        args_pretty = args_str
                    hitl_items.append(
                        HITLRequest(
                            id=str(uuid.uuid4()),
                            tool_name=tool_name,
                            args_json=args_pretty,
                            description=f"Approve call to **{tool_name}**?",
                        )
                    )
                self.pending_hitl = hitl_items
                self.is_thinking = False
                self.input_disabled = True  # keep input disabled until resolved
                yield
            else:
                self.is_thinking = False
                self.processing_tool = False
                self.input_disabled = False
                yield

        except Exception as exc:
            logger.exception("Agent run error")
            self._flush_assistant_message()
            self.reasoning_text = ""
            self._append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    role="error",
                    content=f"Error: {exc}",
                    is_error=True,
                )
            )
            self.is_thinking = False
            self.processing_tool = False
            self.input_disabled = False
            yield

    # ------------------------------------------------------------------ #
    # HITL handlers
    # ------------------------------------------------------------------ #

    async def approve_hitl(self, hitl_id: str) -> None:
        """Approve the HITL request with the given *hitl_id* and resume the run."""
        pending = self._pending_result
        if pending is None:
            return

        interruptions: list[ToolApprovalItem] = pending.interruptions
        if not interruptions:
            self.pending_hitl = []
            self._pending_result = None
            self.input_disabled = False
            yield
            return

        # Approve all pending interruptions (there is typically only one)
        run_state = pending.to_state()
        for item in interruptions:
            run_state.approve(item)

        self.pending_hitl = []
        self._pending_result = None
        self.is_thinking = True
        self.processing_tool = False
        self.streaming_text = ""
        self.reasoning_text = ""
        self._tool_args_by_call_id = {}
        self._active_tool_calls = set()
        self._tool_event_index_by_call_id = {}
        yield

        ctx = self._make_ctx()
        try:
            result = run_agent_streaming(ctx, run_state)
            async for event in result.stream_events():
                if self._handle_stream_event(event):
                    yield

            self._flush_assistant_message()
            self.reasoning_text = ""

            try:
                self._conversation_history = list(result.to_input_list())
            except Exception:
                pass

            new_interruptions: list[ToolApprovalItem] = result.interruptions
            if new_interruptions:
                self._pending_result = result
                hitl_items = []
                for item in new_interruptions:
                    raw = item.raw_item
                    tool_name = item.tool_name or getattr(raw, "name", "unknown")
                    args_str = getattr(raw, "arguments", "{}")
                    try:
                        args_pretty = json.dumps(json.loads(args_str), indent=2)
                    except (json.JSONDecodeError, TypeError):
                        args_pretty = args_str
                    hitl_items.append(
                        HITLRequest(
                            id=str(uuid.uuid4()),
                            tool_name=tool_name,
                            args_json=args_pretty,
                            description=f"Approve call to **{tool_name}**?",
                        )
                    )
                self.pending_hitl = hitl_items
                self.is_thinking = False
                self.input_disabled = True
                yield
            else:
                self.is_thinking = False
                self.processing_tool = False
                self.input_disabled = False
                yield

        except Exception as exc:
            logger.exception("Agent resume error")
            self._flush_assistant_message()
            self.reasoning_text = ""
            self._append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    role="error",
                    content=f"Error during approved action: {exc}",
                    is_error=True,
                )
            )
            self.is_thinking = False
            self.processing_tool = False
            self.input_disabled = False
            yield

    async def deny_hitl(self, hitl_id: str) -> None:
        """Deny all pending HITL requests and inform the agent."""
        pending = self._pending_result
        if pending is not None:
            interruptions: list[ToolApprovalItem] = pending.interruptions
            if interruptions:
                run_state = pending.to_state()
                for item in interruptions:
                    run_state.reject(
                        item,
                        rejection_message="User denied this action.",
                    )

        self.pending_hitl = []
        self._pending_result = None
        self._append(
            ChatMessage(
                id=str(uuid.uuid4()),
                role="assistant",
                content="Action denied. Let me know how you'd like to proceed.",
            )
        )
        self.is_thinking = False
        self.processing_tool = False
        self.input_disabled = False
        self._tool_args_by_call_id = {}
        self._active_tool_calls = set()
        self._tool_event_index_by_call_id = {}
        yield

    def clear_chat(self) -> None:
        """Clear the entire conversation history."""
        self.messages = []
        self.streaming_text = ""
        self.reasoning_text = ""
        self.pending_hitl = []
        self._pending_result = None
        self._conversation_history = []
        self.is_thinking = False
        self.processing_tool = False
        self.input_disabled = False
        self._tool_args_by_call_id = {}
        self._active_tool_calls = set()
        self._tool_event_index_by_call_id = {}
