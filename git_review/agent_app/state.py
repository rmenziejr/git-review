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
3.  Text deltas are accumulated in ``streaming_text``; tool-call and
    reasoning events are appended to ``messages``; each ``yield`` inside
    the loop pushes the incremental update live.
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
from typing import Any, Optional

import reflex as rx

from ..agent import AgentContext, run_agent_streaming
from ..config import AppSettings
from ..ui_workflows import (
    apply_agile_labels,
    apply_agile_relationships,
    append_milestone_to_batch,
    create_milestone,
    create_milestones_batch,
    fetch_requirements_from_repo,
    list_milestones,
    load_default_milestones_text,
    parse_requirements,
    run_agile_planner,
    run_agile_planner_state,
    submit_issues,
    summarize_activity,
    sync_servicenow,
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


class ChatMessage(rx.Model):
    """A single entry in the chat history."""

    id: str
    role: str
    """'user' | 'assistant' | 'reasoning' | 'tool_call' | 'tool_result' | 'error'"""
    content: str
    tool_name: str = ""
    args_json: str = ""
    is_error: bool = False


class HITLRequest(rx.Model):
    """A pending write-tool approval request surfaced in the HITL panel."""

    id: str
    tool_name: str
    args_json: str
    description: str


class RequirementDraft(rx.Model):
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
    is_thinking: bool = False

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

    agile_repo: str = ""
    agile_capacity: str = "10"
    agile_sprints: str = "3"
    agile_status: str = ""
    agile_dependencies_markdown: str = ""
    agile_plan_markdown: str = ""
    agile_apply_status: str = ""

    # ---- Backend-only (not sent to frontend) ----
    _pending_result: Any = None
    _conversation_history: list = []
    _agile_result: Any = None

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
                queue_text, status = load_default_milestones_text(settings.default_milestones_json)
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
            logger.warning("Ignoring requirement draft update for unknown field: %s", field_name)
            return
        drafts = list(self.requirement_drafts)
        if not 0 <= idx < len(drafts):
            logger.warning(
                "Ignoring requirement draft update for out-of-range index %s (draft count=%s)",
                idx,
                len(drafts),
            )
            return
        updated = drafts[idx].model_copy()
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
            queue_text, status = load_default_milestones_text(settings.default_milestones_json)
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
        repo_value = (self.requirements_milestones_repo or self.requirements_repo).strip()
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

    def _safe_int(self, raw: str, default: int) -> int:
        try:
            return max(1, int((raw or "").strip() or str(default)))
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
        if self.requirements_repo.strip() and not self.requirements_milestones_repo.strip():
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
        target_repo = (self.requirements_milestones_repo or self.requirements_repo).strip()
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

    async def run_agile_workflow(self) -> None:
        self.agile_status = "Working…"
        self.agile_apply_status = ""
        self.agile_dependencies_markdown = ""
        self.agile_plan_markdown = ""
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
        self._append(
            ChatMessage(
                id=str(uuid.uuid4()),
                role="user",
                content=message,
            )
        )
        yield

        ctx = self._make_ctx()
        history = self._conversation_history or None

        try:
            result = run_agent_streaming(ctx, message, history=history)
            async for event in result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    event_type = getattr(event.data, "type", "")
                    if event_type == "response.output_text.delta":
                        self.streaming_text += event.data.delta
                        yield
                    elif event_type == "response.reasoning_text.delta":
                        self.streaming_text += event.data.delta
                        yield

                elif isinstance(event, RunItemStreamEvent):
                    if event.name == "reasoning_item_created":
                        parts = getattr(event.item.raw_item, "content", []) or []
                        text = "".join(
                            getattr(p, "text", "") for p in parts
                        )
                        if text:
                            self._append(
                                ChatMessage(
                                    id=str(uuid.uuid4()),
                                    role="reasoning",
                                    content=text,
                                )
                            )
                            yield

                    elif event.name == "tool_called":
                        raw = event.item.raw_item
                        tool_name = getattr(raw, "name", "unknown")
                        args_str = getattr(raw, "arguments", "{}")
                        try:
                            args_pretty = json.dumps(
                                json.loads(args_str), indent=2
                            )
                        except (json.JSONDecodeError, TypeError):
                            args_pretty = args_str
                        call_id = getattr(raw, "call_id", "") or str(
                            uuid.uuid4()
                        )
                        self._append(
                            ChatMessage(
                                id=call_id,
                                role="tool_call",
                                content="",
                                tool_name=tool_name,
                                args_json=args_pretty,
                            )
                        )
                        yield

                    elif event.name == "tool_output":
                        output = str(getattr(event.item, "output", ""))
                        tool_name = getattr(
                            event.item.raw_item, "name", ""
                        ) or getattr(event.item, "tool_name", "unknown")
                        try:
                            output_pretty = json.dumps(
                                json.loads(output), indent=2
                            )
                        except (json.JSONDecodeError, TypeError):
                            output_pretty = output
                        self._append(
                            ChatMessage(
                                id=str(uuid.uuid4()),
                                role="tool_result",
                                content=output_pretty,
                                tool_name=tool_name,
                            )
                        )
                        yield

            # ---- Stream finished ----------------------------------------
            # Persist streaming text as the final assistant message
            if self.streaming_text:
                self._append(
                    ChatMessage(
                        id=str(uuid.uuid4()),
                        role="assistant",
                        content=self.streaming_text,
                    )
                )
                self.streaming_text = ""

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
                        args_pretty = json.dumps(
                            json.loads(args_str), indent=2
                        )
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
                self.input_disabled = False
                yield

        except Exception as exc:
            logger.exception("Agent run error")
            if self.streaming_text:
                self._append(
                    ChatMessage(
                        id=str(uuid.uuid4()),
                        role="assistant",
                        content=self.streaming_text,
                    )
                )
                self.streaming_text = ""
            self._append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    role="error",
                    content=f"Error: {exc}",
                    is_error=True,
                )
            )
            self.is_thinking = False
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
        yield

        ctx = self._make_ctx()
        try:
            result = run_agent_streaming(ctx, run_state)
            async for event in result.stream_events():
                if isinstance(event, RawResponsesStreamEvent):
                    event_type = getattr(event.data, "type", "")
                    if event_type in (
                        "response.output_text.delta",
                        "response.reasoning_text.delta",
                    ):
                        self.streaming_text += event.data.delta
                        yield

                elif isinstance(event, RunItemStreamEvent):
                    if event.name == "tool_called":
                        raw = event.item.raw_item
                        tool_name = getattr(raw, "name", "unknown")
                        args_str = getattr(raw, "arguments", "{}")
                        try:
                            args_pretty = json.dumps(
                                json.loads(args_str), indent=2
                            )
                        except (json.JSONDecodeError, TypeError):
                            args_pretty = args_str
                        call_id = getattr(raw, "call_id", "") or str(uuid.uuid4())
                        self._append(
                            ChatMessage(
                                id=call_id,
                                role="tool_call",
                                content="",
                                tool_name=tool_name,
                                args_json=args_pretty,
                            )
                        )
                        yield

                    elif event.name == "tool_output":
                        output = str(getattr(event.item, "output", ""))
                        tool_name = getattr(
                            event.item.raw_item, "name", ""
                        ) or getattr(event.item, "tool_name", "unknown")
                        try:
                            output_pretty = json.dumps(
                                json.loads(output), indent=2
                            )
                        except (json.JSONDecodeError, TypeError):
                            output_pretty = output
                        self._append(
                            ChatMessage(
                                id=str(uuid.uuid4()),
                                role="tool_result",
                                content=output_pretty,
                                tool_name=tool_name,
                            )
                        )
                        yield

            if self.streaming_text:
                self._append(
                    ChatMessage(
                        id=str(uuid.uuid4()),
                        role="assistant",
                        content=self.streaming_text,
                    )
                )
                self.streaming_text = ""

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
                self.input_disabled = False
                yield

        except Exception as exc:
            logger.exception("Agent resume error")
            if self.streaming_text:
                self._append(
                    ChatMessage(
                        id=str(uuid.uuid4()),
                        role="assistant",
                        content=self.streaming_text,
                    )
                )
                self.streaming_text = ""
            self._append(
                ChatMessage(
                    id=str(uuid.uuid4()),
                    role="error",
                    content=f"Error during approved action: {exc}",
                    is_error=True,
                )
            )
            self.is_thinking = False
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
        self.input_disabled = False
        yield

    def clear_chat(self) -> None:
        """Clear the entire conversation history."""
        self.messages = []
        self.streaming_text = ""
        self.pending_hitl = []
        self._pending_result = None
        self._conversation_history = []
        self.is_thinking = False
        self.input_disabled = False
