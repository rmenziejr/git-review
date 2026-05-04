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
    settings_open: bool = False

    # ---- Backend-only (not sent to frontend) ----
    _pending_result: Any = None
    _conversation_history: list = []

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

    def set_input_value(self, value: str) -> None:
        self.input_value = value

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
