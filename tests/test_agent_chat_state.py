"""Tests for streaming chat state aggregation."""

from __future__ import annotations

from types import SimpleNamespace

from git_review.webapp.agent_app.state import AppState, ChatMessage


def _state_with_active_assistant() -> AppState:
    state = AppState(_reflex_internal_init=True)
    state.messages = [
        ChatMessage(id="user-1", role="user", content="List issues"),
        ChatMessage(id="assistant-1", role="assistant", content=""),
    ]
    return state


def _responses_event(event_type: str, **data: object) -> SimpleNamespace:
    return SimpleNamespace(
        type=event_type,
        data=SimpleNamespace(type=event_type, **data),
    )


def test_streaming_output_delta_updates_active_assistant_bubble() -> None:
    state = _state_with_active_assistant()

    handled = state._handle_stream_event(
        _responses_event("response.output_text.delta", delta="Hello")
    )

    assert handled is True
    assert state.messages[-1].role == "assistant"
    assert state.messages[-1].content == "Hello"
    assert state.streaming_text == "Hello"


def test_reasoning_delta_updates_active_assistant_bubble() -> None:
    state = _state_with_active_assistant()

    handled = state._handle_stream_event(
        _responses_event("response.reasoning_text.delta", delta="Checking repo")
    )

    assert handled is True
    assert state.messages[-1].role == "assistant"
    assert state.messages[-1].reasoning_text == "Checking repo"
    assert state.reasoning_text == "Checking repo"
