"""Tests for git_review.langgraph_middleware."""

from __future__ import annotations

import logging
import time
from unittest.mock import MagicMock

import pytest

from git_review.langgraph_middleware import (
    LoggingMiddleware,
    RetryMiddleware,
    TokenCountingMiddleware,
    TokenUsageCounter,
    apply_middleware,
)


# ---------------------------------------------------------------------------
# LoggingMiddleware
# ---------------------------------------------------------------------------

def test_logging_middleware_passes_state_through() -> None:
    def node(state):
        return {"result": "ok"}

    wrapped = LoggingMiddleware()(node)
    assert wrapped({"input": "x"}) == {"result": "ok"}


def test_logging_middleware_logs_entry_and_exit(caplog) -> None:
    def my_node(state):
        return {"val": 1}

    with caplog.at_level(logging.DEBUG, logger="git_review.langgraph_middleware"):
        wrapped = LoggingMiddleware()(my_node)
        wrapped({})

    log_text = " ".join(caplog.messages)
    assert "my_node" in log_text
    assert "entering" in log_text
    assert "completed" in log_text


def test_logging_middleware_logs_error_on_exception(caplog) -> None:
    def failing_node(state):
        raise ValueError("boom")

    with caplog.at_level(logging.ERROR, logger="git_review.langgraph_middleware"):
        wrapped = LoggingMiddleware()(failing_node)
        with pytest.raises(ValueError, match="boom"):
            wrapped({})

    assert any("failed" in m for m in caplog.messages)


def test_logging_middleware_preserves_function_name() -> None:
    def special_node(state):
        return {}

    wrapped = LoggingMiddleware()(special_node)
    assert wrapped.__name__ == "special_node"


def test_logging_middleware_accepts_custom_logger() -> None:
    custom_logger = MagicMock(spec=logging.Logger)
    custom_logger.log = MagicMock()

    def node(state):
        return {}

    wrapped = LoggingMiddleware(node_logger=custom_logger)(node)
    wrapped({})

    assert custom_logger.log.called


# ---------------------------------------------------------------------------
# TokenUsageCounter
# ---------------------------------------------------------------------------

def test_token_usage_counter_accumulates_totals() -> None:
    counter = TokenUsageCounter()
    counter.record("node_a", 100)
    counter.record("node_b", 50)
    counter.record("node_a", 25)

    assert counter.total_tokens == 175
    assert counter.by_node["node_a"] == 125
    assert counter.by_node["node_b"] == 50


def test_token_usage_counter_starts_at_zero() -> None:
    counter = TokenUsageCounter()
    assert counter.total_tokens == 0
    assert counter.by_node == {}


# ---------------------------------------------------------------------------
# TokenCountingMiddleware
# ---------------------------------------------------------------------------

def test_token_counting_middleware_records_summary_text_tokens() -> None:
    text = "x" * 400  # 400 chars → ~100 tokens at 4 chars/token
    counter = TokenUsageCounter()

    def node(state):
        return {"summary_text": text}

    wrapped = TokenCountingMiddleware(counter=counter)(node)
    wrapped({})

    assert counter.total_tokens == 100
    assert counter.by_node.get("node") == 100


def test_token_counting_middleware_records_zero_for_no_summary_text() -> None:
    counter = TokenUsageCounter()

    def node(state):
        return {"other_field": "value"}

    wrapped = TokenCountingMiddleware(counter=counter)(node)
    wrapped({})

    assert counter.total_tokens == 0


def test_token_counting_middleware_creates_counter_if_not_provided() -> None:
    mw = TokenCountingMiddleware()
    assert isinstance(mw.counter, TokenUsageCounter)


def test_token_counting_middleware_shared_counter() -> None:
    counter = TokenUsageCounter()
    mw = TokenCountingMiddleware(counter=counter)

    def node_a(state):
        return {"summary_text": "a" * 40}

    def node_b(state):
        return {"summary_text": "b" * 80}

    mw(node_a)({})
    mw(node_b)({})

    assert counter.total_tokens == 30  # 10 + 20
    assert counter.by_node["node_a"] == 10
    assert counter.by_node["node_b"] == 20


# ---------------------------------------------------------------------------
# RetryMiddleware
# ---------------------------------------------------------------------------

def test_retry_middleware_succeeds_on_first_attempt() -> None:
    calls = []

    def node(state):
        calls.append(1)
        return {"done": True}

    wrapped = RetryMiddleware(max_attempts=3)(node)
    result = wrapped({})

    assert result == {"done": True}
    assert len(calls) == 1


def test_retry_middleware_retries_on_failure(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)  # skip real delays
    attempt_count = [0]

    def flaky_node(state):
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise RuntimeError("transient error")
        return {"ok": True}

    wrapped = RetryMiddleware(max_attempts=3, initial_delay=0.0)(flaky_node)
    result = wrapped({})

    assert result == {"ok": True}
    assert attempt_count[0] == 3


def test_retry_middleware_raises_after_max_attempts(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def always_fails(state):
        raise ValueError("persistent error")

    wrapped = RetryMiddleware(max_attempts=2, initial_delay=0.0)(always_fails)
    with pytest.raises(ValueError, match="persistent error"):
        wrapped({})


def test_retry_middleware_does_not_catch_excluded_exceptions(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)

    def node(state):
        raise TypeError("not retried")

    # Only retry on RuntimeError, not TypeError
    wrapped = RetryMiddleware(max_attempts=3, exceptions=(RuntimeError,))(node)
    with pytest.raises(TypeError, match="not retried"):
        wrapped({})


def test_retry_middleware_raises_for_invalid_max_attempts() -> None:
    with pytest.raises(ValueError, match="max_attempts"):
        RetryMiddleware(max_attempts=0)


def test_retry_middleware_preserves_function_name() -> None:
    def my_node(state):
        return {}

    wrapped = RetryMiddleware()(my_node)
    assert wrapped.__name__ == "my_node"


# ---------------------------------------------------------------------------
# apply_middleware
# ---------------------------------------------------------------------------

def test_apply_middleware_stacks_multiple_wrappers() -> None:
    """Middleware should be applied outermost-first (left to right)."""
    call_order: list[str] = []

    class OrderTracker:
        def __init__(self, name: str) -> None:
            self._name = name

        def __call__(self, node_func):
            name = self._name

            def wrapper(state):
                call_order.append(f"enter:{name}")
                result = node_func(state)
                call_order.append(f"exit:{name}")
                return result

            return wrapper

    def base_node(state):
        call_order.append("base")
        return {}

    wrapped = apply_middleware(
        base_node,
        OrderTracker("A"),
        OrderTracker("B"),
    )
    wrapped({})

    assert call_order == ["enter:A", "enter:B", "base", "exit:B", "exit:A"]


def test_apply_middleware_no_middleware_returns_original() -> None:
    def node(state):
        return {"x": 1}

    wrapped = apply_middleware(node)
    assert wrapped({}) == {"x": 1}


def test_apply_middleware_with_real_middleware(monkeypatch) -> None:
    monkeypatch.setattr(time, "sleep", lambda s: None)
    counter = TokenUsageCounter()

    def node(state):
        return {"summary_text": "t" * 100}

    wrapped = apply_middleware(
        node,
        LoggingMiddleware(),
        TokenCountingMiddleware(counter=counter),
        RetryMiddleware(max_attempts=2),
    )
    result = wrapped({})

    assert result["summary_text"] == "t" * 100
    assert counter.total_tokens == 25  # 100 chars / 4
