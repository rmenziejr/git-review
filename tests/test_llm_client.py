"""Tests for LLMClient."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from git_review.llm_client import LLMClient, _build_user_message
from git_review.models import Commit, Issue, PullRequest, ReviewSummary

SINCE = datetime(2024, 1, 1, tzinfo=timezone.utc)
UNTIL = datetime(2024, 1, 31, tzinfo=timezone.utc)


def _make_summary() -> ReviewSummary:
    return ReviewSummary(
        owner="acme",
        repo="app",
        since=SINCE,
        until=UNTIL,
        commits=[
            Commit(
                sha="abc1234567890",
                message="feat: add login page",
                author="Alice",
                authored_at=datetime(2024, 1, 10, tzinfo=timezone.utc),
                url="https://github.com/acme/app/commit/abc1234567890",
                repo="acme/app",
            )
        ],
        issues=[
            Issue(
                number=1,
                title="Login bug",
                state="closed",
                author="bob",
                created_at=datetime(2024, 1, 5, tzinfo=timezone.utc),
                closed_at=datetime(2024, 1, 12, tzinfo=timezone.utc),
                url="https://github.com/acme/app/issues/1",
                repo="acme/app",
                labels=["bug"],
            )
        ],
        pull_requests=[
            PullRequest(
                number=10,
                title="Add dark mode",
                state="closed",
                author="carol",
                created_at=datetime(2024, 1, 8, tzinfo=timezone.utc),
                merged_at=datetime(2024, 1, 15, tzinfo=timezone.utc),
                url="https://github.com/acme/app/pull/10",
                repo="acme/app",
                reviewer_comments={"alice": 2, "bob": 1},
            )
        ],
    )


# ---------------------------------------------------------------------------
# _build_user_message
# ---------------------------------------------------------------------------

def test_build_user_message_contains_key_info() -> None:
    summary = _make_summary()
    msg = _build_user_message(summary)

    assert "acme/app" in msg
    assert "feat: add login page" in msg
    assert "Login bug" in msg
    assert "Add dark mode" in msg
    assert "2024-01-01" in msg
    assert "2024-01-31" in msg
    # Reviewer comment info should be included
    assert "alice" in msg
    assert "2 comment" in msg


# ---------------------------------------------------------------------------
# LLMClient.summarise
# ---------------------------------------------------------------------------

def test_summarise_populates_summary_text() -> None:
    summary = _make_summary()

    mock_choice = MagicMock()
    mock_choice.message.content = "## Highlights\nGreat work this month."

    mock_response = MagicMock()
    mock_response.choices = [mock_choice]

    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.return_value = mock_response

    mock_openai_cls = MagicMock(return_value=mock_openai_instance)

    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        client = LLMClient(api_key="sk-fake")
        result = client.summarise(summary)

    assert "Highlights" in result and "Great work" in result


def test_llm_client_passes_custom_base_url() -> None:
    mock_openai_cls = MagicMock()

    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        LLMClient(api_key="sk-fake", base_url="http://localhost:11434/v1")

    _, kwargs = mock_openai_cls.call_args
    assert kwargs.get("base_url") == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# LLMClient custom system_prompt
# ---------------------------------------------------------------------------

def test_llm_client_accepts_valid_custom_system_prompt() -> None:
    mock_openai_cls = MagicMock()
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        client = LLMClient(api_key="sk-fake", system_prompt="Summarise in {{ n }} words.")
    assert client._system_prompt_template == "Summarise in {{ n }} words."


def test_llm_client_raises_on_unknown_variable_in_system_prompt() -> None:
    with pytest.raises(ValueError, match="unknown variable"):
        with patch("git_review.llm_client.OpenAI", MagicMock()):
            LLMClient(api_key="sk-fake", system_prompt="{{ bogus_var }}")


def test_llm_client_renders_custom_prompt_with_context() -> None:
    summary = _make_summary()

    mock_choice = MagicMock()
    mock_choice.message.content = "Custom summary"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_instance = MagicMock()
    mock_openai_instance.chat.completions.create.return_value = mock_response

    with patch("git_review.llm_client.OpenAI", MagicMock(return_value=mock_openai_instance)):
        client = LLMClient(
            api_key="sk-fake",
            system_prompt="Counts: {{ n_commits }} / {{ n_issues }} / {{ n_prs }}.",
        )
        client.summarise(summary)

    call_kwargs = mock_openai_instance.chat.completions.create.call_args[1]
    system_msg = next(m["content"] for m in call_kwargs["messages"] if m["role"] == "system")
    # _make_summary() has 1 commit, 1 issue, 1 PR
    assert "Counts: 1 / 1 / 1." in system_msg


def test_llm_client_uses_default_prompt_when_none_given() -> None:
    mock_openai_cls = MagicMock()
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        client = LLMClient(api_key="sk-fake")
    from git_review.llm_client import _DEFAULT_SYSTEM_PROMPT
    assert client._system_prompt_template is _DEFAULT_SYSTEM_PROMPT
