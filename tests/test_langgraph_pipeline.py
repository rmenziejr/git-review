"""Tests for git_review.langgraph_pipeline."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from git_review.models import Commit, Issue, PullRequest, ReviewSummary

SINCE = datetime(2024, 1, 1, tzinfo=timezone.utc)
UNTIL = datetime(2024, 1, 31, tzinfo=timezone.utc)


def _make_summary(
    commits: int = 1, issues: int = 1, prs: int = 1
) -> ReviewSummary:
    summary = ReviewSummary(
        owner="acme",
        repo="app",
        since=SINCE,
        until=UNTIL,
    )
    for i in range(commits):
        summary.commits.append(
            Commit(
                sha=f"abc{i:010d}",
                message=f"feat: change {i}",
                author="alice",
                authored_at=SINCE,
                url=f"https://github.com/acme/app/commit/abc{i}",
                repo="acme/app",
            )
        )
    for i in range(issues):
        summary.issues.append(
            Issue(
                number=i + 1,
                title=f"Issue {i + 1}",
                state="open",
                author="bob",
                created_at=SINCE,
                closed_at=None,
                url=f"https://github.com/acme/app/issues/{i + 1}",
                repo="acme/app",
            )
        )
    for i in range(prs):
        summary.pull_requests.append(
            PullRequest(
                number=i + 100,
                title=f"PR {i + 1}",
                state="merged",
                author="carol",
                created_at=SINCE,
                merged_at=UNTIL,
                url=f"https://github.com/acme/app/pull/{i + 100}",
                repo="acme/app",
            )
        )
    return summary


def _make_mock_openai(response_text: str = "## Highlights\nGreat month.") -> MagicMock:
    """Return a mock OpenAI class whose create() returns *response_text*."""
    mock_choice = MagicMock()
    mock_choice.message.content = response_text
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_instance = MagicMock()
    mock_instance.chat.completions.create.return_value = mock_response
    return MagicMock(return_value=mock_instance)


# ---------------------------------------------------------------------------
# validate_data node (via graph)
# ---------------------------------------------------------------------------

def test_graph_sets_empty_validation_errors_for_valid_summary() -> None:
    from git_review.langgraph_pipeline import build_review_graph

    mock_openai_cls = _make_mock_openai()
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(openai_api_key="sk-fake")
        result = graph.invoke({"summary": _make_summary()})

    assert result["validation_errors"] == []


def test_graph_records_validation_error_for_empty_activity() -> None:
    from git_review.langgraph_pipeline import build_review_graph

    mock_openai_cls = _make_mock_openai()
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(openai_api_key="sk-fake")
        result = graph.invoke({"summary": _make_summary(commits=0, issues=0, prs=0)})

    assert len(result["validation_errors"]) == 1
    assert "No commits" in result["validation_errors"][0]


def test_graph_records_validation_error_for_invalid_date_range() -> None:
    from git_review.langgraph_pipeline import build_review_graph

    bad_summary = ReviewSummary(
        owner="acme",
        repo="app",
        since=UNTIL,   # since > until → invalid
        until=SINCE,
        commits=[
            Commit(
                sha="abc0000000001",
                message="fix: something",
                author="alice",
                authored_at=SINCE,
                url="https://github.com/acme/app/commit/abc0000000001",
                repo="acme/app",
            )
        ],
    )

    mock_openai_cls = _make_mock_openai()
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(openai_api_key="sk-fake")
        result = graph.invoke({"summary": bad_summary})

    assert any("Invalid date window" in e for e in result["validation_errors"])


# ---------------------------------------------------------------------------
# summarize node (via graph)
# ---------------------------------------------------------------------------

def test_graph_populates_summary_text() -> None:
    from git_review.langgraph_pipeline import build_review_graph

    expected = "## Highlights\nThis was a productive sprint."
    mock_openai_cls = _make_mock_openai(expected)
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(openai_api_key="sk-fake")
        result = graph.invoke({"summary": _make_summary()})

    assert result["summary_text"] == expected


def test_graph_sets_needs_refinement_false_for_long_summary() -> None:
    from git_review.langgraph_pipeline import build_review_graph, MIN_SUMMARY_CHARS

    long_text = "x" * (MIN_SUMMARY_CHARS + 50)
    mock_openai_cls = _make_mock_openai(long_text)
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(openai_api_key="sk-fake")
        result = graph.invoke({"summary": _make_summary()})

    assert result.get("needs_refinement") is False


# ---------------------------------------------------------------------------
# refine node (via graph)
# ---------------------------------------------------------------------------

def test_graph_triggers_refine_for_short_summary() -> None:
    """When the first summary is too short the refine node should produce a longer one."""
    from git_review.langgraph_pipeline import build_review_graph, MIN_SUMMARY_CHARS

    short_text = "OK"
    refined_text = "R" * (MIN_SUMMARY_CHARS + 100)

    mock_choice_short = MagicMock()
    mock_choice_short.message.content = short_text
    mock_response_short = MagicMock()
    mock_response_short.choices = [mock_choice_short]

    mock_choice_refined = MagicMock()
    mock_choice_refined.message.content = refined_text
    mock_response_refined = MagicMock()
    mock_response_refined.choices = [mock_choice_refined]

    mock_instance = MagicMock()
    # first call → short; second call (refine) → long
    mock_instance.chat.completions.create.side_effect = [
        mock_response_short,
        mock_response_refined,
    ]
    mock_openai_cls = MagicMock(return_value=mock_instance)

    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(openai_api_key="sk-fake")
        result = graph.invoke({"summary": _make_summary()})

    assert result["summary_text"] == refined_text
    assert mock_instance.chat.completions.create.call_count == 2


# ---------------------------------------------------------------------------
# Checkpointing / MemorySaver
# ---------------------------------------------------------------------------

def test_graph_accepts_memory_saver_checkpointer() -> None:
    from git_review.langgraph_pipeline import build_review_graph
    from langgraph.checkpoint.memory import MemorySaver

    mock_openai_cls = _make_mock_openai("A" * 500)
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(
            openai_api_key="sk-fake",
            checkpointer=MemorySaver(),
        )
        result = graph.invoke(
            {"summary": _make_summary()},
            config={"configurable": {"thread_id": "test-thread"}},
        )

    assert "summary_text" in result


# ---------------------------------------------------------------------------
# Custom model and base_url forwarded
# ---------------------------------------------------------------------------

def test_graph_uses_custom_model() -> None:
    from git_review.langgraph_pipeline import build_review_graph

    mock_openai_cls = _make_mock_openai("Summary")
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        graph = build_review_graph(openai_api_key="sk-fake", model="gpt-4o")
        graph.invoke({"summary": _make_summary()})

    instance = mock_openai_cls.return_value
    call_kwargs = instance.chat.completions.create.call_args[1]
    assert call_kwargs.get("model") == "gpt-4o"


def test_graph_uses_custom_base_url() -> None:
    from git_review.langgraph_pipeline import build_review_graph

    mock_openai_cls = _make_mock_openai("Summary")
    with patch("git_review.llm_client.OpenAI", mock_openai_cls):
        build_review_graph(
            openai_api_key="sk-fake",
            base_url="http://localhost:11434/v1",
        )

    _, kwargs = mock_openai_cls.call_args
    assert kwargs.get("base_url") == "http://localhost:11434/v1"
