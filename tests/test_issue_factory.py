"""Tests for IssueFactory and IssueDraft."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from git_review.issue_factory import IssueDraft, IssueFactory, IssueList


# ---------------------------------------------------------------------------
# IssueDraft model
# ---------------------------------------------------------------------------

def test_issue_draft_requires_title_and_body() -> None:
    draft = IssueDraft(title="Fix login bug", body="Login fails on mobile")
    assert draft.title == "Fix login bug"
    assert draft.body == "Login fails on mobile"
    assert draft.labels == []
    assert draft.assignees == []


def test_issue_draft_with_labels_and_assignees() -> None:
    draft = IssueDraft(
        title="Add dark mode",
        body="Users want dark mode",
        labels=["enhancement", "good first issue"],
        assignees=["alice"],
    )
    assert "enhancement" in draft.labels
    assert "alice" in draft.assignees


def test_issue_list_contains_issues() -> None:
    issue_list = IssueList(
        issues=[
            IssueDraft(title="Issue 1", body="Body 1"),
            IssueDraft(title="Issue 2", body="Body 2", labels=["bug"]),
        ]
    )
    assert len(issue_list.issues) == 2
    assert issue_list.issues[1].labels == ["bug"]


# ---------------------------------------------------------------------------
# IssueFactory.parse_requirements
# ---------------------------------------------------------------------------

def _make_mock_openai_with_drafts(drafts: list[IssueDraft]) -> MagicMock:
    """Build a mock OpenAI client that returns *drafts* as parsed structured output."""
    mock_parsed = IssueList(issues=drafts)
    mock_message = MagicMock()
    mock_message.parsed = mock_parsed
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai_instance = MagicMock()
    mock_openai_instance.beta.chat.completions.parse.return_value = mock_response
    return MagicMock(return_value=mock_openai_instance)


def test_parse_requirements_returns_issue_drafts() -> None:
    expected_drafts = [
        IssueDraft(title="Add OAuth login", body="Users need OAuth", labels=["enhancement"]),
        IssueDraft(title="Fix crash on startup", body="App crashes", labels=["bug"]),
    ]
    mock_openai_cls = _make_mock_openai_with_drafts(expected_drafts)

    mock_gh = MagicMock()
    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(github_client=mock_gh, openai_api_key="sk-fake")
        drafts = factory.parse_requirements("## Requirements\n- Add OAuth\n- Fix crash")

    assert len(drafts) == 2
    assert drafts[0].title == "Add OAuth login"
    assert drafts[1].labels == ["bug"]


def test_parse_requirements_passes_markdown_to_llm() -> None:
    mock_openai_cls = _make_mock_openai_with_drafts([])
    mock_gh = MagicMock()
    markdown = "# My Requirements\n- Feature A\n- Feature B"

    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(github_client=mock_gh, openai_api_key="sk-fake")
        factory.parse_requirements(markdown)

    openai_instance = mock_openai_cls.return_value
    call_kwargs = openai_instance.beta.chat.completions.parse.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    user_msg = next(m["content"] for m in messages if m["role"] == "user")
    assert markdown in user_msg


def test_parse_requirements_uses_response_format() -> None:
    mock_openai_cls = _make_mock_openai_with_drafts([])
    mock_gh = MagicMock()

    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(github_client=mock_gh, openai_api_key="sk-fake")
        factory.parse_requirements("# Requirements")

    openai_instance = mock_openai_cls.return_value
    call_kwargs = openai_instance.beta.chat.completions.parse.call_args
    assert call_kwargs.kwargs.get("response_format") is IssueList


def test_factory_uses_custom_model() -> None:
    mock_openai_cls = _make_mock_openai_with_drafts([])
    mock_gh = MagicMock()

    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(
            github_client=mock_gh,
            openai_api_key="sk-fake",
            model="gpt-4o",
        )
        factory.parse_requirements("# Reqs")

    openai_instance = mock_openai_cls.return_value
    call_kwargs = openai_instance.beta.chat.completions.parse.call_args
    assert call_kwargs.kwargs.get("model") == "gpt-4o"


def test_factory_passes_base_url() -> None:
    mock_openai_cls = _make_mock_openai_with_drafts([])
    mock_gh = MagicMock()

    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        IssueFactory(
            github_client=mock_gh,
            openai_api_key="sk-fake",
            base_url="http://localhost:11434/v1",
        )

    _, kwargs = mock_openai_cls.call_args
    assert kwargs.get("base_url") == "http://localhost:11434/v1"


# ---------------------------------------------------------------------------
# IssueFactory.push_issues
# ---------------------------------------------------------------------------

def test_push_issues_calls_create_issue_for_each_draft() -> None:
    drafts = [
        IssueDraft(title="Issue A", body="Body A", labels=["bug"], assignees=["alice"]),
        IssueDraft(title="Issue B", body="Body B"),
    ]
    mock_gh = MagicMock()
    mock_gh.create_issue.side_effect = [
        {"number": 1, "html_url": "https://github.com/acme/app/issues/1"},
        {"number": 2, "html_url": "https://github.com/acme/app/issues/2"},
    ]
    mock_openai_cls = _make_mock_openai_with_drafts([])

    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(github_client=mock_gh, openai_api_key="sk-fake")
        results = factory.push_issues("acme", "app", drafts)

    assert len(results) == 2
    assert mock_gh.create_issue.call_count == 2

    first_call = mock_gh.create_issue.call_args_list[0]
    assert first_call.kwargs["title"] == "Issue A"
    assert first_call.kwargs["labels"] == ["bug"]
    assert first_call.kwargs["assignees"] == ["alice"]

    second_call = mock_gh.create_issue.call_args_list[1]
    assert second_call.kwargs["title"] == "Issue B"
    assert second_call.kwargs["labels"] is None
    assert second_call.kwargs["assignees"] is None


def test_push_issues_returns_empty_list_for_no_drafts() -> None:
    mock_gh = MagicMock()
    mock_openai_cls = _make_mock_openai_with_drafts([])

    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(github_client=mock_gh, openai_api_key="sk-fake")
        results = factory.push_issues("acme", "app", [])

    assert results == []
    mock_gh.create_issue.assert_not_called()


# ---------------------------------------------------------------------------
# IssueFactory custom system_prompt
# ---------------------------------------------------------------------------

def test_factory_accepts_valid_custom_system_prompt() -> None:
    mock_openai_cls = _make_mock_openai_with_drafts([])
    mock_gh = MagicMock()
    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(
            github_client=mock_gh,
            openai_api_key="sk-fake",
            system_prompt="Custom static prompt.",
        )
    assert factory._system_prompt == "Custom static prompt."


def test_factory_raises_on_template_variable_in_system_prompt() -> None:
    mock_openai_cls = _make_mock_openai_with_drafts([])
    mock_gh = MagicMock()
    with pytest.raises(ValueError, match="unknown variable"):
        with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
            IssueFactory(
                github_client=mock_gh,
                openai_api_key="sk-fake",
                system_prompt="{{ unknown_var }}",
            )


def test_factory_uses_custom_prompt_in_llm_call() -> None:
    mock_openai_cls = _make_mock_openai_with_drafts([])
    mock_gh = MagicMock()
    custom = "Only extract bug-related issues."

    with patch("git_review.issue_factory.OpenAI", mock_openai_cls):
        factory = IssueFactory(
            github_client=mock_gh,
            openai_api_key="sk-fake",
            system_prompt=custom,
        )
        factory.parse_requirements("# Requirements\n- Fix crash")

    openai_instance = mock_openai_cls.return_value
    call_kwargs = openai_instance.beta.chat.completions.parse.call_args
    messages = call_kwargs.kwargs.get("messages") or call_kwargs.args[0]
    system_msg = next(m["content"] for m in messages if m["role"] == "system")
    assert system_msg == custom
