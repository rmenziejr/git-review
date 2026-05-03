"""Tests for AgilePlanner and related helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from git_review.agile_planner import (
    AgilePlanner,
    _extract_explicit_dependencies,
    _extract_issue_refs,
    _trim,
    _LLMAgileResponse,
    _LLMDependency,
    _LLMSprint,
    _LLMLabelRec,
)
from git_review.models import (
    AgilePlanResult,
    Issue,
    IssueDependency,
    PullRequest,
    SprintRecommendation,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_DT = datetime(2024, 1, 1, tzinfo=timezone.utc)


def _make_issue(
    number: int,
    title: str = "Issue",
    body: str = "",
    labels: list[str] | None = None,
    assignees: list[str] | None = None,
    github_id: int | None = None,
) -> Issue:
    return Issue(
        number=number,
        title=title,
        state="open",
        author="alice",
        created_at=_DT,
        closed_at=None,
        url=f"https://github.com/acme/app/issues/{number}",
        repo="acme/app",
        body=body,
        labels=labels or [],
        assignees=assignees or [],
        github_id=github_id or (1000 + number),
    )


def _make_pr(number: int, title: str = "PR", body: str = "", draft: bool = False) -> PullRequest:
    return PullRequest(
        number=number,
        title=title,
        state="open",
        author="bob",
        created_at=_DT,
        merged_at=None,
        url=f"https://github.com/acme/app/pull/{number}",
        repo="acme/app",
        body=body,
        draft=draft,
    )


# ---------------------------------------------------------------------------
# _trim helper
# ---------------------------------------------------------------------------


def test_trim_short_string_unchanged() -> None:
    assert _trim("hello world", 200) == "hello world"


def test_trim_truncates_long_string() -> None:
    result = _trim("a" * 300, 200)
    assert len(result) == 200
    assert result.endswith("…")


def test_trim_collapses_whitespace() -> None:
    assert _trim("  foo   bar  ") == "foo bar"


# ---------------------------------------------------------------------------
# _extract_issue_refs helper
# ---------------------------------------------------------------------------


def test_extract_issue_refs_basic() -> None:
    assert _extract_issue_refs("Fixes #12 and #34") == [12, 34]


def test_extract_issue_refs_empty() -> None:
    assert _extract_issue_refs("") == []
    assert _extract_issue_refs(None) == []  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _extract_explicit_dependencies
# ---------------------------------------------------------------------------


def test_explicit_dep_blocked_by() -> None:
    issues = [_make_issue(5), _make_issue(3)]
    issue5 = _make_issue(5, body="This is blocked by #3")
    deps = _extract_explicit_dependencies([issue5, issues[1]], [])
    assert len(deps) == 1
    dep = deps[0]
    assert dep.from_issue == 5
    assert dep.to_issue == 3
    assert dep.dep_type == "blocked-by"
    assert dep.confidence == 1.0
    assert dep.source == "explicit"


def test_explicit_dep_blocks() -> None:
    issues = [_make_issue(1, body="blocks #2"), _make_issue(2)]
    deps = _extract_explicit_dependencies(issues, [])
    # "#1 blocks #2" → #2 is blocked by #1
    assert len(deps) == 1
    assert deps[0].from_issue == 2
    assert deps[0].to_issue == 1
    assert deps[0].dep_type == "blocked-by"


def test_explicit_dep_depends_on() -> None:
    issues = [_make_issue(7, body="depends on #8"), _make_issue(8)]
    deps = _extract_explicit_dependencies(issues, [])
    assert any(d.from_issue == 7 and d.to_issue == 8 and d.dep_type == "blocked-by" for d in deps)


def test_explicit_dep_pr_closes_issue() -> None:
    issues = [_make_issue(10)]
    pr = _make_pr(20, body="Closes #10")
    deps = _extract_explicit_dependencies(issues, [pr])
    # PR 20 closes issue 10 → PR 20 blocked-by issue 10
    assert any(d.from_issue == 20 and d.to_issue == 10 for d in deps)


def test_explicit_dep_no_self_reference() -> None:
    issues = [_make_issue(5, body="blocks #5")]
    deps = _extract_explicit_dependencies(issues, [])
    assert deps == []


def test_explicit_dep_skips_unknown_ref() -> None:
    issues = [_make_issue(5, body="blocked by #999")]
    deps = _extract_explicit_dependencies(issues, [])
    # #999 not in issue set → dep is skipped
    assert deps == []


# ---------------------------------------------------------------------------
# AgilePlanner._build_context_message
# ---------------------------------------------------------------------------


def _make_planner_with_mock_gh() -> tuple[AgilePlanner, MagicMock]:
    mock_gh = MagicMock()
    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(
            github_client=mock_gh,
            openai_api_key="sk-test",
            sprint_capacity=5,
            num_sprints=2,
        )
    return planner, mock_gh


def test_build_context_message_contains_issue_info() -> None:
    planner, _ = _make_planner_with_mock_gh()
    issues = [
        _make_issue(1, title="Fix login bug", labels=["bug"], assignees=["alice"]),
        _make_issue(2, title="Add dark mode"),
    ]
    prs = [_make_pr(10, title="WIP: login fix", body="Closes #1")]
    existing_deps: list[IssueDependency] = []

    msg = planner._build_context_message(issues, prs, existing_deps)

    assert "Sprint capacity: 5" in msg
    assert "Number of sprints to plan: 2" in msg
    assert "#1: Fix login bug" in msg
    assert "#2: Add dark mode" in msg
    assert "bug" in msg
    assert "alice" in msg
    assert "#10" in msg


def test_build_context_message_shows_existing_deps() -> None:
    planner, _ = _make_planner_with_mock_gh()
    issues = [_make_issue(1), _make_issue(2)]
    existing_deps = [
        IssueDependency(
            from_issue=2, to_issue=1, dep_type="blocked-by",
            confidence=1.0, reason="already recorded", source="github",
        )
    ]
    msg = planner._build_context_message(issues, [], existing_deps)
    assert "Existing Blocking Relationships" in msg
    assert "#2 is blocked-by #1" in msg
    assert "source: github" in msg


# ---------------------------------------------------------------------------
# AgilePlanner._plan (LLM parsing)
# ---------------------------------------------------------------------------


def _mock_llm_response(deps=None, sprints=None, labels=None, summary="Test summary") -> _LLMAgileResponse:
    return _LLMAgileResponse(
        dependencies=deps or [],
        sprint_plan=sprints or [],
        label_recommendations=labels or [],
        summary=summary,
    )


def _make_mock_openai(llm_response: _LLMAgileResponse) -> MagicMock:
    mock_message = MagicMock()
    mock_message.parsed = llm_response
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client = MagicMock()
    mock_client.beta.chat.completions.parse.return_value = mock_response
    return mock_client


def test_plan_merges_explicit_and_llm_deps() -> None:
    issues = [_make_issue(1), _make_issue(2), _make_issue(3)]
    prs: list[PullRequest] = []

    # Issue 2 has explicit "blocked by #1"
    issues[1] = _make_issue(2, body="blocked by #1")

    llm_response = _mock_llm_response(
        deps=[
            _LLMDependency(from_issue=3, to_issue=2, type="blocked-by", confidence=0.7, reason="inferred"),
        ],
        sprints=[
            _LLMSprint(sprint_number=1, issues=[1], theme="Foundation", rationale="Start here.", deferred=[2, 3]),
        ],
        summary="All good.",
    )

    mock_gh = MagicMock()
    mock_gh.get_issue_blocked_by.return_value = []  # no existing GitHub deps

    with patch("git_review.agile_planner.OpenAI") as MockOpenAI:
        instance = MockOpenAI.return_value
        instance.beta.chat.completions.parse.return_value = (
            _make_mock_openai(llm_response).beta.chat.completions.parse.return_value
        )
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")
        planner._client = _make_mock_openai(llm_response)
        existing_deps = _extract_explicit_dependencies(issues, prs)
        result = planner._plan("acme", "app", issues, prs, existing_deps)

    # Should have the explicit dep (#2 blocked-by #1) and the LLM dep (#3 blocked-by #2)
    dep_pairs = {(d.from_issue, d.to_issue) for d in result.dependencies}
    assert (2, 1) in dep_pairs
    assert (3, 2) in dep_pairs

    # LLM dep should have source="llm"
    llm_dep = next(d for d in result.dependencies if d.from_issue == 3)
    assert llm_dep.source == "llm"

    assert result.summary_text == "All good."
    assert len(result.sprints) == 1
    assert result.sprints[0].theme == "Foundation"


def test_plan_deduplicates_existing_github_deps() -> None:
    """LLM should not duplicate relationships already fetched from GitHub."""
    issues = [_make_issue(1), _make_issue(2)]

    existing_github_deps = [
        IssueDependency(
            from_issue=2, to_issue=1, dep_type="blocked-by",
            confidence=1.0, reason="in github", source="github",
        )
    ]

    # LLM returns the same relationship
    llm_response = _mock_llm_response(
        deps=[
            _LLMDependency(from_issue=2, to_issue=1, type="blocked-by", confidence=0.9, reason="llm says so"),
        ],
        sprints=[_LLMSprint(sprint_number=1, issues=[1, 2], theme="All", rationale="do it", deferred=[])],
    )

    mock_gh = MagicMock()
    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")
        planner._client = _make_mock_openai(llm_response)
        result = planner._plan("acme", "app", issues, [], existing_github_deps)

    dep_pairs = [(d.from_issue, d.to_issue) for d in result.dependencies]
    # Should appear exactly once
    assert dep_pairs.count((2, 1)) == 1


# ---------------------------------------------------------------------------
# AgilePlanner.apply_relationships
# ---------------------------------------------------------------------------


def test_apply_relationships_dry_run_makes_no_api_calls() -> None:
    mock_gh = MagicMock()
    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")

    issues = [_make_issue(1, github_id=1001), _make_issue(2, github_id=1002)]
    deps = [
        IssueDependency(
            from_issue=2, to_issue=1, dep_type="blocked-by",
            confidence=0.8, reason="inferred", source="llm",
        )
    ]
    result = AgilePlanResult(
        owner="acme", repo="app",
        issues=issues, pull_requests=[], dependencies=deps,
    )

    responses = planner.apply_relationships("acme", "app", result, dry_run=True)

    assert responses == []
    mock_gh.add_issue_blocked_by.assert_not_called()


def test_apply_relationships_calls_github_api() -> None:
    mock_gh = MagicMock()
    mock_gh.add_issue_blocked_by.return_value = {"id": 1}

    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")

    issues = [_make_issue(1, github_id=1001), _make_issue(2, github_id=1002)]
    deps = [
        IssueDependency(
            from_issue=2, to_issue=1, dep_type="blocked-by",
            confidence=0.8, reason="inferred", source="llm",
        )
    ]
    result = AgilePlanResult(
        owner="acme", repo="app",
        issues=issues, pull_requests=[], dependencies=deps,
    )

    responses = planner.apply_relationships("acme", "app", result, dry_run=False)

    # #2 is blocked by #1 (github_id=1001)
    mock_gh.add_issue_blocked_by.assert_called_once_with("acme", "app", 2, 1001)
    assert len(responses) == 1


def test_apply_relationships_skips_explicit_and_github_sources() -> None:
    mock_gh = MagicMock()
    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")

    issues = [_make_issue(1, github_id=1001), _make_issue(2, github_id=1002)]
    deps = [
        IssueDependency(
            from_issue=2, to_issue=1, dep_type="blocked-by",
            confidence=1.0, reason="already in github", source="github",
        ),
        IssueDependency(
            from_issue=2, to_issue=1, dep_type="blocked-by",
            confidence=1.0, reason="explicit text", source="explicit",
        ),
    ]
    result = AgilePlanResult(
        owner="acme", repo="app",
        issues=issues, pull_requests=[], dependencies=deps,
    )

    responses = planner.apply_relationships("acme", "app", result, dry_run=False)

    mock_gh.add_issue_blocked_by.assert_not_called()
    assert responses == []


# ---------------------------------------------------------------------------
# AgilePlanner.apply_labels
# ---------------------------------------------------------------------------


def test_apply_labels_dry_run_makes_no_api_calls() -> None:
    mock_gh = MagicMock()
    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")

    issues = [_make_issue(1, labels=["bug"])]
    result = AgilePlanResult(
        owner="acme", repo="app",
        issues=issues, pull_requests=[], dependencies=[],
        label_recommendations={1: ["priority: high"]},
    )

    responses = planner.apply_labels("acme", "app", result, dry_run=True)

    assert responses == []
    mock_gh.update_issue_labels.assert_not_called()


def test_apply_labels_merges_and_calls_api() -> None:
    mock_gh = MagicMock()
    mock_gh.update_issue_labels.return_value = {"number": 1}

    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")

    issues = [_make_issue(1, labels=["bug"])]
    result = AgilePlanResult(
        owner="acme", repo="app",
        issues=issues, pull_requests=[], dependencies=[],
        label_recommendations={1: ["priority: high"]},
    )

    responses = planner.apply_labels("acme", "app", result, dry_run=False)

    mock_gh.update_issue_labels.assert_called_once()
    call_args = mock_gh.update_issue_labels.call_args
    labels_applied = call_args[0][3]  # 4th positional arg
    assert "bug" in labels_applied
    assert "priority: high" in labels_applied
    assert len(responses) == 1


def test_apply_labels_skips_if_no_change() -> None:
    mock_gh = MagicMock()
    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")

    issues = [_make_issue(1, labels=["bug", "priority: high"])]
    result = AgilePlanResult(
        owner="acme", repo="app",
        issues=issues, pull_requests=[], dependencies=[],
        label_recommendations={1: ["priority: high"]},  # already present
    )

    planner.apply_labels("acme", "app", result, dry_run=False)
    mock_gh.update_issue_labels.assert_not_called()


# ---------------------------------------------------------------------------
# AgilePlanner.analyse (integration – mocked GitHub + LLM)
# ---------------------------------------------------------------------------


def test_analyse_calls_github_and_returns_result() -> None:
    issues = [_make_issue(1), _make_issue(2, body="blocked by #1")]
    prs = [_make_pr(10)]

    llm_response = _mock_llm_response(
        sprints=[
            _LLMSprint(sprint_number=1, issues=[1], theme="Theme", rationale="Do #1 first.", deferred=[2]),
        ],
        summary="Two issues found.",
    )

    mock_gh = MagicMock()
    mock_gh.get_open_issues.return_value = issues
    mock_gh.get_open_pull_requests.return_value = prs
    mock_gh.get_issue_blocked_by.return_value = []

    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")
        planner._client = _make_mock_openai(llm_response)

    result = planner.analyse("acme", "app")

    mock_gh.get_open_issues.assert_called_once_with("acme", "app")
    mock_gh.get_open_pull_requests.assert_called_once_with("acme", "app")
    assert isinstance(result, AgilePlanResult)
    assert result.owner == "acme"
    assert result.repo == "app"
    assert len(result.issues) == 2
    assert result.summary_text == "Two issues found."
    # Explicit dep from body parsing
    assert any(d.from_issue == 2 and d.to_issue == 1 for d in result.dependencies)


def test_analyse_org_aggregates_across_repos() -> None:
    issues_a = [_make_issue(1)]
    issues_b = [_make_issue(2)]
    prs_a: list[PullRequest] = []
    prs_b: list[PullRequest] = []

    llm_response = _mock_llm_response(
        sprints=[_LLMSprint(sprint_number=1, issues=[1, 2], theme="All", rationale="Both.", deferred=[])],
    )

    mock_gh = MagicMock()
    mock_gh.list_repos.return_value = ["repo-a", "repo-b"]
    mock_gh.get_open_issues.side_effect = [issues_a, issues_b]
    mock_gh.get_open_pull_requests.side_effect = [prs_a, prs_b]
    mock_gh.get_issue_blocked_by.return_value = []

    with patch("git_review.agile_planner.OpenAI"):
        planner = AgilePlanner(github_client=mock_gh, openai_api_key="sk-test")
        planner._client = _make_mock_openai(llm_response)

    result = planner.analyse_org("acme")

    assert result.owner == "acme"
    assert result.repo == "*"
    assert len(result.issues) == 2


# ---------------------------------------------------------------------------
# AgilePlanResult and model dataclasses
# ---------------------------------------------------------------------------


def test_agile_plan_result_defaults() -> None:
    result = AgilePlanResult(owner="org", repo="repo")
    assert result.issues == []
    assert result.pull_requests == []
    assert result.dependencies == []
    assert result.sprints == []
    assert result.summary_text == ""
    assert result.label_recommendations == {}


def test_issue_dependency_fields() -> None:
    dep = IssueDependency(
        from_issue=5, to_issue=3, dep_type="blocked-by",
        confidence=0.9, reason="test reason", source="llm",
    )
    assert dep.from_issue == 5
    assert dep.dep_type == "blocked-by"
    assert dep.source == "llm"


def test_sprint_recommendation_defaults() -> None:
    sprint = SprintRecommendation(sprint_number=1)
    assert sprint.issues == []
    assert sprint.deferred == []
    assert sprint.theme == ""
