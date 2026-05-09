"""Tests for GitHub -> ServiceNow source-of-truth sync helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock

from git_review.github_source_sync import (
    GitHubSourceOfTruthSync,
    get_repo_cursor,
    load_sync_cursor,
    save_sync_cursor,
    set_repo_cursor,
)
from git_review.models import Issue, Milestone


def _issue(number: int = 1) -> Issue:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return Issue(
        number=number,
        title=f"Issue {number}",
        state="open",
        author="alice",
        created_at=now,
        closed_at=None,
        url=f"https://github.com/acme/app/issues/{number}",
        repo="acme/app",
        labels=["bug"],
        assignees=["alice"],
        milestone="v1.0",
        github_id=1000 + number,
        updated_at=now,
    )


def _milestone(number: int = 1) -> Milestone:
    now = datetime(2025, 1, 1, tzinfo=timezone.utc)
    return Milestone(
        number=number,
        title=f"v{number}.0",
        state="open",
        description="",
        due_on=now,
        open_issues=1,
        closed_issues=0,
        url=f"https://github.com/acme/app/milestone/{number}",
        repo="acme/app",
        updated_at=now,
    )


def test_sync_repo_dry_run_reports_creates_without_writes() -> None:
    gh = MagicMock()
    gh.list_milestones.return_value = [_milestone(1)]
    gh.get_issues.return_value = [_issue(1)]

    snow = MagicMock()
    snow.query_one.side_effect = [None, None]

    syncer = GitHubSourceOfTruthSync(
        gh,
        snow,
        milestone_table="u_github_milestones",
        issue_table="u_github_issues",
    )

    report, _ = syncer.sync_repo("acme", "app", dry_run=True)

    assert report.milestones_created == 1
    assert report.issues_created == 1
    snow.create_record.assert_not_called()
    snow.update_record.assert_not_called()


def test_sync_repo_non_dry_run_updates_and_reports_conflicts() -> None:
    gh = MagicMock()
    gh.list_milestones.return_value = [_milestone(1)]
    gh.get_issues.return_value = [_issue(1)]

    snow = MagicMock()
    snow.query_one.side_effect = [
        {
            "sys_id": "ms1",
            "u_github_milestone_title": "Old Title",
            "u_github_repo": "acme/app",
            "u_github_milestone_number": "1",
        },
        {
            "sys_id": "is1",
            "u_github_title": "Old Issue Title",
            "u_github_repo": "acme/app",
            "u_github_issue_number": "1",
        },
    ]

    syncer = GitHubSourceOfTruthSync(
        gh,
        snow,
        milestone_table="u_github_milestones",
        issue_table="u_github_issues",
    )

    report, _ = syncer.sync_repo("acme", "app", dry_run=False)

    assert report.milestones_updated == 1
    assert report.issues_updated == 1
    assert len(report.conflicts) >= 2
    assert snow.update_record.call_count == 2


def test_sync_repo_optional_back_sync_applies_only_whitelisted_metadata() -> None:
    gh = MagicMock()
    gh.list_milestones.return_value = []
    gh.get_issues.return_value = [_issue(2)]

    snow = MagicMock()
    snow.query_one.side_effect = [
        None,  # issue upsert record missing -> create
        {
            "sys_id": "is2",
            "u_backsync_labels": "priority: high,bug",
            "u_backsync_assignees": "bob",
        },
    ]

    syncer = GitHubSourceOfTruthSync(
        gh,
        snow,
        milestone_table="u_github_milestones",
        issue_table="u_github_issues",
    )

    report, _ = syncer.sync_repo(
        "acme",
        "app",
        dry_run=False,
        allow_back_sync_fields=("labels", "assignees"),
    )

    assert report.back_sync_updates == 2
    assert gh.update_issue.call_count == 2


def test_cursor_helpers_round_trip(tmp_path) -> None:
    path = tmp_path / "cursor.json"
    data = load_sync_cursor(str(path))
    assert data == {}

    now = datetime(2025, 1, 1, 12, 0, tzinfo=timezone.utc)
    set_repo_cursor(data, "acme/app", now)
    save_sync_cursor(str(path), data)

    loaded = load_sync_cursor(str(path))
    parsed = get_repo_cursor(loaded, "acme/app")
    assert parsed is not None
    assert parsed == now
