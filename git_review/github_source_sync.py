"""GitHub-source-of-truth sync from GitHub to ServiceNow."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin

import requests

from .github_client import GitHubClient
from .models import Issue, Milestone


def _iso(dt: Optional[datetime]) -> str:
    if dt is None:
        return ""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_sync_cursor(path: str) -> dict[str, str]:
    p = Path(path)
    if not p.exists():
        return {}
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(raw, dict):
        return {}
    return {str(k): str(v) for k, v in raw.items()}


def save_sync_cursor(path: str, cursor_data: dict[str, str]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(cursor_data, indent=2, sort_keys=True), encoding="utf-8")


def get_repo_cursor(cursor_data: dict[str, str], repo_key: str) -> Optional[datetime]:
    raw = cursor_data.get(repo_key)
    if not raw:
        return None
    value = raw.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(value).astimezone(timezone.utc)
    except ValueError:
        return None


def set_repo_cursor(cursor_data: dict[str, str], repo_key: str, value: datetime) -> None:
    cursor_data[repo_key] = _iso(value)


@dataclass
class SyncConflict:
    entity: str
    github_key: str
    field: str
    github_value: str
    servicenow_value: str


@dataclass
class SyncReport:
    milestones_scanned: int = 0
    milestones_created: int = 0
    milestones_updated: int = 0
    issues_scanned: int = 0
    issues_created: int = 0
    issues_updated: int = 0
    back_sync_updates: int = 0
    conflicts: list[SyncConflict] = field(default_factory=list)


class ServiceNowClient:
    """Minimal ServiceNow Table API client."""

    def __init__(
        self,
        instance_url: str,
        *,
        user: Optional[str] = None,
        password: Optional[str] = None,
        token: Optional[str] = None,
        timeout: int = 30,
    ) -> None:
        if not token and not (user and password):
            raise ValueError("ServiceNow auth required: pass token or user/password.")
        self._base_url = instance_url.rstrip("/") + "/"
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update(
            {
                "Accept": "application/json",
                "Content-Type": "application/json",
            }
        )
        if token:
            self._session.headers["Authorization"] = f"Bearer {token}"
        else:
            self._session.auth = (user or "", password or "")

    def _request(self, method: str, path: str, **kwargs: object) -> object:
        url = urljoin(self._base_url, path.lstrip("/"))
        response = self._session.request(method, url, timeout=self._timeout, **kwargs)
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict) and "result" in data:
            return data["result"]
        return data

    def query_one(self, table: str, query: str) -> Optional[dict]:
        result = self._request(
            "GET",
            f"/api/now/table/{table}",
            params={"sysparm_query": query, "sysparm_limit": 1},
        )
        if isinstance(result, list):
            return result[0] if result else None
        if isinstance(result, dict):
            return result
        return None

    def create_record(self, table: str, payload: dict) -> dict:
        result = self._request("POST", f"/api/now/table/{table}", json=payload)
        return result if isinstance(result, dict) else {}

    def update_record(self, table: str, sys_id: str, payload: dict) -> dict:
        result = self._request("PATCH", f"/api/now/table/{table}/{sys_id}", json=payload)
        return result if isinstance(result, dict) else {}


class GitHubSourceOfTruthSync:
    """Sync changed GitHub milestones/issues into ServiceNow (one-way default)."""

    def __init__(
        self,
        github_client: GitHubClient,
        servicenow_client: ServiceNowClient,
        *,
        milestone_table: str,
        issue_table: str,
    ) -> None:
        self._gh = github_client
        self._sn = servicenow_client
        self._milestone_table = milestone_table
        self._issue_table = issue_table

    def sync_repo(
        self,
        owner: str,
        repo: str,
        *,
        since: Optional[datetime] = None,
        dry_run: bool = True,
        allow_back_sync_fields: tuple[str, ...] = (),
    ) -> tuple[SyncReport, datetime]:
        now = datetime.now(tz=timezone.utc)
        report = SyncReport()
        repo_key = f"{owner}/{repo}"

        milestones = self._gh.list_milestones(owner, repo, state="all")
        if since:
            milestones = [
                m for m in milestones if m.updated_at is None or m.updated_at >= since
            ]
        report.milestones_scanned = len(milestones)

        issues_since = since or datetime(1970, 1, 1, tzinfo=timezone.utc)
        issues = self._gh.get_issues(owner, repo, issues_since, now, state="all")
        report.issues_scanned = len(issues)

        for milestone in milestones:
            action = self._sync_milestone(repo_key, milestone, dry_run=dry_run, report=report)
            if action == "created":
                report.milestones_created += 1
            elif action == "updated":
                report.milestones_updated += 1

        for issue in issues:
            action = self._sync_issue(repo_key, issue, dry_run=dry_run, report=report)
            if action == "created":
                report.issues_created += 1
            elif action == "updated":
                report.issues_updated += 1

        if allow_back_sync_fields:
            for issue in issues:
                report.back_sync_updates += self._back_sync_issue_metadata(
                    repo_key, issue, allow_back_sync_fields, dry_run=dry_run
                )

        updated_values = [now]
        updated_values += [m.updated_at for m in milestones if m.updated_at]
        updated_values += [i.updated_at for i in issues if i.updated_at]
        next_cursor = max(updated_values)
        return report, next_cursor

    def _sync_milestone(
        self,
        repo_key: str,
        milestone: Milestone,
        *,
        dry_run: bool,
        report: SyncReport,
    ) -> str:
        query = (
            f"u_github_repo={repo_key}"
            f"^u_github_milestone_number={milestone.number}"
        )
        payload = {
            "u_github_repo": repo_key,
            "u_github_milestone_number": str(milestone.number),
            "u_github_milestone_url": milestone.url,
            "u_github_milestone_title": milestone.title,
            "u_github_state": milestone.state,
            "u_github_due_on": _iso(milestone.due_on),
            "u_github_updated_at": _iso(milestone.updated_at),
            "u_github_source_of_truth": "github",
        }
        existing = self._sn.query_one(self._milestone_table, query)
        return self._upsert_record(
            table=self._milestone_table,
            entity="milestone",
            key=f"{repo_key}#{milestone.number}",
            payload=payload,
            existing=existing,
            dry_run=dry_run,
            report=report,
        )

    def _sync_issue(
        self,
        repo_key: str,
        issue: Issue,
        *,
        dry_run: bool,
        report: SyncReport,
    ) -> str:
        query = f"u_github_repo={repo_key}^u_github_issue_number={issue.number}"
        payload = {
            "u_github_repo": repo_key,
            "u_github_issue_number": str(issue.number),
            "u_github_issue_id": str(issue.github_id or ""),
            "u_github_issue_url": issue.url,
            "u_github_title": issue.title,
            "u_github_body": issue.body or "",
            "u_github_state": issue.state,
            "u_github_labels": ",".join(issue.labels),
            "u_github_assignees": ",".join(issue.assignees),
            "u_github_milestone_title": issue.milestone or "",
            "u_github_updated_at": _iso(issue.updated_at),
            "u_github_source_of_truth": "github",
        }
        existing = self._sn.query_one(self._issue_table, query)
        return self._upsert_record(
            table=self._issue_table,
            entity="issue",
            key=f"{repo_key}#{issue.number}",
            payload=payload,
            existing=existing,
            dry_run=dry_run,
            report=report,
        )

    def _upsert_record(
        self,
        *,
        table: str,
        entity: str,
        key: str,
        payload: dict,
        existing: Optional[dict],
        dry_run: bool,
        report: SyncReport,
    ) -> str:
        if not existing:
            if not dry_run:
                self._sn.create_record(table, payload)
            return "created"

        changes = {
            field: value
            for field, value in payload.items()
            if str(existing.get(field, "")) != str(value)
        }
        if not changes:
            return "unchanged"

        for field, github_value in changes.items():
            servicenow_value = str(existing.get(field, ""))
            if servicenow_value:
                report.conflicts.append(
                    SyncConflict(
                        entity=entity,
                        github_key=key,
                        field=field,
                        github_value=str(github_value),
                        servicenow_value=servicenow_value,
                    )
                )

        if not dry_run:
            sys_id = str(existing.get("sys_id", ""))
            if not sys_id:
                raise ValueError(f"Cannot update {entity} {key}: missing sys_id in ServiceNow record.")
            self._sn.update_record(table, sys_id, changes)
        return "updated"

    def _back_sync_issue_metadata(
        self,
        repo_key: str,
        issue: Issue,
        allow_fields: tuple[str, ...],
        *,
        dry_run: bool,
    ) -> int:
        allow = set(allow_fields)
        if not allow:
            return 0

        query = f"u_github_repo={repo_key}^u_github_issue_number={issue.number}"
        existing = self._sn.query_one(self._issue_table, query)
        if not existing:
            return 0

        updates = 0
        owner, repo = repo_key.split("/", 1)

        if "labels" in allow and "u_backsync_labels" in existing:
            labels_raw = str(existing.get("u_backsync_labels", "")).strip()
            labels = sorted({x.strip() for x in labels_raw.split(",") if x.strip()})
            if labels and set(labels) != set(issue.labels):
                if not dry_run:
                    self._gh.update_issue(owner, repo, issue.number, labels=labels)
                updates += 1

        if "assignees" in allow and "u_backsync_assignees" in existing:
            assignees_raw = str(existing.get("u_backsync_assignees", "")).strip()
            assignees = sorted({x.strip() for x in assignees_raw.split(",") if x.strip()})
            if assignees and set(assignees) != set(issue.assignees):
                if not dry_run:
                    self._gh.update_issue(owner, repo, issue.number, assignees=assignees)
                updates += 1

        return updates
