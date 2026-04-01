"""Data models for git-review."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Commit:
    """A single Git commit on GitHub."""

    sha: str
    message: str
    author: str
    authored_at: datetime
    url: str
    repo: str


@dataclass
class Issue:
    """A GitHub issue (excludes pull requests)."""

    number: int
    title: str
    state: str
    author: str
    created_at: datetime
    closed_at: Optional[datetime]
    url: str
    repo: str
    labels: list[str] = field(default_factory=list)
    body: str = ""


@dataclass
class PullRequest:
    """A GitHub pull request."""

    number: int
    title: str
    state: str
    author: str
    created_at: datetime
    merged_at: Optional[datetime]
    url: str
    repo: str
    labels: list[str] = field(default_factory=list)
    body: str = ""


@dataclass
class ReviewSummary:
    """The aggregated result returned to callers and the CLI."""

    owner: str
    repo: str
    since: datetime
    until: datetime
    commits: list[Commit] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    pull_requests: list[PullRequest] = field(default_factory=list)
    summary_text: str = ""
