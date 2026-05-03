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
    additions: int = 0
    deletions: int = 0


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
    comments: int = 0
    assignees: list[str] = field(default_factory=list)
    milestone: Optional[str] = None
    github_id: Optional[int] = None


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
    review_comments: int = 0
    commits_count: int = 0
    additions: int = 0
    deletions: int = 0
    changed_files: int = 0
    draft: bool = False
    base_branch: str = ""
    head_branch: str = ""
    requested_reviewers: list[str] = field(default_factory=list)
    reviewer_comments: dict[str, int] = field(default_factory=dict)


@dataclass
class Release:
    """A GitHub repository release."""

    tag: str
    name: str
    body: str
    created_at: datetime
    published_at: Optional[datetime]
    url: str
    repo: str
    author: str = ""
    prerelease: bool = False
    draft: bool = False


@dataclass
class Contributor:
    """A contributor to a GitHub repository."""

    login: str
    contributions: int
    url: str
    repo: str


@dataclass
class Milestone:
    """A GitHub repository milestone."""

    number: int
    title: str
    state: str
    description: str = ""
    due_on: Optional[datetime] = None
    open_issues: int = 0
    closed_issues: int = 0
    url: str = ""
    repo: str = ""


@dataclass
class AuthorSummary:
    """Activity for a single author partitioned out of a :class:`ReviewSummary`."""

    author: str
    commits: list[Commit] = field(default_factory=list)
    issues: list[Issue] = field(default_factory=list)
    pull_requests: list[PullRequest] = field(default_factory=list)
    releases: list[Release] = field(default_factory=list)


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
    releases: list[Release] = field(default_factory=list)
    contributors: list[Contributor] = field(default_factory=list)
    summary_text: str = ""


@dataclass
class IssueDependency:
    """A directed dependency relationship between two issues."""

    from_issue: int
    to_issue: int
    dep_type: str
    confidence: float
    reason: str
    source: str = "llm"


@dataclass
class SprintRecommendation:
    """Issues recommended for a single sprint."""

    sprint_number: int
    issues: list[int] = field(default_factory=list)
    theme: str = ""
    rationale: str = ""
    deferred: list[int] = field(default_factory=list)


@dataclass
class AgilePlanResult:
    """Top-level result from the agile planner."""

    owner: str
    repo: str
    issues: list[Issue] = field(default_factory=list)
    pull_requests: list[PullRequest] = field(default_factory=list)
    dependencies: list[IssueDependency] = field(default_factory=list)
    sprints: list[SprintRecommendation] = field(default_factory=list)
    summary_text: str = ""
    label_recommendations: dict[int, list[str]] = field(default_factory=dict)
