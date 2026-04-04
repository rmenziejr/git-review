# git-review

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![PyPI](https://img.shields.io/pypi/v/git-review)](https://pypi.org/project/git-review/)
[![Python](https://img.shields.io/pypi/pyversions/git-review)](https://pypi.org/project/git-review/)

A Python CLI tool and SDK that connects to the GitHub API to extract commits,
issues, and pull requests submitted within a configurable time window, then
generates a concise summary via an LLM (OpenAI or any compatible API).

A [VS Code extension](#vs-code-extension) is also available for a GUI workflow
directly inside your editor.

---

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Quick Start](#quick-start)
- [VS Code Extension](#vs-code-extension)
- [Python SDK](#python-sdk)
- [LangGraph Pipeline](#langgraph-pipeline)
- [Project Layout](#project-layout)
- [Running Tests](#running-tests)
- [License](#license)

---

## Features

| Feature | Details |
|---|---|
| **GitHub data extraction** | Commits, issues, and pull requests |
| **Flexible time windows** | Explicit `--since`/`--until` dates *or* a `--days` delta |
| **Author filter** | Narrow commits to a specific GitHub user |
| **AI summarisation** | OpenAI (default) or any OpenAI-compatible endpoint |
| **Markdown output** | Save the AI summary to a `.md` file with `--output` |
| **Commit message generator** | Write a Conventional Commit message from your staged diff |
| **Issue generator** | Parse a markdown requirements file and create GitHub issues via LLM |
| **LangGraph pipeline** | Multi-step review graph with checkpointing, conditional refinement, and middleware |
| **Rich terminal output** | Colour-coded tables and a formatted summary panel |
| **Python SDK** | Use `GitHubClient` and `LLMClient` directly in your own code |
| **VS Code extension** | Run all commands from the Command Palette inside VS Code |

---

## Installation

### With pip

```bash
pip install git-review
```

### With uvx (no install needed)

Run directly from PyPI without installing into your environment:

```bash
uvx git-review review --repo owner/repo --no-summary
uvx git-review commit-message
```

### From source

```bash
git clone https://github.com/rmenziejr/git-review.git
cd git-review
uv sync
```

### Development (with test dependencies)

```bash
uv sync --extra dev
```

### With LangGraph support

```bash
pip install 'git-review[langgraph]'
# or with uv:
uv sync --extra langgraph
```

---

## Configuration

Copy `.env.example` to `.env` and fill in your credentials:

```bash
cp .env.example .env
```

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub PAT (avoids `--token`) |
| `OPENAI_API_KEY` | OpenAI API key (avoids `--openai-key`) |
| `OPENAI_BASE_URL` | Custom API base URL (avoids `--base-url`) |
| `GITREVIEW_REPO` | Default repository (avoids `--repo`) |

---

## Quick Start

### Review a repository

```bash
# Last 7 days for a single repo – table output only (no LLM key required)
git-review review --repo owner/repo --no-summary

# All repos for a GitHub user or org
git-review review --owner myorg --days 14 --no-summary

# Use a GitHub token to avoid rate limits
git-review review --repo owner/repo --token ghp_xxx --no-summary

# AI summary – single repo, last 14 days
git-review review --repo owner/repo --days 14 \
  --token ghp_xxx --openai-key sk-xxx

# AI summary saved to a markdown file
git-review review --repo owner/repo --days 14 \
  --token ghp_xxx --openai-key sk-xxx --output summary.md

# AI summary – ALL repos for an org
git-review review --owner myorg --days 7 \
  --token ghp_xxx --openai-key sk-xxx

# Explicit date range
git-review review --repo owner/repo \
  --since 2024-01-01 --until 2024-01-31 \
  --token ghp_xxx --openai-key sk-xxx

# Filter commits by author
git-review review --repo owner/repo --days 30 --author myusername

# Use a local Ollama model (no OpenAI key needed)
git-review review --repo owner/repo --days 7 \
  --base-url http://localhost:11434/v1 --model llama3
```

### Generate a commit message

Stage your changes, then run:

```bash
# Generate a Conventional Commit message from your staged diff
git-review commit-message

# Use a different model or a local Ollama instance
git-review commit-message --model gpt-4o
git-review commit-message --base-url http://localhost:11434/v1 --model llama3

# Point at a different git repository
git-review commit-message --repo-path /path/to/other/repo
```

The command reads the staged diff (`git diff --staged`) – or the unstaged
diff if nothing is staged – and asks the LLM to write a
[Conventional Commits](https://www.conventionalcommits.org/) message.

### Create GitHub issues from requirements

```bash
# Parse a markdown requirements file and push issues to GitHub
git-review create-issues --repo owner/repo --requirements requirements.md \
  --token ghp_xxx --openai-key sk-xxx
```

---

## VS Code Extension

The git-review VS Code extension lets you run all commands from the Command
Palette without leaving your editor.

### Installing the extension

Download the latest `git-review-*.vsix` from the
[Releases page](https://github.com/rmenziejr/git-review/releases) and install
it with:

```bash
code --install-extension git-review-<version>.vsix
```

Or via the VS Code UI: **Extensions → ⋯ → Install from VSIX…**

> **Prerequisite:** `git-review` must be installed and on your `PATH`:
> ```bash
> pip install git-review
> # or
> uv tool install git-review
> ```

### Extension commands

| Command Palette entry | Description |
|---|---|
| **Git Review: Review Repository** | Fetch GitHub activity and generate an AI summary |
| **Git Review: Generate Commit Message** | Write a Conventional Commit for your staged diff |
| **Git Review: Create Issues from Requirements** | Parse a markdown file and push GitHub issues |

### Extension settings

Open **File → Preferences → Settings** and search for **"git review"**.

| Setting | Description | CLI / Env equivalent |
|---|---|---|
| `gitReview.githubToken` | GitHub personal access token | `--token` / `GITHUB_TOKEN` |
| `gitReview.openaiApiKey` | OpenAI API key | `--openai-key` / `OPENAI_API_KEY` |
| `gitReview.openaiBaseUrl` | Custom OpenAI-compatible base URL | `--base-url` / `OPENAI_BASE_URL` |
| `gitReview.defaultRepo` | Default repository (`owner/repo`) | `--repo` / `GITREVIEW_REPO` |
| `gitReview.model` | LLM model (default: `gpt-4o-mini`) | `--model` |
| `gitReview.defaultDays` | Days to look back (default: `7`) | `--days` |

---

## Python SDK

```python
from datetime import datetime, timedelta, timezone
from git_review import GitHubClient, LLMClient, ReviewSummary

since = datetime.now(tz=timezone.utc) - timedelta(days=14)
until = datetime.now(tz=timezone.utc)

gh = GitHubClient(token="ghp_xxx")

# --- Single repo ---
commits = gh.get_commits("owner", "repo", since, until)
issues  = gh.get_issues("owner", "repo", since, until)
prs     = gh.get_pull_requests("owner", "repo", since, until)

summary = ReviewSummary(
    owner="owner", repo="repo",
    since=since, until=until,
    commits=commits, issues=issues, pull_requests=prs,
)

# --- All repos for an org ---
all_repos = gh.list_repos("myorg")
summary_all = ReviewSummary(owner="myorg", repo="*", since=since, until=until)
for repo_name in all_repos:
    summary_all.commits       += gh.get_commits("myorg", repo_name, since, until)
    summary_all.issues        += gh.get_issues("myorg", repo_name, since, until)
    summary_all.pull_requests += gh.get_pull_requests("myorg", repo_name, since, until)

llm = LLMClient(api_key="sk-xxx")
text = llm.summarise(summary_all)
print(text)
```

### SDK reference

#### `GitHubClient(token=None, base_url="https://api.github.com/")`

| Method | Returns | Description |
|---|---|---|
| `list_repos(owner)` | `list[str]` | All non-archived repo names for a user or org |
| `get_commits(owner, repo, since, until, author=None)` | `list[Commit]` | Commits in the time window |
| `get_issues(owner, repo, since, until, state="all")` | `list[Issue]` | Issues (PRs excluded) |
| `get_pull_requests(owner, repo, since, until, state="all")` | `list[PullRequest]` | Pull requests |

#### `LLMClient(api_key=None, model="gpt-4o-mini", base_url=None)`

| Method | Returns | Description |
|---|---|---|
| `summarise(summary: ReviewSummary)` | `str` | Markdown-formatted AI summary |

#### `CommitMessageGenerator(api_key=None, model="gpt-4o-mini", base_url=None)`

| Method | Returns | Description |
|---|---|---|
| `generate(diff: str)` | `str` | Conventional Commit message for the given diff |

---

## LangGraph Pipeline

> **Requires the `langgraph` extra:**
> ```bash
> pip install 'git-review[langgraph]'
> ```

The `git_review.langgraph_pipeline` module implements the same summarisation
workflow as `LLMClient` as a [LangGraph](https://github.com/langchain-ai/langgraph)
`StateGraph`.  Using a graph gives you **explicit state management**,
**checkpointing / persistence**, and **conditional branching** – all without
changing the underlying GitHub or LLM code.

### Graph nodes

| Node | Purpose |
|---|---|
| `validate_data` | Checks the `ReviewSummary` for common problems (empty activity, invalid date window) and records any warnings in `validation_errors`. |
| `summarize` | Calls `LLMClient.summarise()` and sets `needs_refinement=True` when the result is shorter than `MIN_SUMMARY_CHARS` (200). |
| `refine` | *(Conditional)* Re-invokes the LLM with an extended prompt that explicitly asks for a longer answer.  Only runs when `needs_refinement` is `True`. |

### Using the pipeline

```python
from datetime import datetime, timedelta, timezone

from git_review.github_client import GitHubClient
from git_review.langgraph_pipeline import build_review_graph
from git_review.models import ReviewSummary
from langgraph.checkpoint.memory import MemorySaver

since = datetime.now(tz=timezone.utc) - timedelta(days=7)
until = datetime.now(tz=timezone.utc)

gh = GitHubClient(token="ghp_xxx")
summary = ReviewSummary(owner="myorg", repo="myrepo", since=since, until=until)
summary.commits = gh.get_commits("myorg", "myrepo", since, until)
summary.issues  = gh.get_issues("myorg",  "myrepo", since, until)

# Build and run the graph (with in-memory checkpointing)
graph = build_review_graph(
    openai_api_key="sk-xxx",
    checkpointer=MemorySaver(),
)
result = graph.invoke(
    {"summary": summary},
    config={"configurable": {"thread_id": "sprint-42"}},
)

print(result["summary_text"])
print("Validation warnings:", result["validation_errors"])
```

### Middleware

The `git_review.langgraph_middleware` module provides composable **middleware**
wrappers that add cross-cutting behaviour to any node function without
modifying its business logic.

#### Available middleware

| Class | What it does |
|---|---|
| `LoggingMiddleware` | Logs node entry, exit, and elapsed time. |
| `TokenCountingMiddleware` | Estimates token usage from `summary_text` and accumulates totals in a `TokenUsageCounter`. |
| `RetryMiddleware` | Wraps a node with exponential-backoff retry logic for transient failures. |

#### Example: wrapping graph nodes with middleware

```python
from langgraph.checkpoint.memory import MemorySaver

from git_review.langgraph_middleware import (
    apply_middleware,
    LoggingMiddleware,
    RetryMiddleware,
    TokenCountingMiddleware,
    TokenUsageCounter,
)
from git_review.langgraph_pipeline import (
    make_summarize_node,
    make_validate_node,
    make_refine_node,
    _route_after_summarize,
    _DEFAULT_SYSTEM_PROMPT,
    ReviewState,
)
from git_review.llm_client import LLMClient
from langgraph.graph import END, StateGraph

# Shared token counter – inspect after the run
counter = TokenUsageCounter()

llm_client = LLMClient(api_key="sk-xxx")

wrapped_validate = apply_middleware(
    make_validate_node(),
    LoggingMiddleware(),
)
wrapped_summarize = apply_middleware(
    make_summarize_node(llm_client),
    LoggingMiddleware(),
    TokenCountingMiddleware(counter=counter),
    RetryMiddleware(max_attempts=3),
)
wrapped_refine = apply_middleware(
    make_refine_node(llm_client, _DEFAULT_SYSTEM_PROMPT),
    LoggingMiddleware(),
    RetryMiddleware(max_attempts=2),
)

builder = StateGraph(ReviewState)
builder.add_node("validate_data", wrapped_validate)
builder.add_node("summarize",     wrapped_summarize)
builder.add_node("refine",        wrapped_refine)
builder.set_entry_point("validate_data")
builder.add_edge("validate_data", "summarize")
builder.add_conditional_edges(
    "summarize",
    _route_after_summarize,
    {"refine": "refine", END: END},
)
builder.add_edge("refine", END)

graph = builder.compile(checkpointer=MemorySaver())
result = graph.invoke(
    {"summary": summary},
    config={"configurable": {"thread_id": "run-1"}},
)

print(f"Summary: {result['summary_text'][:200]}…")
print(f"Approx tokens used: {counter.total_tokens}")
```

---



```
git_review/
├── __init__.py                  # Public SDK exports
├── models.py                    # Dataclasses: Commit, Issue, PullRequest, ReviewSummary
├── github_client.py             # GitHub REST API wrapper
├── llm_client.py                # OpenAI-compatible LLM summarisation
├── commit_message_generator.py  # Commit message generation from git diffs
├── issue_factory.py             # LLM-powered GitHub issue creation
├── prompt_utils.py              # Jinja2 prompt template loader
├── langgraph_pipeline.py        # LangGraph StateGraph review pipeline
├── langgraph_middleware.py      # Logging, token-counting, and retry middleware
└── cli.py                       # Click CLI entry-point
tests/
├── test_github_client.py
├── test_llm_client.py
├── test_commit_message_generator.py
├── test_langgraph_pipeline.py
├── test_langgraph_middleware.py
└── test_cli.py
vscode-extension/
├── src/extension.ts             # VS Code extension source
└── package.json                 # Extension manifest
```

---

## Running Tests

```bash
uv run pytest tests/ -v
```

---

## License

This project is licensed under the [MIT License](LICENSE).

