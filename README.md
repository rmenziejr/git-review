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
- [Web App (Gradio)](#web-app-gradio)
- [Agent & Reflex UI](#agent--reflex-ui)
- [VS Code Extension](#vs-code-extension)
- [Python SDK](#python-sdk)
- [Project Layout](#project-layout)
- [Running Tests](#running-tests)
- [License](#license)

---

## Features

| Feature | Details |
|---|---|
| **GitHub data extraction** | Commits, issues, pull requests, releases, and contributors |
| **Flexible time windows** | Explicit `--since`/`--until` dates *or* a `--days` delta |
| **Author filter** | Narrow commits to a specific GitHub user |
| **AI summarisation** | OpenAI (default) or any OpenAI-compatible endpoint |
| **Markdown output** | Save the AI summary to a `.md` file with `--output` |
| **Commit message generator** | Write a Conventional Commit message from your staged diff |
| **Issue generator** | Parse a markdown requirements file and create GitHub issues via LLM |
| **Milestone management** | Create and list GitHub milestones from the CLI or web app |
| **Agile planner** | Fetch all open issues/PRs, detect blocking dependencies, and generate a prioritised sprint plan via LLM |
| **Native dependency API** | Write inferred blocking/blocked-by relationships back to GitHub using the native issue-dependencies REST API |
| **Custom prompts** | Override the LLM system prompt with a Jinja2 template via `--prompt-file` |
| **Thinking mode** | Extended reasoning for commit-message generation with `--think` |
| **Rich terminal output** | Colour-coded tables and a formatted summary panel |
| **Gradio web app** | Browser-based UI with tabs for activity summary, milestones, requirements parsing, issue submission, and agile planning |
| **Conversational agent** | OpenAI Agents SDK–powered assistant that can list, create, and update issues and PRs with streaming thoughts, tool-call UX, and human-in-the-loop approval for write operations |
| **Reflex frontend** | React-based chat UI (served by Reflex) with live streaming, collapsible tool-call cards, reasoning/thinking indicators, and an HITL approval panel |
| **Python SDK** | Use `GitHubClient`, `LLMClient`, and `AgilePlanner` directly in your own code |
| **VS Code extension** | Run all commands from the Command Palette inside VS Code |

---

## Installation

### With pip

```bash
pip install git-review
```

### With pip (including the Gradio web app)

```bash
pip install 'git-review[gradio]'
```

### With uv (add to a project)

Add directly from GitHub into your `uv`-managed project:

```bash
uv add git+https://github.com/rmenziejr/git-review.git
```

Including the Gradio web app extra:

```bash
uv add "git-review[gradio] @ git+https://github.com/rmenziejr/git-review.git"
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
| `GIT_REVIEW_MODEL` | Default LLM model (default: `gpt-4o-mini`, avoids `--model`) |
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

# Use a custom system prompt from a Jinja2 template file
git-review review --repo owner/repo --days 7 \
  --openai-key sk-xxx --prompt-file my_prompt.j2
```

### Generate a commit message

Stage your changes, then run:

```bash
# Generate a Conventional Commit message from your staged diff
git-review commit-message

# Use a different model or a local Ollama instance
git-review commit-message --model gpt-4o
git-review commit-message --base-url http://localhost:11434/v1 --model llama3

# Enable extended reasoning (thinking mode)
git-review commit-message --think

# Use a custom system prompt from a Jinja2 template file
git-review commit-message --prompt-file my_prompt.j2

# Point at a different git repository
git-review commit-message --repo-path /path/to/other/repo
```

The command reads the staged diff (`git diff --staged`) – or the unstaged
diff if nothing is staged – and asks the LLM to write a
[Conventional Commits](https://www.conventionalcommits.org/) message.
After generating the message you will be prompted to optionally edit it and
then commit directly.

### Create a milestone

```bash
# Create a milestone with a title and optional due date
git-review create-milestone --repo owner/repo --title "v1.0 Release" \
  --description "All tasks for v1.0" --due-on 2025-12-31 \
  --token ghp_xxx
```

### Create GitHub issues from requirements

```bash
# Parse a local markdown requirements file and push issues to GitHub
git-review create-issues --repo owner/repo --requirements requirements.md \
  --token ghp_xxx --openai-key sk-xxx

# Fetch the requirements file directly from the repository
git-review create-issues --repo owner/repo \
  --requirements-path docs/requirements.md \
  --token ghp_xxx --openai-key sk-xxx

# Skip interactive approval and push all issues immediately
git-review create-issues --repo owner/repo --requirements requirements.md \
  --token ghp_xxx --openai-key sk-xxx --yes

# Dry-run: parse and display drafts without creating any issues
git-review create-issues --repo owner/repo --requirements requirements.md \
  --token ghp_xxx --openai-key sk-xxx --dry-run

# Attach every created issue to a specific milestone number
git-review create-issues --repo owner/repo --requirements requirements.md \
  --token ghp_xxx --openai-key sk-xxx --milestone 3

# Let the LLM assign each issue to the most relevant open milestone
git-review create-issues --repo owner/repo --requirements requirements.md \
  --token ghp_xxx --openai-key sk-xxx --use-milestones

# Use a custom system prompt from a Jinja2 template file
git-review create-issues --repo owner/repo --requirements requirements.md \
  --token ghp_xxx --openai-key sk-xxx --prompt-file my_prompt.j2
```

### Agile sprint planning

```bash
# Generate a sprint plan for a single repo (table + LLM analysis)
git-review agile --repo owner/repo --token ghp_xxx --openai-key sk-xxx

# Org-level plan — plan across ALL repos for an owner (three equivalent forms):
git-review agile --repo owner/*   --token ghp_xxx --openai-key sk-xxx
git-review agile --repo owner     --token ghp_xxx --openai-key sk-xxx
git-review agile --owner owner    --token ghp_xxx --openai-key sk-xxx

# Show only open issues/PRs without running the LLM
git-review agile --repo owner/repo --token ghp_xxx --no-summary

# Customise sprint size and number of sprints
git-review agile --repo owner/repo --sprint-capacity 8 --sprints 4 \
  --token ghp_xxx --openai-key sk-xxx

# Write the plan to a markdown file
git-review agile --repo owner/repo \
  --token ghp_xxx --openai-key sk-xxx --output plan.md

# Preview what new blocking relationships would be recorded (dry-run)
git-review agile --repo owner/repo \
  --token ghp_xxx --openai-key sk-xxx \
  --apply-relationships --dry-run

# Record new inferred blocking relationships in GitHub's dependency API
git-review agile --repo owner/repo \
  --token ghp_xxx --openai-key sk-xxx --apply-relationships

# Apply priority label recommendations to issues
git-review agile --repo owner/repo \
  --token ghp_xxx --openai-key sk-xxx --apply-labels
```

The `agile` command reads **existing** GitHub blocking/blocked-by relationships
(via `GET /repos/{owner}/{repo}/issues/{number}/dependencies/blocked_by`),
merges them with text-extracted and LLM-inferred relationships, and presents
the full dependency graph alongside a sprint-by-sprint backlog.

When `--apply-relationships` is used (without `--dry-run`) the tool prompts
for confirmation, then calls
`POST /repos/{owner}/{repo}/issues/{number}/dependencies/blocked_by` to
register each new LLM-inferred dependency directly in GitHub — no labels
involved.

---

## Web App (Gradio)

git-review ships an optional browser-based UI powered by
[Gradio](https://www.gradio.app/). It provides the same features as the CLI
through an easy-to-use interface.

### Installing the web app

```bash
pip install 'git-review[gradio]'
```

### Launching the web app

```bash
git-review-app
# or
python -m git_review.app
```

By default the app listens on `http://0.0.0.0:7860`. Open that URL in your
browser to access the UI.

### Web app tabs

| Tab | Description |
|---|---|
| **📊 Summarize Activity** | Fetch GitHub activity (commits, issues, PRs, releases) for a repo and generate an AI summary |
| **🏁 Milestones** | Create a new milestone or list existing milestones in a repository |
| **📄 Parse Requirements** | Upload or fetch a markdown requirements file and generate issue drafts via LLM |
| **🚀 Submit Issues** | Review, edit, and selectively push the generated issue drafts to GitHub |
| **🗂️ Agile Planner** | Fetch open issues/PRs, show the dependency graph, generate a sprint plan, and optionally write blocking relationships back to GitHub |

### Web app configuration

The app reads from the same `.env` file (or environment variables) as the CLI,
with two additional variables for the server:

| Variable | Description | Default |
|---|---|---|
| `GITHUB_TOKEN` | GitHub personal access token | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `GIT_REVIEW_MODEL` | Default LLM model | `gpt-4o-mini` |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible base URL | — |
| `GRADIO_SERVER_NAME` | Hostname or IP address to bind to | `0.0.0.0` |
| `GRADIO_SERVER_PORT` | TCP port | `7860` |

---

## Agent & Reflex UI

git-review ships a conversational AI agent built on the
[OpenAI Agents SDK](https://github.com/openai/openai-agents-python) with a
[Reflex](https://reflex.dev/) React frontend.  The agent can list, search,
create, and update GitHub issues and pull requests, generate sprint plans, and
parse requirements into issue drafts — all through a streaming chat interface.

### Installing the agent

```bash
pip install 'git-review[agent]'
```

### Launching the agent UI

```bash
git-review-agent
```

Reflex will compile the frontend (Node.js is required on first run) and open
the app at `http://localhost:3000`.

### Configuring the agent

Open the ⚙ Settings panel in the UI (top-right corner) or set the following
environment variables / `.env` entries:

| Variable | Description | Default |
|---|---|---|
| `GITHUB_TOKEN` | GitHub personal access token | — |
| `OPENAI_API_KEY` | OpenAI API key | — |
| `OPENAI_BASE_URL` | Custom OpenAI-compatible base URL (Ollama, Azure, etc.) | — |
| `AGENT_MODEL` | LLM model used by the agent | `gpt-4o` |

### Streaming UX

| Element | Description |
|---|---|
| **Chat thread** | Markdown-rendered messages scroll in real time as the model generates them |
| **Thinking indicator** | Animated dots show while the agent is processing |
| **Reasoning cards** | Collapsible accordion cards display the model's chain-of-thought (when using a reasoning model) |
| **Tool-call cards** | Each tool invocation appears inline with its arguments and result |
| **HITL approval panel** | Any write operation (create issue, create PR, update issue/PR, mark PR ready) pauses and shows an **Approve / Deny** panel before executing |

### Human-in-the-loop (HITL) flow

1. The agent decides to call a write tool (e.g. `push_issue_draft`).
2. The run **pauses automatically** — no GitHub mutation has happened yet.
3. A prominent approval card appears above the chat input showing the tool
   name and its arguments.
4. Click **Approve** to execute the action and continue, or **Deny** to
   cancel and let the agent know.
5. All read operations (list issues, list PRs, agile plan, etc.) execute
   immediately without any approval step.

### Using a local model (Ollama)

```bash
# Start Ollama with a compatible model
ollama run qwen2.5:7b

# Point the agent at Ollama
OPENAI_BASE_URL=http://localhost:11434/v1 AGENT_MODEL=qwen2.5:7b git-review-agent
```

### Python SDK usage

```python
import asyncio
from git_review.agent import AgentContext, run_agent_streaming
from agents.stream_events import RawResponsesStreamEvent

ctx = AgentContext(
    owner="myorg",
    repo="myrepo",
    github_token="ghp_...",
    openai_api_key="sk-...",
    model="gpt-4o",
)

async def main():
    result = run_agent_streaming(ctx, "List open issues")
    async for event in result.stream_events():
        if isinstance(event, RawResponsesStreamEvent):
            if getattr(event.data, "type", "") == "response.output_text.delta":
                print(event.data.delta, end="", flush=True)
    print()

    # Handle HITL interruptions
    if result.interruptions:
        state = result.to_state()
        state.approve(result.interruptions[0])   # or state.reject(...)
        result2 = run_agent_streaming(ctx, state)
        async for event in result2.stream_events():
            ...

asyncio.run(main())
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
| `gitReview.defaultOwner` | Default owner/user for owner-only review runs (used when `defaultRepo` is empty) | `--owner` |
| `gitReview.reviewAuthor` | Optional commit author filter for review runs | `--author` |
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

# Render the same Rich tables used by the CLI:
from git_review import render_review_tables
render_review_tables(summary_all)
```

### SDK reference

#### `GitHubClient(token=None, base_url="https://api.github.com/")`

| Method | Returns | Description |
|---|---|---|
| `list_repos(owner)` | `list[str]` | All non-archived repo names for a user or org |
| `get_commits(owner, repo, since, until, author=None)` | `list[Commit]` | Commits in the time window |
| `get_issues(owner, repo, since, until, state="all")` | `list[Issue]` | Issues (PRs excluded) |
| `get_pull_requests(owner, repo, since, until, state="all")` | `list[PullRequest]` | Pull requests |
| `get_open_issues(owner, repo)` | `list[Issue]` | All currently open issues (no time filter) |
| `get_open_pull_requests(owner, repo)` | `list[PullRequest]` | All currently open PRs (no time filter) |
| `get_issue(owner, repo, issue_number)` | `Issue` | Fetch a single issue by number |
| `update_issue(owner, repo, number, *, title, body, state, labels, assignees, milestone)` | `dict` | Update an issue via PATCH |
| `create_pull_request(owner, repo, title, body, head, base, draft)` | `dict` | Create a pull request |
| `update_pull_request(owner, repo, number, *, title, body, state, draft, base)` | `dict` | Update a PR via PATCH |
| `get_issue_blocked_by(owner, repo, issue_number)` | `list[dict]` | Issues that block the given issue (GitHub dependency API) |
| `get_issue_blocking(owner, repo, issue_number)` | `list[dict]` | Issues the given issue is blocking (GitHub dependency API) |
| `add_issue_blocked_by(owner, repo, issue_number, blocking_issue_id)` | `dict` | Record a blocked-by relationship via GitHub's native API |
| `remove_issue_blocked_by(owner, repo, issue_number, blocking_issue_id)` | `dict` | Remove a blocked-by relationship |
| `update_issue_labels(owner, repo, issue_number, labels)` | `dict` | Replace all labels on an issue |

#### `LLMClient(api_key=None, model="gpt-4o-mini", base_url=None)`

| Method | Returns | Description |
|---|---|---|
| `summarise(summary: ReviewSummary)` | `str` | Markdown-formatted AI summary |

#### `CommitMessageGenerator(api_key=None, model="gpt-4o-mini", base_url=None)`

| Method | Returns | Description |
|---|---|---|
| `generate(diff: str)` | `str` | Conventional Commit message for the given diff |

#### `AgilePlanner(github_client, openai_api_key=None, model="gpt-4o-mini", base_url=None, sprint_capacity=10, num_sprints=3)`

| Method | Returns | Description |
|---|---|---|
| `analyse(owner, repo)` | `AgilePlanResult` | Fetch open issues/PRs, read GitHub dependencies, infer new ones with LLM, generate sprint plan |
| `analyse_org(owner)` | `AgilePlanResult` | Same as `analyse` but aggregates across all repos for the owner |
| `apply_relationships(owner, repo, result, dry_run=True)` | `list[dict]` | Write new LLM-inferred blocked-by relationships to GitHub's dependency API |
| `apply_labels(owner, repo, result, dry_run=True)` | `list[dict]` | Apply priority label recommendations to issues |

```python
from git_review import GitHubClient, AgilePlanner

gh = GitHubClient(token="ghp_xxx")
planner = AgilePlanner(
    github_client=gh,
    openai_api_key="sk-xxx",
    sprint_capacity=8,
    num_sprints=3,
)

result = planner.analyse("myorg", "myrepo")
print(result.summary_text)

for sprint in result.sprints:
    print(f"Sprint {sprint.sprint_number} — {sprint.theme}")
    for issue_number in sprint.issues:
        print(f"  #{issue_number}")

# Preview what would be written to GitHub
planner.apply_relationships("myorg", "myrepo", result, dry_run=True)

# Actually record new inferred blocking relationships
planner.apply_relationships("myorg", "myrepo", result, dry_run=False)
```

---

## Project Layout

```
git_review/
├── __init__.py                  # Public SDK exports
├── models.py                    # Dataclasses: Commit, Issue, PullRequest, ReviewSummary, IssueDependency, SprintRecommendation, AgilePlanResult
├── github_client.py             # GitHub REST API wrapper (including native dependency endpoints)
├── llm_client.py                # OpenAI-compatible LLM summarisation
├── agile_planner.py             # Agile sprint planner (dependency graph + sprint plan via LLM)
├── commit_message_generator.py  # Commit message generation from git diffs
├── issue_factory.py             # LLM-powered GitHub issue creation
├── prompt_utils.py              # Jinja2 prompt template loader
├── config.py                    # AppSettings (pydantic-settings): env vars and .env support
├── app.py                       # Gradio web application (optional, requires [gradio] extra)
├── cli.py                       # Click CLI entry-point
├── agent_tools.py               # OpenAI Agents SDK tool functions (requires [agent] extra)
├── agent.py                     # Agent builder + streaming run helper
└── agent_app/                   # Reflex chat frontend (requires [agent] extra)
    ├── agent_app.py             # Reflex app entry point + main() for git-review-agent script
    ├── rxconfig.py              # Reflex configuration
    ├── state.py                 # Reactive app state (chat history, HITL queue, settings)
    └── components/
        ├── chat.py              # Chat thread with streaming text + thinking indicator
        ├── tool_call.py         # Collapsible tool-call / tool-result cards
        ├── hitl_panel.py        # HITL approve / deny panel
        └── settings.py          # Settings sidebar (repo, token, model, base_url)
tests/
├── test_agile_planner.py
├── test_agent_tools.py          # Unit tests for agent tool functions and HITL approval flags
├── test_github_client.py
├── test_llm_client.py
├── test_commit_message_generator.py
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
