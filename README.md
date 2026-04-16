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
| **Custom prompts** | Override the LLM system prompt with a Jinja2 template via `--prompt-file` |
| **Thinking mode** | Extended reasoning for commit-message generation with `--think` |
| **Rich terminal output** | Colour-coded tables and a formatted summary panel |
| **Gradio web app** | Browser-based UI with tabs for activity summary, milestones, requirements parsing, and issue submission |
| **Python SDK** | Use `GitHubClient` and `LLMClient` directly in your own code |
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

#### `LLMClient(api_key=None, model="gpt-4o-mini", base_url=None)`

| Method | Returns | Description |
|---|---|---|
| `summarise(summary: ReviewSummary)` | `str` | Markdown-formatted AI summary |

#### `CommitMessageGenerator(api_key=None, model="gpt-4o-mini", base_url=None)`

| Method | Returns | Description |
|---|---|---|
| `generate(diff: str)` | `str` | Conventional Commit message for the given diff |

---

## Project Layout

```
git_review/
├── __init__.py                  # Public SDK exports
├── models.py                    # Dataclasses: Commit, Issue, PullRequest, ReviewSummary
├── github_client.py             # GitHub REST API wrapper
├── llm_client.py                # OpenAI-compatible LLM summarisation
├── commit_message_generator.py  # Commit message generation from git diffs
├── issue_factory.py             # LLM-powered GitHub issue creation
├── prompt_utils.py              # Jinja2 prompt template loader
├── config.py                    # AppSettings (pydantic-settings): env vars and .env support
├── app.py                       # Gradio web application (optional, requires [gradio] extra)
└── cli.py                       # Click CLI entry-point
tests/
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
