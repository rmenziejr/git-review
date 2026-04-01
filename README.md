# git-review

A Python CLI tool and SDK that connects to the GitHub API to extract commits,
issues, and pull requests submitted within a configurable time window, then
generates a concise summary via an LLM (OpenAI or any compatible API).

---

## Features

| Feature | Details |
|---|---|
| **GitHub data extraction** | Commits, issues, and pull requests |
| **Flexible time windows** | Explicit `--since`/`--until` dates *or* a `--days` delta |
| **Author filter** | Narrow commits to a specific GitHub user |
| **AI summarisation** | OpenAI (default) or any OpenAI-compatible endpoint |
| **Rich terminal output** | Colour-coded tables and a formatted summary panel |
| **Python SDK** | Use `GitHubClient` and `LLMClient` directly in your own code |

---

## Installation

### From source

```bash
git clone https://github.com/rmenziejr/git-review.git
cd git-review
pip install .
```

### Development (with test dependencies)

```bash
pip install -e ".[dev]"
```

---

## Quick start

```bash
# Last 7 days (default) – table output only (no LLM key required)
git-review review --repo owner/repo --no-summary

# Use a GitHub token to avoid rate limits
git-review review --repo owner/repo --token ghp_xxx --no-summary

# AI summary – last 14 days
git-review review --repo owner/repo --days 14 \
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

### Environment variables

| Variable | Description |
|---|---|
| `GITHUB_TOKEN` | GitHub PAT (avoids `--token`) |
| `OPENAI_API_KEY` | OpenAI API key (avoids `--openai-key`) |
| `OPENAI_BASE_URL` | Custom API base URL (avoids `--base-url`) |
| `GITREVIEW_REPO` | Default repository (avoids `--repo`) |

---

## Python SDK

```python
from datetime import datetime, timedelta, timezone
from git_review import GitHubClient, LLMClient, ReviewSummary

since = datetime.now(tz=timezone.utc) - timedelta(days=14)
until = datetime.now(tz=timezone.utc)

gh = GitHubClient(token="ghp_xxx")
commits = gh.get_commits("owner", "repo", since, until)
issues  = gh.get_issues("owner", "repo", since, until)
prs     = gh.get_pull_requests("owner", "repo", since, until)

summary = ReviewSummary(
    owner="owner", repo="repo",
    since=since, until=until,
    commits=commits, issues=issues, pull_requests=prs,
)

llm = LLMClient(api_key="sk-xxx")
text = llm.summarise(summary)
print(text)
```

### SDK reference

#### `GitHubClient(token=None, base_url="https://api.github.com/")`

| Method | Returns | Description |
|---|---|---|
| `get_commits(owner, repo, since, until, author=None)` | `list[Commit]` | Commits in the time window |
| `get_issues(owner, repo, since, until, state="all")` | `list[Issue]` | Issues (PRs excluded) |
| `get_pull_requests(owner, repo, since, until, state="all")` | `list[PullRequest]` | Pull requests |

#### `LLMClient(api_key=None, model="gpt-4o-mini", base_url=None)`

| Method | Returns | Description |
|---|---|---|
| `summarise(summary: ReviewSummary)` | `str` | Markdown-formatted AI summary |

---

## Running tests

```bash
pytest tests/ -v
```

---

## Project layout

```
git_review/
├── __init__.py        # Public SDK exports
├── models.py          # Dataclasses: Commit, Issue, PullRequest, ReviewSummary
├── github_client.py   # GitHub REST API wrapper
├── llm_client.py      # OpenAI-compatible LLM summarisation
└── cli.py             # Click CLI entry-point
tests/
├── test_github_client.py
├── test_llm_client.py
└── test_cli.py
```

