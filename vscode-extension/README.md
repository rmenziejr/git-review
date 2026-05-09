# git-review VS Code Extension

Integrates the [git-review](https://github.com/rmenziejr/git-review) CLI into Visual Studio Code, providing commands and a settings UI for all git-review options.

---

## Features

| Command (Command Palette) | Description |
|---|---|
| **Git Review: Review Repository** | Fetch GitHub activity (commits, issues, PRs, releases) and generate an AI summary |
| **Git Review: Generate Commit Message** | Write a Conventional Commit message for your staged (or unstaged) diff |
| **Git Review: Create Issues from Requirements** | Parse a markdown requirements file and push GitHub issues via LLM |
| **Git Review: Create Milestone** | Create a GitHub milestone with an optional due date and description |
| **Git Review: Agile Plan** | Fetch open issues/PRs and generate an AI-powered sprint plan |

All commands run in a dedicated **git-review** terminal panel, giving you full interactive output and the ability to scroll back through results.

---

## Requirements

`git-review` must be installed and available on your `PATH`.  Install it with:

```bash
pip install git-review
# or, if you use uv:
uv tool install git-review
```

---

## Extension Settings

Open **File → Preferences → Settings** (or press `Ctrl+,`) and search for **"git review"** to configure the extension.

| Setting | Description | CLI / Env equivalent |
|---|---|---|
| `gitReview.githubToken` | GitHub personal access token | `--token` / `GITHUB_TOKEN` |
| `gitReview.openaiApiKey` | OpenAI API key | `--openai-key` / `OPENAI_API_KEY` |
| `gitReview.openaiBaseUrl` | Custom OpenAI-compatible base URL (Ollama, Azure, Groq…) | `--base-url` / `OPENAI_BASE_URL` |
| `gitReview.defaultRepo` | Default repository in `owner/repo` format | `--repo` / `GITREVIEW_REPO` |
| `gitReview.defaultOwner` | Default owner/org for owner-only runs (used when `defaultRepo` is empty) | `--owner` |
| `gitReview.reviewAuthor` | Optional commit author filter for review runs | `--author` |
| `gitReview.model` | LLM model (default: `gpt-4o-mini`) | `--model` / `GIT_REVIEW_MODEL` |
| `gitReview.defaultDays` | Days to look back for reviews (default: `7`) | `--days` |
| `gitReview.defaultBranch` | Branch to fetch commits from for review (blank = repo default, `*` = all branches) | `--branch` |
| `gitReview.enableThinking` | Enable extended reasoning (thinking mode) for commit message generation | `--think` |
| `gitReview.sprintCapacity` | Maximum issues per sprint for Agile Plan (default: `10`) | `--sprint-capacity` |
| `gitReview.sprints` | Number of future sprints to plan for Agile Plan (default: `3`) | `--sprints` |

Settings are read every time a command is invoked, so changes take effect immediately without reloading the window.

---

## Usage

### Review a repository

1. Open the Command Palette (`Ctrl+Shift+P` / `Cmd+Shift+P`)
2. Run **Git Review: Review Repository**
3. Enter `owner/repo` when prompted (or leave blank to review by owner/org)
4. Watch the terminal for the rich-formatted table output and AI summary

If `gitReview.defaultRepo` is set in your settings the prompt is skipped.  
Set `gitReview.defaultBranch` to restrict commits to a specific branch.

### Generate a commit message

1. Stage your changes (`git add …`)
2. Run **Git Review: Generate Commit Message** from the Command Palette
3. The Conventional Commit message is printed in the terminal — copy it into the Source Control panel commit box

Enable `gitReview.enableThinking` to use extended reasoning for more thoughtful commit messages.

### Create issues from requirements

1. Run **Git Review: Create Issues from Requirements** from the Command Palette
2. Enter the target repository when prompted (or skip if `gitReview.defaultRepo` is set)
3. Pick your markdown requirements file in the file-picker dialog
4. Review the draft issues in the terminal and answer the interactive prompts

### Create a milestone

1. Run **Git Review: Create Milestone** from the Command Palette
2. Enter the target repository when prompted (or skip if `gitReview.defaultRepo` is set)
3. Provide a milestone title, optional description, and optional due date (YYYY-MM-DD)

### Run an agile sprint plan

1. Run **Git Review: Agile Plan** from the Command Palette
2. Enter the repository or owner/org when prompted (or skip if defaults are set)
3. The terminal shows open issues/PRs, inferred dependencies, and an AI-generated sprint plan

Configure `gitReview.sprintCapacity` and `gitReview.sprints` to tailor the plan.

---

## Known Issues / Tips

* Sensitive values (tokens, API keys) are stored in plain text in VS Code's settings file. Consider using the **Secret Storage** or keeping them in your `.env` file and leaving the extension fields empty.
* The extension does not bundle the Python package — you must have `git-review` on your `PATH` yourself.
