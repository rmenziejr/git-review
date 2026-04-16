"""Gradio web application for git-review issue management.

Provides a browser-based UI to:
- Summarize GitHub repository activity with an AI-generated report.
- Create GitHub milestones.
- Upload a markdown requirements document and generate issue drafts via LLM.
- Review, edit, and selectively submit those drafts as real GitHub issues.

Launch
------
::

    python -m git_review.app
    # or
    gradio git_review/app.py

Settings can be configured via environment variables or a ``.env`` file.
See :class:`~git_review.config.AppSettings` for all available settings.

Environment variables
---------------------
GITHUB_TOKEN
    GitHub personal access token with repo write access.
OPENAI_API_KEY
    OpenAI API key used for requirements parsing and summarisation.
GIT_REVIEW_MODEL
    LLM model (default: gpt-4o-mini).
OPENAI_BASE_URL
    Custom OpenAI-compatible base URL (optional).
GRADIO_SERVER_NAME
    Hostname or IP address for the Gradio server (default: 0.0.0.0).
GRADIO_SERVER_PORT
    TCP port for the Gradio server (default: 7860).
"""

from __future__ import annotations

import logging
import tempfile
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

logger = logging.getLogger(__name__)

try:
    import gradio as gr
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'gradio' package is required to run the web app. "
        "Install it with:  pip install 'git-review[gradio]'"
    ) from _exc

if not hasattr(gr, "Blocks"):  # pragma: no cover
    _gradio_version = getattr(gr, "__version__", "unknown")
    raise ImportError(
        f"gradio>=4.0 is required but {_gradio_version!r} is installed. "
        "Upgrade with:  pip install 'git-review[gradio]'"
    )

from .config import AppSettings
from .github_client import GitHubClient
from .issue_factory import IssueDraft, IssueFactory
from .llm_client import LLMClient
from .models import ReviewSummary

# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _make_clients(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
) -> tuple[GitHubClient, IssueFactory]:
    gh = GitHubClient(token=github_token or None)
    factory = IssueFactory(
        github_client=gh,
        openai_api_key=openai_key or None,
        model=model or "gpt-4o-mini",
        base_url=base_url or None,
    )
    return gh, factory


def _drafts_to_table(drafts: list[IssueDraft]) -> list[list[Any]]:
    """Convert drafts to a list-of-rows suitable for gr.Dataframe."""
    return [
        [
            i + 1,
            d.title,
            d.body,
            ", ".join(d.labels),
            ", ".join(d.assignees),
            str(d.milestone) if d.milestone is not None else "",
        ]
        for i, d in enumerate(drafts)
    ]


def _table_to_drafts(rows: list[list[Any]]) -> list[IssueDraft]:
    """Reconstruct IssueDraft objects from edited table rows."""
    drafts: list[IssueDraft] = []
    for row in rows:
        # row: [#, title, body, labels, assignees, milestone]
        title = str(row[1]).strip()
        body = str(row[2]).strip()
        if not title:
            continue
        labels_raw = str(row[3]) if len(row) > 3 else ""
        assignees_raw = str(row[4]) if len(row) > 4 else ""
        milestone_raw = str(row[5]).strip() if len(row) > 5 else ""
        labels = [lbl.strip() for lbl in labels_raw.split(",") if lbl.strip()]
        assignees = [a.strip() for a in assignees_raw.split(",") if a.strip()]
        milestone: Optional[int] = None
        if milestone_raw:
            try:
                milestone = int(milestone_raw)
            except ValueError:
                pass
        drafts.append(
            IssueDraft(
                title=title,
                body=body,
                labels=labels,
                assignees=assignees,
                milestone=milestone,
            )
        )
    return drafts


# ---------------------------------------------------------------------------
# Tab: Summarize Activity
# ---------------------------------------------------------------------------

def _summarize_activity(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
    repo: str,
    days: int,
    since_str: str,
    until_str: str,
    author: str,
    system_prompt: str,
    all_repos: bool = False,
) -> tuple[str, str]:
    """Fetch GitHub activity and return (summary_markdown, status_text)."""
    repo = (repo or "").strip()
    if not repo:
        label = "owner name" if all_repos else "'owner/repo' format"
        return "", f"❌  Please enter the repository in {label}."

    if all_repos:
        # Accept either "owner" or "owner/repo" – extract just the owner part.
        owner = repo.split("/", 1)[0].strip()
        repo_label = "*"
    else:
        if "/" not in repo:
            return "", "❌  Please enter the repository in 'owner/repo' format."
        parts = repo.split("/", 1)
        owner, repo_label = parts[0], parts[1]

    effective_key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not (base_url or "").strip():
        return "", (
            "❌  An OpenAI API key is required. "
            "Enter it above or set OPENAI_API_KEY."
        )

    # Resolve date window
    now_utc = datetime.now(tz=timezone.utc).replace(
        hour=23, minute=59, second=59, microsecond=0
    )
    try:
        since = (
            datetime.strptime(since_str.strip(), "%Y-%m-%d").replace(tzinfo=timezone.utc)
            if since_str.strip()
            else (now_utc - timedelta(days=max(1, int(days)))).replace(
                hour=0, minute=0, second=0, microsecond=0
            )
        )
        until = (
            datetime.strptime(until_str.strip(), "%Y-%m-%d").replace(
                hour=23, minute=59, second=59, tzinfo=timezone.utc
            )
            if until_str.strip()
            else now_utc
        )
    except ValueError as exc:
        return "", f"❌  Invalid date: {exc}"

    if since > until:
        return "", "❌  'Since' date must be earlier than 'Until' date."

    gh = GitHubClient(token=github_token or None)
    review_data = ReviewSummary(owner=owner, repo=repo_label, since=since, until=until)
    errors: list[str] = []
    author_filter = author.strip() or None
    section_to_attr = {
        "commits": "commits",
        "issues": "issues",
        "pull_requests": "pull_requests",
        "releases": "releases",
        "contributors": "contributors",
    }

    if all_repos:
        try:
            repo_names = gh.list_repos(owner)
        except Exception as exc:
            return "", f"❌  Could not list repositories for '{owner}': {exc}"
        if not repo_names:
            return "", f"⚠️  No non-archived repositories found for '{owner}'."
    else:
        repo_names = [repo_label]

    for repo_name in repo_names:
        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = {
                executor.submit(
                    gh.get_commits,
                    owner, repo_name, since, until,
                    author=author_filter,
                    include_stats=True,
                ): "commits",
                executor.submit(
                    gh.get_issues, owner, repo_name, since, until
                ): "issues",
                executor.submit(
                    gh.get_pull_requests, owner, repo_name, since, until, include_details=True
                ): "pull_requests",
                executor.submit(
                    gh.get_releases, owner, repo_name, since, until
                ): "releases",
                executor.submit(
                    gh.get_contributors, owner, repo_name
                ): "contributors",
            }

            for future in as_completed(futures):
                data_label = futures[future]
                try:
                    results = future.result()
                    attr_name = section_to_attr[data_label]
                    getattr(review_data, attr_name).extend(results)
                except Exception as exc:
                    display_label = data_label.replace("_", " ")
                    errors.append(
                        f"⚠️  Could not fetch {display_label} for {owner}/{repo_name}: {exc}"
                    )

    try:
        llm = LLMClient(
            api_key=effective_key or None,
            model=(model or "").strip() or "gpt-4o-mini",
            base_url=(base_url or "").strip() or None,
            system_prompt=system_prompt.strip() or None,
        )
        summary_text = llm.summarise(review_data)
    except Exception as exc:
        return "", f"❌  Error generating summary: {exc}"

    if all_repos:
        repo_display = f"{owner}/* ({len(repo_names)} repos)"
    else:
        repo_display = f"{owner}/{repo_label}"
    status = f"✅  Summary generated for {repo_display} ({since.date()} → {until.date()})."
    if errors:
        status += "\n" + "\n".join(errors)
    return summary_text, status


# ---------------------------------------------------------------------------
# Tab: Create Milestone
# ---------------------------------------------------------------------------

def _create_milestone(
    github_token: str,
    repo: str,
    title: str,
    description: str,
    due_on: str,
    state: str,
) -> str:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."
    if not title.strip():
        return "❌  Milestone title is required."

    parts = repo.strip().split("/", 1)
    owner, repo_name = parts[0], parts[1]

    due_on_iso: Optional[str] = None
    if due_on.strip():
        try:
            due_dt = datetime.strptime(due_on.strip(), "%Y-%m-%d").replace(
                hour=0, minute=0, second=0, tzinfo=timezone.utc
            )
            due_on_iso = due_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
        except ValueError:
            return f"❌  Invalid due date '{due_on}'. Use YYYY-MM-DD format."

    gh = GitHubClient(token=github_token or None)
    try:
        result = gh.create_milestone(
            owner=owner,
            repo=repo_name,
            title=title.strip(),
            description=description.strip(),
            due_on=due_on_iso,
            state=state,
        )
    except Exception as exc:
        return f"❌  Error creating milestone: {exc}"

    number = result.get("number", "?")
    url = result.get("html_url", "")
    return f"✅  Milestone #{number} '{title.strip()}' created successfully.\n{url}"


def _list_milestones(github_token: str, repo: str, state: str) -> str:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."

    parts = repo.strip().split("/", 1)
    owner, repo_name = parts[0], parts[1]

    gh = GitHubClient(token=github_token or None)
    try:
        milestones = gh.list_milestones(owner, repo_name, state=state)
    except Exception as exc:
        return f"❌  Error listing milestones: {exc}"

    if not milestones:
        return f"No milestones found (state={state})."

    lines = [f"Found {len(milestones)} milestone(s):\n"]
    for m in milestones:
        due = str(m.due_on.date()) if m.due_on else "—"
        lines.append(
            f"  #{m.number}  {m.title}  (state={m.state}, due={due}, "
            f"open={m.open_issues}, closed={m.closed_issues})"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tab: Parse Requirements
# ---------------------------------------------------------------------------

_DRAFT_COLUMNS = ["#", "Title", "Body", "Labels", "Assignees", "Milestone #"]

_EMPTY_TABLE: list[list[Any]] = []


def _fetch_requirements_from_repo(
    github_token: str,
    repo: str,
    path: str,
) -> tuple[str, str]:
    """Fetch a requirements file from a GitHub repo and return (content, status)."""
    if not repo or "/" not in repo:
        return "", "❌  Please enter the repository in 'owner/repo' format."
    path = (path or "docs/requirements.md").strip()
    if not path:
        return "", "❌  Please enter a file path."

    parts = repo.strip().split("/", 1)
    owner, repo_name = parts[0], parts[1]
    gh = GitHubClient(token=github_token or None)
    try:
        content = gh.get_file_content(owner, repo_name, path)
    except Exception as exc:
        return "", f"❌  Error fetching file: {exc}"
    return content, f"✅  Fetched '{path}' from {owner}/{repo_name} ({len(content)} chars)."


def _parse_requirements(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
    requirements_text: str,
    requirements_file: Any,
    use_milestones: bool,
    milestones_repo: str,
) -> tuple[list[list[Any]], str]:
    """Parse requirements text or uploaded file and return draft rows."""
    if requirements_file is not None:
        # gr.File returns a path to a Gradio-managed temporary file.
        # Resolve and validate the path to prevent path-traversal attacks.
        try:
            real_file = os.path.realpath(str(requirements_file))
            real_tmp = os.path.realpath(tempfile.gettempdir())
            if not real_file.startswith(real_tmp + os.sep):
                return _EMPTY_TABLE, "❌  Uploaded file path is outside the allowed directory."
            with open(real_file, encoding="utf-8") as fh:
                requirements_text = fh.read()
        except Exception as exc:
            return _EMPTY_TABLE, f"❌  Error reading uploaded file: {exc}"

    if not requirements_text.strip():
        return _EMPTY_TABLE, "❌  Provide requirements text or upload a markdown file."

    effective_key = (openai_key or "").strip() or os.environ.get("OPENAI_API_KEY")
    if not effective_key and not (base_url or "").strip():
        return _EMPTY_TABLE, (
            "❌  An OpenAI API key is required. "
            "Enter it in the field above or set OPENAI_API_KEY."
        )

    # Optionally fetch milestones to guide the LLM
    milestones = None
    if use_milestones and milestones_repo and "/" in milestones_repo:
        try:
            ms_parts = milestones_repo.strip().split("/", 1)
            ms_owner, ms_repo_name = ms_parts[0], ms_parts[1]
            gh_ms = GitHubClient(token=github_token or None)
            milestones = gh_ms.list_milestones(ms_owner, ms_repo_name, state="open") or None
        except Exception:
            milestones = None

    try:
        _, factory = _make_clients(github_token, effective_key or "", model, base_url)
        drafts = factory.parse_requirements(requirements_text, milestones=milestones)
    except Exception as exc:
        return _EMPTY_TABLE, f"❌  Error parsing requirements: {exc}"

    if not drafts:
        return _EMPTY_TABLE, "⚠️  No issues were extracted from the requirements document."

    milestone_note = (
        f" (with {len(milestones)} milestone(s) as context)" if milestones else ""
    )
    rows = _drafts_to_table(drafts)
    return rows, f"✅  Extracted {len(drafts)} issue draft(s){milestone_note}. Review and edit below."


# ---------------------------------------------------------------------------
# Tab: Submit Issues
# ---------------------------------------------------------------------------

def _submit_issues(
    github_token: str,
    repo: str,
    milestone_override: str,
    draft_rows: Any,
) -> str:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."

    # Gradio may pass a pandas DataFrame when the input is a gr.Dataframe component.
    try:
        import pandas as pd
        if isinstance(draft_rows, pd.DataFrame):
            draft_rows = draft_rows.values.tolist()
    except ImportError:
        pass

    if not draft_rows:
        return "❌  No issue drafts to submit. Parse requirements first."

    parts = repo.strip().split("/", 1)
    owner, repo_name = parts[0], parts[1]

    milestone_num: Optional[int] = None
    if str(milestone_override).strip():
        try:
            milestone_num = int(str(milestone_override).strip())
        except ValueError:
            return f"❌  Invalid milestone number '{milestone_override}'."

    drafts = _table_to_drafts(draft_rows)
    if not drafts:
        return "❌  All rows appear empty. Nothing to submit."

    gh = GitHubClient(token=github_token or None)
    factory = IssueFactory(
        github_client=gh,
        openai_api_key=None,
        model="gpt-4o-mini",
    )

    lines = [f"Creating {len(drafts)} issue(s) in {owner}/{repo_name}…\n"]
    try:
        results = factory.push_issues(owner, repo_name, drafts, milestone=milestone_num)
    except Exception as exc:
        return f"❌  Error creating issues: {exc}"

    lines.append(f"✅  Created {len(results)} issue(s):\n")
    for r in results:
        lines.append(f"  #{r.get('number')}  {r.get('html_url', '')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Build the Gradio UI
# ---------------------------------------------------------------------------

def build_app() -> gr.Blocks:
    settings = AppSettings()

    with gr.Blocks(title="git-review · Issue Manager") as app:
        gr.Markdown("# 🔍 git-review · Issue Manager")
        gr.Markdown(
            "Summarize GitHub activity, manage milestones, and create issues "
            "from requirements documents."
        )

        # ── Shared credentials accordion ────────────────────────────────────
        with gr.Accordion("🔑 Credentials & Settings", open=True):
            with gr.Row():
                github_token = gr.Textbox(
                    label="GitHub Token",
                    placeholder="ghp_…  (or set GITHUB_TOKEN env var)",
                    type="password",
                    value=settings.github_token,
                )
                openai_key = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="sk-…  (or set OPENAI_API_KEY env var)",
                    type="password",
                    value=settings.openai_api_key,
                )
            with gr.Row():
                model = gr.Textbox(
                    label="LLM Model",
                    value=settings.git_review_model,
                )
                base_url = gr.Textbox(
                    label="Custom API Base URL (optional)",
                    placeholder="http://localhost:11434/v1",
                    value=settings.openai_base_url,
                )

        with gr.Tabs():
            # ── Tab 1: Summarize Activity ─────────────────────────────────
            with gr.Tab("📊 Summarize Activity"):
                gr.Markdown(
                    "Fetch GitHub activity for a repository and generate an "
                    "AI-powered summary. Customize the prompt or use the default."
                )
                with gr.Row():
                    sum_repo = gr.Textbox(
                        label="Repository (owner/repo) or Owner",
                        placeholder="myorg/myrepo",
                    )
                    sum_all_repos = gr.Checkbox(
                        label="Summarize all repositories for this owner",
                        value=False,
                    )
                with gr.Row():
                    sum_author = gr.Textbox(
                        label="Filter commits by author (optional)",
                        placeholder="github-username",
                    )
                with gr.Row():
                    sum_days = gr.Number(
                        label="Days back (used when Since is empty)",
                        value=7,
                        minimum=1,
                        maximum=365,
                        precision=0,
                    )
                    sum_since = gr.Textbox(
                        label="Since (YYYY-MM-DD, optional)",
                        placeholder="2024-01-01",
                    )
                    sum_until = gr.Textbox(
                        label="Until (YYYY-MM-DD, optional — defaults to today)",
                        placeholder="2024-01-31",
                    )
                sum_prompt = gr.Textbox(
                    label="Custom system prompt (optional — leave blank to use default)",
                    lines=4,
                    placeholder=(
                        "You are an engineering manager… "
                        "Available Jinja2 variables: {{ n }}, {{ n_commits }}, "
                        "{{ n_issues }}, {{ n_prs }}"
                    ),
                )
                sum_btn = gr.Button("Generate Summary", variant="primary")
                sum_status = gr.Textbox(label="Status", interactive=False)
                sum_output = gr.Markdown(label="AI Summary")

                sum_btn.click(
                    fn=_summarize_activity,
                    inputs=[
                        github_token, openai_key, model, base_url,
                        sum_repo, sum_days, sum_since, sum_until,
                        sum_author, sum_prompt, sum_all_repos,
                    ],
                    outputs=[sum_output, sum_status],
                )

            # ── Tab 2: Milestones ─────────────────────────────────────────
            with gr.Tab("🏁 Milestones"):
                gr.Markdown("### Create a Milestone")
                with gr.Row():
                    ms_repo = gr.Textbox(
                        label="Repository (owner/repo)",
                        placeholder="myorg/myrepo",
                    )
                with gr.Row():
                    ms_title = gr.Textbox(label="Title", placeholder="v1.0 Release")
                    ms_due = gr.Textbox(
                        label="Due Date (YYYY-MM-DD, optional)",
                        placeholder="2024-12-31",
                    )
                ms_description = gr.Textbox(
                    label="Description (optional)",
                    lines=2,
                )
                ms_state = gr.Radio(
                    choices=["open", "closed"],
                    value="open",
                    label="State",
                )
                ms_create_btn = gr.Button("Create Milestone", variant="primary")
                ms_create_output = gr.Textbox(label="Result", interactive=False)

                ms_create_btn.click(
                    fn=_create_milestone,
                    inputs=[github_token, ms_repo, ms_title, ms_description, ms_due, ms_state],
                    outputs=ms_create_output,
                )

                gr.Markdown("---")
                gr.Markdown("### List Milestones")
                with gr.Row():
                    ms_list_repo = gr.Textbox(
                        label="Repository (owner/repo)",
                        placeholder="myorg/myrepo",
                    )
                    ms_list_state = gr.Radio(
                        choices=["open", "closed", "all"],
                        value="open",
                        label="State filter",
                    )
                ms_list_btn = gr.Button("List Milestones")
                ms_list_output = gr.Textbox(label="Milestones", interactive=False, lines=8)

                ms_list_btn.click(
                    fn=_list_milestones,
                    inputs=[github_token, ms_list_repo, ms_list_state],
                    outputs=ms_list_output,
                )

            # ── Tab 3: Parse Requirements ─────────────────────────────────
            with gr.Tab("📄 Parse Requirements"):
                gr.Markdown(
                    "Upload a markdown requirements document, paste it below, or "
                    "fetch it directly from a GitHub repository.  "
                    "The LLM will extract individual issues as drafts."
                )

                # Fetch from repo section
                with gr.Accordion("🔗 Fetch requirements from GitHub repo", open=False):
                    gr.Markdown(
                        "Enter the repository and the path to the requirements file "
                        "(default: `docs/requirements.md`).  "
                        "Click **Fetch** to load it into the text area below."
                    )
                    with gr.Row():
                        fetch_repo = gr.Textbox(
                            label="Repository (owner/repo)",
                            placeholder="myorg/myrepo",
                        )
                        fetch_path = gr.Textbox(
                            label="File path in repo",
                            value="docs/requirements.md",
                            placeholder="docs/requirements.md",
                        )
                    fetch_btn = gr.Button("Fetch from GitHub")
                    fetch_status = gr.Textbox(label="Fetch status", interactive=False)

                req_file = gr.File(
                    label="Upload Markdown File (optional)",
                    file_types=[".md", ".txt"],
                )
                req_text = gr.Textbox(
                    label="Or paste requirements here",
                    lines=10,
                    placeholder="## Feature Requirements\n- Users should be able to log in…",
                )

                # Milestone context option
                with gr.Row():
                    use_milestones_chk = gr.Checkbox(
                        label="Include repo milestones as AI context",
                        value=False,
                        info=(
                            "Fetch open milestones from the repo and pass them to the "
                            "LLM so it can assign each issue to the right milestone."
                        ),
                    )
                    milestones_repo = gr.Textbox(
                        label="Milestones repo (owner/repo)",
                        placeholder="myorg/myrepo  (leave blank to use Fetch repo above)",
                    )

                parse_btn = gr.Button("Parse Requirements", variant="primary")
                parse_status = gr.Textbox(label="Status", interactive=False)

                gr.Markdown("### Issue Drafts (editable)")
                gr.Markdown(
                    "You can edit Title, Body, Labels (comma-separated), "
                    "Assignees (comma-separated), and Milestone # directly in the table."
                )
                drafts_table = gr.Dataframe(
                    headers=_DRAFT_COLUMNS,
                    datatype=["number", "str", "str", "str", "str", "str"],
                    wrap=True,
                )

                def _resolve_milestones_repo(milestones_repo_val: str, fetch_repo_val: str) -> str:
                    """Return milestones_repo if set, otherwise fall back to fetch_repo."""
                    return (milestones_repo_val or "").strip() or (fetch_repo_val or "").strip()

                fetch_btn.click(
                    fn=_fetch_requirements_from_repo,
                    inputs=[github_token, fetch_repo, fetch_path],
                    outputs=[req_text, fetch_status],
                )

                parse_btn.click(
                    fn=lambda gh_tok, oai_key, mdl, burl, rtxt, rfile, use_ms, ms_repo, fr_repo: (
                        _parse_requirements(
                            gh_tok, oai_key, mdl, burl, rtxt, rfile,
                            use_ms,
                            _resolve_milestones_repo(ms_repo, fr_repo),
                        )
                    ),
                    inputs=[
                        github_token, openai_key, model, base_url,
                        req_text, req_file,
                        use_milestones_chk, milestones_repo, fetch_repo,
                    ],
                    outputs=[drafts_table, parse_status],
                )

            # ── Tab 4: Submit Issues ─────────────────────────────────────
            with gr.Tab("🚀 Submit Issues"):
                gr.Markdown(
                    "Review the table below (carried over from the Parse tab) and "
                    "submit selected issues to GitHub."
                )
                submit_repo = gr.Textbox(
                    label="Repository (owner/repo)",
                    placeholder="myorg/myrepo",
                )
                submit_milestone = gr.Textbox(
                    label="Milestone # Override (optional — overrides per-row milestone)",
                    placeholder="1",
                )

                gr.Markdown("### Issue Drafts (editable)")
                submit_table = gr.Dataframe(
                    headers=_DRAFT_COLUMNS,
                    datatype=["number", "str", "str", "str", "str", "str"],
                    wrap=True,
                )

                gr.Markdown(
                    "_Tip: switch back to the **Parse Requirements** tab and "
                    "use the button below to copy the latest drafts here._"
                )
                copy_btn = gr.Button("↩ Copy drafts from Parse tab")
                submit_btn = gr.Button("Submit Issues to GitHub", variant="primary")
                submit_output = gr.Textbox(label="Result", interactive=False, lines=10)

                # Wire copy button: copy drafts_table → submit_table
                copy_btn.click(
                    fn=lambda rows: rows,
                    inputs=drafts_table,
                    outputs=submit_table,
                )

                submit_btn.click(
                    fn=_submit_issues,
                    inputs=[github_token, submit_repo, submit_milestone, submit_table],
                    outputs=submit_output,
                )

    return app


def main() -> None:
    """Launch the Gradio app."""
    from git_review.config import AppSettings

    settings = AppSettings()
    app = build_app()
    app.launch(
        server_name=settings.gradio_server_name,
        server_port=settings.gradio_server_port,
    )


if __name__ == "__main__":
    main()
