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
"""

from __future__ import annotations

import logging
import tempfile
import os
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
) -> tuple[str, str]:
    """Fetch GitHub activity and return (summary_markdown, status_text)."""
    if not repo or "/" not in repo:
        return "", "❌  Please enter the repository in 'owner/repo' format."

    parts = repo.strip().split("/", 1)
    owner, repo_name = parts[0], parts[1]

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
    review_data = ReviewSummary(owner=owner, repo=repo_name, since=since, until=until)
    errors: list[str] = []

    for fetch_fn, label in (
        (
            lambda: gh.get_commits(
                owner, repo_name, since, until,
                author=author.strip() or None,
                include_stats=True,
            ),
            "commits",
        ),
        (lambda: gh.get_issues(owner, repo_name, since, until), "issues"),
        (
            lambda: gh.get_pull_requests(
                owner, repo_name, since, until, include_details=True
            ),
            "pull requests",
        ),
        (lambda: gh.get_releases(owner, repo_name, since, until), "releases"),
        (lambda: gh.get_contributors(owner, repo_name), "contributors"),
    ):
        try:
            results = fetch_fn()
            if label == "commits":
                review_data.commits = results
            elif label == "issues":
                review_data.issues = results
            elif label == "pull requests":
                review_data.pull_requests = results
            elif label == "releases":
                review_data.releases = results
            elif label == "contributors":
                review_data.contributors = results
        except Exception as exc:
            errors.append(f"⚠️  Could not fetch {label}: {exc}")

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

    status = (
        f"✅  Summary generated for {owner}/{repo_name} "
        f"({since.date()} → {until.date()})."
    )
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


def _parse_requirements(
    github_token: str,
    openai_key: str,
    model: str,
    base_url: str,
    requirements_text: str,
    requirements_file: Any,
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

    try:
        _, factory = _make_clients(github_token, effective_key or "", model, base_url)
        drafts = factory.parse_requirements(requirements_text)
    except Exception as exc:
        return _EMPTY_TABLE, f"❌  Error parsing requirements: {exc}"

    if not drafts:
        return _EMPTY_TABLE, "⚠️  No issues were extracted from the requirements document."

    rows = _drafts_to_table(drafts)
    return rows, f"✅  Extracted {len(drafts)} issue draft(s). Review and edit below."


# ---------------------------------------------------------------------------
# Tab: Submit Issues
# ---------------------------------------------------------------------------

def _submit_issues(
    github_token: str,
    repo: str,
    milestone_override: str,
    draft_rows: list[list[Any]],
) -> str:
    if not repo or "/" not in repo:
        return "❌  Please enter the repository in 'owner/repo' format."
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
                        label="Repository (owner/repo)",
                        placeholder="myorg/myrepo",
                    )
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
                        sum_author, sum_prompt,
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
                    "Upload a markdown requirements document or paste it below. "
                    "The LLM will extract individual issues as drafts."
                )
                req_file = gr.File(
                    label="Upload Markdown File (optional)",
                    file_types=[".md", ".txt"],
                )
                req_text = gr.Textbox(
                    label="Or paste requirements here",
                    lines=10,
                    placeholder="## Feature Requirements\n- Users should be able to log in…",
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
                    column_count=6,
                    wrap=True,
                )

                parse_btn.click(
                    fn=_parse_requirements,
                    inputs=[github_token, openai_key, model, base_url, req_text, req_file],
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
                    column_count=6,
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
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()

