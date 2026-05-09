"""Reflex web application for git-review.

Launch
------
::

    git-review-agent

or, from ``git_review/agent_app/``::

    reflex run

The app combines the conversational agent with dedicated workflow pages for
activity summaries, milestones, requirements, ServiceNow sync, and agile
planning. Credentials and defaults come from ``.env`` / environment variables
handled by :class:`git_review.config.AppSettings`.
"""

from __future__ import annotations

import os
import subprocess
import sys

try:
    import reflex as rx
except ImportError as _exc:  # pragma: no cover
    raise ImportError(
        "The 'reflex' package is required to run the agent app. "
        "Install it with:  pip install 'git-review[agent]'"
    ) from _exc

from .components.chat import chat_thread
from .components.hitl_panel import hitl_panel
from .components.settings import settings_panel
from .state import AppState, RequirementDraft

_NAV_ITEMS = [
    ("/", "Overview"),
    ("/agent", "Agent"),
    ("/activity", "Activity"),
    ("/milestones", "Milestones"),
    ("/requirements", "Requirements"),
    ("/servicenow", "ServiceNow"),
    ("/agile", "Agile"),
]


def _nav_links() -> rx.Component:
    return rx.hstack(
        *[
            rx.link(
                rx.button(label, variant="soft", size="2", color_scheme="gray"),
                href=route,
                underline="none",
            )
            for route, label in _NAV_ITEMS
        ],
        spacing="2",
        wrap="wrap",
    )


def _top_bar(*actions: rx.Component) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.hstack(
                rx.hstack(
                    rx.icon("git-branch", size=20, color=rx.color("indigo", 10)),
                    rx.vstack(
                        rx.heading("git-review", size="5"),
                        rx.text(
                            "Production workflows for reviews, planning, and delivery.",
                            size="2",
                            color_scheme="gray",
                        ),
                        spacing="1",
                        align_items="start",
                    ),
                    spacing="3",
                    align_items="center",
                ),
                rx.spacer(),
                rx.hstack(
                    *actions,
                    rx.icon_button(
                        rx.icon("settings", size=18),
                        on_click=AppState.toggle_settings,
                        size="2",
                        variant="ghost",
                        color_scheme="gray",
                    ),
                    spacing="2",
                ),
                width="100%",
                align_items="center",
            ),
            _nav_links(),
            spacing="4",
            width="100%",
        ),
        position="sticky",
        top="0",
        z_index="20",
        background_color=rx.color("gray", 1),
        border_bottom=f"1px solid {rx.color('gray', 4)}",
        padding_x="6",
        padding_y="4",
    )


def _page_shell(
    title: str,
    description: str,
    *children: rx.Component,
    actions: list[rx.Component] | None = None,
) -> rx.Component:
    return rx.box(
        _top_bar(*(actions or [])),
        rx.box(
            rx.vstack(
                rx.vstack(
                    rx.heading(title, size="7"),
                    rx.text(description, size="3", color_scheme="gray"),
                    spacing="2",
                    align_items="start",
                    width="100%",
                ),
                *children,
                spacing="6",
                width="100%",
                align_items="start",
            ),
            width="100%",
            max_width="1180px",
            margin="0 auto",
            padding_x="6",
            padding_y="6",
        ),
        settings_panel(),
        width="100%",
        min_height="100vh",
        background_color=rx.color("gray", 2),
        font_family="Inter, system-ui, sans-serif",
    )


def _section_card(
    title: str,
    description: str,
    *children: rx.Component,
) -> rx.Component:
    return rx.card(
        rx.vstack(
            rx.vstack(
                rx.heading(title, size="4"),
                rx.text(description, size="2", color_scheme="gray"),
                spacing="1",
                align_items="start",
                width="100%",
            ),
            *children,
            spacing="4",
            width="100%",
            align_items="start",
        ),
        width="100%",
    )


def _labeled_field(label: str, control: rx.Component, help_text: str = "") -> rx.Component:
    children = [
        rx.text(label, size="2", weight="medium"),
        control,
    ]
    if help_text:
        children.append(rx.text(help_text, size="1", color_scheme="gray"))
    return rx.vstack(*children, spacing="2", width="100%", align_items="start")


def _choice_button(field_name: str, value: str, label: str, color_scheme: str = "indigo") -> rx.Component:
    return rx.button(
        label,
        size="2",
        variant=rx.cond(getattr(AppState, field_name) == value, "solid", "outline"),
        color_scheme=rx.cond(getattr(AppState, field_name) == value, color_scheme, "gray"),
        on_click=AppState.set_workflow_field(field_name, value),
    )


def _status_block(title: str, text: rx.Var | str) -> rx.Component:
    return rx.vstack(
        rx.text(title, size="2", weight="medium"),
        rx.box(
            rx.text(text, white_space="pre-wrap", size="2"),
            width="100%",
            background_color="white",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="12px",
            padding="4",
            min_height="96px",
        ),
        spacing="2",
        width="100%",
        align_items="start",
    )


def _markdown_block(title: str, text: rx.Var | str) -> rx.Component:
    return rx.vstack(
        rx.text(title, size="2", weight="medium"),
        rx.box(
            rx.markdown(text),
            width="100%",
            background_color="white",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="12px",
            padding="4",
            min_height="160px",
        ),
        spacing="2",
        width="100%",
        align_items="start",
    )


def _repo_shortcut(field_name: str) -> rx.Component:
    return rx.button(
        "Use repo from settings",
        size="1",
        variant="ghost",
        color_scheme="gray",
        on_click=AppState.use_settings_repo(field_name),
    )


def _workflow_card(title: str, description: str, href: str) -> rx.Component:
    return rx.link(
        rx.card(
            rx.vstack(
                rx.hstack(
                    rx.heading(title, size="4"),
                    rx.spacer(),
                    rx.icon("arrow-right", size=18, color=rx.color("indigo", 10)),
                    width="100%",
                    align_items="center",
                ),
                rx.text(description, size="2", color_scheme="gray"),
                spacing="3",
                width="100%",
                align_items="start",
            ),
            width="100%",
        ),
        href=href,
        underline="none",
        width="100%",
    )


def _chat_input() -> rx.Component:
    return rx.hstack(
        rx.text_area(
            value=AppState.input_value,
            on_change=AppState.set_input_value,
            placeholder="Ask me to list issues, plan a sprint, create a PR draft…",
            disabled=AppState.input_disabled,
            size="3",
            flex="1",
            min_rows=1,
            max_rows=6,
            on_key_down=rx.cond(
                rx.Var.create("event.key === 'Enter' && !event.shiftKey"),
                rx.prevent_default(AppState.send_message()),
                rx.Var.create("null"),
            ),
        ),
        rx.icon_button(
            rx.icon("send", size=18),
            on_click=AppState.send_message,
            disabled=AppState.input_disabled,
            size="3",
            color_scheme="indigo",
            variant="solid",
        ),
        width="100%",
        spacing="2",
        align_items="end",
    )


def _overview_page() -> rx.Component:
    return _page_shell(
        "Workspace overview",
        "Move from discovery to execution with a single Reflex app instead of jumping between separate tools.",
        rx.vstack(
            _section_card(
                "Recommended flow",
                "Start with the structured workflows, then drop into the agent when you need open-ended help.",
                rx.vstack(
                    rx.text("1. Configure credentials and default repo in Settings."),
                    rx.text("2. Use Activity and Agile to understand the current state."),
                    rx.text("3. Convert requirements into issue drafts and submit them."),
                    rx.text("4. Use the agent for follow-up analysis, issue triage, and PR work."),
                    spacing="2",
                    width="100%",
                    align_items="start",
                ),
            ),
            _workflow_card(
                "Conversational agent",
                "Streaming chat with tool calls, reasoning, and human-in-the-loop approvals.",
                "/agent",
            ),
            _workflow_card(
                "Activity summary",
                "Generate an AI summary across one repo or every repo in an org.",
                "/activity",
            ),
            _workflow_card(
                "Milestones",
                "Create release milestones and review the current roadmap in one place.",
                "/milestones",
            ),
            _workflow_card(
                "Requirements to issues",
                "Fetch or paste requirements, generate editable drafts, and submit them without leaving the page.",
                "/requirements",
            ),
            _workflow_card(
                "ServiceNow sync",
                "Preview or apply the GitHub → ServiceNow sync with shared settings.",
                "/servicenow",
            ),
            _workflow_card(
                "Agile planner",
                "Generate dependency-aware sprint plans and apply the approved changes back to GitHub.",
                "/agile",
            ),
            spacing="4",
            width="100%",
        ),
    )


def _agent_page() -> rx.Component:
    return _page_shell(
        "Conversational agent",
        "Use the agent for exploratory work, issue updates, pull requests, and human-in-the-loop approvals.",
        rx.card(
            rx.vstack(
                chat_thread(),
                rx.box(
                    rx.vstack(
                        hitl_panel(),
                        _chat_input(),
                        spacing="3",
                        width="100%",
                    ),
                    width="100%",
                    padding_top="3",
                    border_top=f"1px solid {rx.color('gray', 4)}",
                ),
                spacing="4",
                width="100%",
                height="calc(100vh - 260px)",
            ),
            width="100%",
        ),
        actions=[
            rx.button(
                rx.icon("trash-2", size=14),
                "Clear chat",
                size="2",
                variant="ghost",
                color_scheme="gray",
                on_click=AppState.clear_chat,
            )
        ],
    )


def _activity_page() -> rx.Component:
    return _page_shell(
        "Activity summary",
        "Summarize repository activity with a tighter workflow and shared settings.",
        _section_card(
            "Summary inputs",
            "Review one repository or switch to org mode for an aggregate view.",
            rx.hstack(
                _labeled_field(
                    "Repository or owner",
                    rx.input(
                        value=AppState.summary_repo,
                        on_change=lambda value: AppState.set_workflow_field("summary_repo", value),
                        placeholder="owner/repo or owner",
                        width="100%",
                    ),
                ),
                _repo_shortcut("summary_repo"),
                width="100%",
                align_items="end",
            ),
            rx.hstack(
                _labeled_field(
                    "Commit author filter",
                    rx.input(
                        value=AppState.summary_author,
                        on_change=lambda value: AppState.set_workflow_field("summary_author", value),
                        placeholder="github-username",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Days back",
                    rx.input(
                        value=AppState.summary_days,
                        on_change=lambda value: AppState.set_workflow_field("summary_days", value),
                        placeholder="7",
                        width="100%",
                    ),
                ),
                width="100%",
            ),
            rx.hstack(
                _labeled_field(
                    "Since",
                    rx.input(
                        value=AppState.summary_since,
                        on_change=lambda value: AppState.set_workflow_field("summary_since", value),
                        placeholder="YYYY-MM-DD",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Until",
                    rx.input(
                        value=AppState.summary_until,
                        on_change=lambda value: AppState.set_workflow_field("summary_until", value),
                        placeholder="YYYY-MM-DD",
                        width="100%",
                    ),
                ),
                width="100%",
            ),
            _labeled_field(
                "Custom system prompt",
                rx.text_area(
                    value=AppState.summary_prompt,
                    on_change=lambda value: AppState.set_workflow_field("summary_prompt", value),
                    min_rows=4,
                    width="100%",
                ),
            ),
            rx.hstack(
                rx.button(
                    rx.cond(AppState.summary_all_repos, "Owner mode enabled", "Single repo mode"),
                    size="2",
                    variant=rx.cond(AppState.summary_all_repos, "solid", "outline"),
                    color_scheme=rx.cond(AppState.summary_all_repos, "indigo", "gray"),
                    on_click=AppState.toggle_workflow_flag("summary_all_repos"),
                ),
                rx.button("Generate summary", on_click=AppState.generate_summary, color_scheme="indigo"),
                spacing="3",
            ),
        ),
        _status_block("Status", AppState.summary_status),
        _markdown_block("Summary", AppState.summary_output),
    )


def _milestones_page() -> rx.Component:
    return _page_shell(
        "Milestones",
        "Create milestones and review the current release plan without leaving the app.",
        _section_card(
            "Create a milestone",
            "Use the shared repo settings or override them here for a one-off release.",
            rx.hstack(
                _labeled_field(
                    "Repository",
                    rx.input(
                        value=AppState.milestone_repo,
                        on_change=lambda value: AppState.set_workflow_field("milestone_repo", value),
                        placeholder="owner/repo",
                        width="100%",
                    ),
                ),
                _repo_shortcut("milestone_repo"),
                width="100%",
                align_items="end",
            ),
            rx.hstack(
                _labeled_field(
                    "Title",
                    rx.input(
                        value=AppState.milestone_title,
                        on_change=lambda value: AppState.set_workflow_field("milestone_title", value),
                        placeholder="v1.0 Release",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Due date",
                    rx.input(
                        value=AppState.milestone_due_on,
                        on_change=lambda value: AppState.set_workflow_field("milestone_due_on", value),
                        placeholder="YYYY-MM-DD",
                        width="100%",
                    ),
                ),
                width="100%",
            ),
            _labeled_field(
                "Description",
                rx.text_area(
                    value=AppState.milestone_description,
                    on_change=lambda value: AppState.set_workflow_field("milestone_description", value),
                    min_rows=3,
                    width="100%",
                ),
            ),
            rx.hstack(
                _choice_button("milestone_state", "open", "Open"),
                _choice_button("milestone_state", "closed", "Closed"),
                rx.button("Create milestone", on_click=AppState.create_milestone_workflow, color_scheme="indigo"),
                spacing="3",
            ),
            _status_block("Create result", AppState.milestone_create_result),
        ),
        _section_card(
            "List milestones",
            "Review the current milestone backlog with a quick repo-level query.",
            rx.hstack(
                _labeled_field(
                    "Repository",
                    rx.input(
                        value=AppState.milestone_list_repo,
                        on_change=lambda value: AppState.set_workflow_field("milestone_list_repo", value),
                        placeholder="owner/repo",
                        width="100%",
                    ),
                ),
                _repo_shortcut("milestone_list_repo"),
                width="100%",
                align_items="end",
            ),
            rx.hstack(
                _choice_button("milestone_list_state", "open", "Open"),
                _choice_button("milestone_list_state", "closed", "Closed"),
                _choice_button("milestone_list_state", "all", "All"),
                rx.button("List milestones", on_click=AppState.list_milestones_workflow),
                spacing="3",
            ),
            _status_block("Milestones", AppState.milestone_list_output),
        ),
    )


def _draft_editor(draft: RequirementDraft, idx: rx.Var) -> rx.Component:
    return rx.card(
        rx.vstack(
            _labeled_field(
                "Title",
                rx.input(
                    value=draft.title,
                    on_change=lambda value: AppState.update_requirement_draft(value, idx, "title"),
                    width="100%",
                ),
            ),
            _labeled_field(
                "Body",
                rx.text_area(
                    value=draft.body,
                    on_change=lambda value: AppState.update_requirement_draft(value, idx, "body"),
                    min_rows=8,
                    width="100%",
                ),
            ),
            rx.hstack(
                _labeled_field(
                    "Labels",
                    rx.input(
                        value=draft.labels,
                        on_change=lambda value: AppState.update_requirement_draft(value, idx, "labels"),
                        placeholder="bug, enhancement",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Assignees",
                    rx.input(
                        value=draft.assignees,
                        on_change=lambda value: AppState.update_requirement_draft(value, idx, "assignees"),
                        placeholder="alice, bob",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Milestone #",
                    rx.input(
                        value=draft.milestone,
                        on_change=lambda value: AppState.update_requirement_draft(value, idx, "milestone"),
                        placeholder="1",
                        width="100%",
                    ),
                ),
                width="100%",
                align_items="start",
            ),
            spacing="3",
            width="100%",
            align_items="start",
        ),
        width="100%",
    )


def _requirements_page() -> rx.Component:
    return _page_shell(
        "Requirements to issues",
        "Fetch requirements, generate issue drafts, edit them inline, and submit them from one page.",
        _section_card(
            "Step 1 · Load requirements",
            "Pull a markdown file from GitHub or paste the source document directly.",
            rx.hstack(
                _labeled_field(
                    "Repository",
                    rx.input(
                        value=AppState.requirements_repo,
                        on_change=lambda value: AppState.set_workflow_field("requirements_repo", value),
                        placeholder="owner/repo",
                        width="100%",
                    ),
                ),
                _repo_shortcut("requirements_repo"),
                width="100%",
                align_items="end",
            ),
            _labeled_field(
                "Requirements path",
                rx.input(
                    value=AppState.requirements_path,
                    on_change=lambda value: AppState.set_workflow_field("requirements_path", value),
                    placeholder="docs/requirements.md",
                    width="100%",
                ),
            ),
            rx.hstack(
                rx.button("Fetch from GitHub", on_click=AppState.fetch_requirements_text),
                rx.button("Copy repo to submit target", on_click=AppState.sync_submit_repo_from_requirements, variant="ghost"),
                spacing="3",
            ),
            _status_block("Fetch status", AppState.requirements_fetch_status),
            _labeled_field(
                "Requirements text",
                rx.text_area(
                    value=AppState.requirements_text,
                    on_change=lambda value: AppState.set_workflow_field("requirements_text", value),
                    min_rows=12,
                    width="100%",
                ),
            ),
        ),
        _section_card(
            "Step 2 · Generate drafts",
            "Optionally provide milestone context so the generated issues map to the current plan.",
            rx.hstack(
                rx.button(
                    rx.cond(
                        AppState.requirements_use_milestones,
                        "Milestone context enabled",
                        "Milestone context disabled",
                    ),
                    size="2",
                    variant=rx.cond(AppState.requirements_use_milestones, "solid", "outline"),
                    color_scheme=rx.cond(AppState.requirements_use_milestones, "indigo", "gray"),
                    on_click=AppState.toggle_workflow_flag("requirements_use_milestones"),
                ),
                rx.button("Parse requirements", on_click=AppState.parse_requirements_workflow, color_scheme="indigo"),
                rx.button("Clear drafts", on_click=AppState.clear_requirement_drafts, variant="ghost", color_scheme="gray"),
                spacing="3",
            ),
            _labeled_field(
                "Milestones repo",
                rx.input(
                    value=AppState.requirements_milestones_repo,
                    on_change=lambda value: AppState.set_workflow_field("requirements_milestones_repo", value),
                    placeholder="owner/repo",
                    width="100%",
                ),
                "Leave blank to reuse the requirements repository.",
            ),
            _status_block("Parse status", AppState.requirements_status),
        ),
        _section_card(
            "Step 3 · Review drafts",
            "Edit the generated drafts before you push them to GitHub.",
            rx.cond(
                AppState.requirement_drafts.length() > 0,
                rx.vstack(
                    rx.foreach(AppState.requirement_drafts, _draft_editor),
                    spacing="4",
                    width="100%",
                ),
                rx.box(
                    rx.text("No drafts yet. Parse requirements to generate editable issue drafts.", color_scheme="gray"),
                    width="100%",
                    background_color="white",
                    border=f"1px dashed {rx.color('gray', 5)}",
                    border_radius="12px",
                    padding="4",
                ),
            ),
        ),
        _section_card(
            "Step 4 · Submit issues",
            "Push the reviewed drafts into GitHub using the same credentials and repo settings.",
            rx.hstack(
                _labeled_field(
                    "Submit repository",
                    rx.input(
                        value=AppState.submit_repo,
                        on_change=lambda value: AppState.set_workflow_field("submit_repo", value),
                        placeholder="owner/repo",
                        width="100%",
                    ),
                ),
                _repo_shortcut("submit_repo"),
                width="100%",
                align_items="end",
            ),
            _labeled_field(
                "Milestone override",
                rx.input(
                    value=AppState.submit_milestone_override,
                    on_change=lambda value: AppState.set_workflow_field("submit_milestone_override", value),
                    placeholder="Optional milestone number",
                    width="100%",
                ),
            ),
            rx.button("Submit issues", on_click=AppState.submit_requirement_drafts, color_scheme="indigo"),
            _status_block("Submit status", AppState.submit_status),
        ),
    )


def _servicenow_page() -> rx.Component:
    return _page_shell(
        "ServiceNow sync",
        "Run the GitHub source-of-truth sync with the same credentials used by the agent.",
        _section_card(
            "Sync controls",
            "Use the shared ServiceNow settings and choose whether to preview or apply the sync.",
            rx.hstack(
                _labeled_field(
                    "Repository",
                    rx.input(
                        value=AppState.sync_repo,
                        on_change=lambda value: AppState.set_workflow_field("sync_repo", value),
                        placeholder="owner/repo",
                        width="100%",
                    ),
                ),
                _repo_shortcut("sync_repo"),
                width="100%",
                align_items="end",
            ),
            rx.hstack(
                rx.button(
                    rx.cond(AppState.sync_dry_run, "Dry run enabled", "Apply mode enabled"),
                    size="2",
                    variant=rx.cond(AppState.sync_dry_run, "solid", "outline"),
                    color_scheme=rx.cond(AppState.sync_dry_run, "indigo", "gray"),
                    on_click=AppState.toggle_workflow_flag("sync_dry_run"),
                ),
                rx.button("Run sync", on_click=AppState.run_servicenow_sync, color_scheme="indigo"),
                spacing="3",
            ),
            _labeled_field(
                "Back-sync allowlist",
                rx.input(
                    value=AppState.sync_back_sync_fields,
                    on_change=lambda value: AppState.set_workflow_field("sync_back_sync_fields", value),
                    placeholder="labels,assignees",
                    width="100%",
                ),
                "Only labels and assignees are supported.",
            ),
            _status_block("Sync result", AppState.sync_result),
        ),
    )


def _agile_page() -> rx.Component:
    return _page_shell(
        "Agile planner",
        "Generate dependency-aware sprint plans and apply the approved graph or label updates.",
        _section_card(
            "Planning inputs",
            "Point the planner at a repository or owner and tune the sprint configuration.",
            rx.hstack(
                _labeled_field(
                    "Repository or owner",
                    rx.input(
                        value=AppState.agile_repo,
                        on_change=lambda value: AppState.set_workflow_field("agile_repo", value),
                        placeholder="owner/repo or owner/*",
                        width="100%",
                    ),
                ),
                _repo_shortcut("agile_repo"),
                width="100%",
                align_items="end",
            ),
            rx.hstack(
                _labeled_field(
                    "Sprint capacity",
                    rx.input(
                        value=AppState.agile_capacity,
                        on_change=lambda value: AppState.set_workflow_field("agile_capacity", value),
                        placeholder="10",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Number of sprints",
                    rx.input(
                        value=AppState.agile_sprints,
                        on_change=lambda value: AppState.set_workflow_field("agile_sprints", value),
                        placeholder="3",
                        width="100%",
                    ),
                ),
                width="100%",
            ),
            rx.hstack(
                rx.button("Generate sprint plan", on_click=AppState.run_agile_workflow, color_scheme="indigo"),
                rx.button("Apply relationships", on_click=AppState.apply_agile_relationships_workflow),
                rx.button("Apply labels", on_click=AppState.apply_agile_labels_workflow),
                spacing="3",
            ),
            _status_block("Planning status", AppState.agile_status),
            _status_block("Apply status", AppState.agile_apply_status),
        ),
        _markdown_block("Dependency graph", AppState.agile_dependencies_markdown),
        _markdown_block("Sprint plan", AppState.agile_plan_markdown),
    )


app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="indigo",
        radius="large",
    ),
)
app.add_page(_overview_page, route="/", on_load=AppState.on_load)
app.add_page(_agent_page, route="/agent", on_load=AppState.on_load)
app.add_page(_activity_page, route="/activity", on_load=AppState.on_load)
app.add_page(_milestones_page, route="/milestones", on_load=AppState.on_load)
app.add_page(_requirements_page, route="/requirements", on_load=AppState.on_load)
app.add_page(_servicenow_page, route="/servicenow", on_load=AppState.on_load)
app.add_page(_agile_page, route="/agile", on_load=AppState.on_load)


def main() -> None:  # pragma: no cover
    """Launch the Reflex app via ``git-review-agent``."""
    app_dir = os.path.dirname(__file__)
    subprocess.run(
        [sys.executable, "-m", "reflex", "run"],
        cwd=app_dir,
        check=True,
    )
