"""Reflex web application for git-review.

Launch
------
Run the installed entry point::

    git-review-agent

Or run Reflex directly from ``git_review/agent_app/``::

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
from .auth import handle_github_oauth_callback, logout_session, start_github_oauth_login
from .state import AppState, RequirementDraft

_NAV_ITEMS = [
    ("/", "Overview"),
    ("/agent", "GitHub Agent"),
    ("/activity", "Activity"),
    ("/milestones", "Milestones"),
    ("/requirements", "Requirements"),
    ("/servicenow", "ServiceNow"),
    ("/agile", "Agile"),
]

SHELL_BG = "linear-gradient(180deg, #f5f8fc 0%, #eef3fa 100%)"
SURFACE_BG = "rgba(255, 255, 255, 0.96)"
SURFACE_BORDER = "1px solid rgba(20, 53, 89, 0.10)"
SURFACE_SHADOW = "0 10px 30px rgba(16, 35, 56, 0.08)"
RAIL_BG = "rgba(250, 252, 255, 0.92)"


def _is_active_route(route: str) -> rx.Var:
    return AppState.router.page.path == route


def _nav_item(route: str, label: str) -> rx.Component:
    active = _is_active_route(route)
    return rx.cond(
        AppState.sidebar_collapsed,
        rx.tooltip(
            rx.link(
                rx.button(
                    rx.box(
                        width=rx.cond(active, "0.72rem", "0.6rem"),
                        height=rx.cond(active, "0.72rem", "0.6rem"),
                        border_radius="999px",
                        background_color=rx.cond(
                            active, rx.color("indigo", 10), rx.color("gray", 8)
                        ),
                    ),
                    variant=rx.cond(active, "solid", "soft"),
                    size="3",
                    color_scheme=rx.cond(active, "indigo", "gray"),
                    width="100%",
                    height="2.3rem",
                    padding="0",
                    box_shadow=rx.cond(
                        active, "0 6px 14px rgba(55, 85, 160, 0.22)", "none"
                    ),
                ),
                href=route,
                underline="none",
                width="100%",
            ),
            content=label,
        ),
        rx.link(
            rx.button(
                rx.hstack(
                    rx.box(
                        width="0.45rem",
                        height="0.45rem",
                        border_radius="999px",
                        background_color=rx.cond(
                            active, rx.color("indigo", 10), rx.color("gray", 7)
                        ),
                    ),
                    rx.text(label, size="2", weight=rx.cond(active, "bold", "medium")),
                    spacing="3",
                    align_items="center",
                    width="100%",
                ),
                variant=rx.cond(active, "solid", "soft"),
                size="2",
                color_scheme=rx.cond(active, "indigo", "gray"),
                width="100%",
                justify="start",
                padding_x="0.7rem",
                box_shadow=rx.cond(
                    active, "0 8px 16px rgba(55, 85, 160, 0.16)", "none"
                ),
            ),
            href=route,
            underline="none",
            width="100%",
        ),
    )


def _sidebar_nav() -> rx.Component:
    return rx.vstack(
        *[_nav_item(route, label) for route, label in _NAV_ITEMS],
        spacing="2",
        width="100%",
        align_items="stretch",
    )


def _sidebar() -> rx.Component:
    return rx.box(
        rx.vstack(
            _sidebar_nav(),
            spacing="3",
            width="100%",
            align_items="stretch",
        ),
        width=rx.cond(AppState.sidebar_collapsed, "0rem", "15rem"),
        min_width=rx.cond(AppState.sidebar_collapsed, "0rem", "15rem"),
        transition="width 0.22s ease",
        border_right=rx.cond(AppState.sidebar_collapsed, "none", SURFACE_BORDER),
        background=RAIL_BG,
        padding=rx.cond(AppState.sidebar_collapsed, "0rem", "0.9rem"),
        align_self="stretch",
        position="sticky",
        top="62px",
        height="calc(100vh - 62px)",
        overflow="hidden",
        opacity=rx.cond(AppState.sidebar_collapsed, 0, 1),
        pointer_events=rx.cond(AppState.sidebar_collapsed, "none", "auto"),
        backdrop_filter="blur(6px)",
    )


def _top_bar(*actions: rx.Component) -> rx.Component:
    return rx.box(
        rx.hstack(
            rx.hstack(
                rx.icon_button(
                    rx.icon(
                        rx.cond(AppState.sidebar_collapsed, "menu", "panel_left_close"),
                        size=15,
                    ),
                    on_click=AppState.toggle_sidebar,
                    size="3",
                    variant="soft",
                    color_scheme="gray",
                    border_radius="10px",
                ),
                rx.icon("git-branch", size=18, color=rx.color("indigo", 10)),
                rx.hstack(
                    rx.heading("git-review", size="3"),
                    rx.text(
                        "Plan, ship, and sync delivery work from one workspace.",
                        size="1",
                        color_scheme="gray",
                        display=rx.breakpoints(initial="none", lg="block"),
                        white_space="nowrap",
                        overflow="hidden",
                        text_overflow="ellipsis",
                    ),
                    spacing="3",
                    align_items="center",
                    min_width="0",
                ),
                spacing="3",
                align_items="center",
                min_width="0",
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
        position="sticky",
        top="0",
        z_index="20",
        background_color="rgba(255, 255, 255, 0.98)",
        border_bottom="1px solid rgba(20, 53, 89, 0.15)",
        box_shadow="0 8px 18px rgba(16, 35, 56, 0.08)",
        padding_x="5",
        padding_y="2.5",
        backdrop_filter="blur(8px)",
    )


def _page_shell(
    title: str,
    description: str,
    *children: rx.Component,
    actions: list[rx.Component] | None = None,
    content_padding_top: str = "5",
    content_padding_bottom: str = "7",
) -> rx.Component:
    return rx.box(
        _top_bar(*(actions or [])),
        rx.hstack(
            _sidebar(),
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.heading(title, size="4"),
                        rx.text(
                            description,
                            size="1",
                            color_scheme="gray",
                            white_space="nowrap",
                            overflow="hidden",
                            text_overflow="ellipsis",
                        ),
                        spacing="3",
                        align_items="center",
                        min_width="0",
                        width="100%",
                    ),
                    *children,
                    spacing="4",
                    width="100%",
                    align_items="start",
                ),
                width="100%",
                max_width="1160px",
                margin="0 auto",
                padding_x="5",
                padding_top=content_padding_top,
                padding_bottom=content_padding_bottom,
            ),
            width="100%",
            align_items="start",
        ),
        settings_panel(),
        width="100%",
        min_height="100vh",
        background=SHELL_BG,
        font_family="Manrope, 'Segoe UI', sans-serif",
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
        border_radius="12px",
        background=SURFACE_BG,
        border=SURFACE_BORDER,
        box_shadow=SURFACE_SHADOW,
    )


def _labeled_field(
    label: rx.Var | str,
    control: rx.Component,
    help_text: rx.Var | str | None = None,
) -> rx.Component:
    children = [
        rx.text(label, size="2", weight="medium"),
        control,
    ]
    if help_text is not None:
        children.append(rx.text(help_text, size="1", color_scheme="gray"))
    return rx.vstack(*children, spacing="2", width="100%", align_items="start")


def _choice_button(
    field_name: str, value: str, label: str, color_scheme: str = "indigo"
) -> rx.Component:
    return rx.button(
        label,
        size="2",
        variant=rx.cond(getattr(AppState, field_name) == value, "solid", "outline"),
        color_scheme=rx.cond(
            getattr(AppState, field_name) == value, color_scheme, "gray"
        ),
        on_click=AppState.set_workflow_field(field_name, value),
    )


def _mode_button(
    label: str,
    active: rx.Var | bool,
    on_click: rx.event.EventHandler,
    icon: str,
) -> rx.Component:
    return rx.button(
        rx.icon(icon, size=14),
        label,
        size="2",
        variant=rx.cond(active, "solid", "ghost"),
        color_scheme=rx.cond(active, "indigo", "gray"),
        on_click=on_click,
        border_radius="6px",
    )


def _activity_mode_toggle() -> rx.Component:
    return rx.hstack(
        _mode_button(
            "Single repo",
            ~AppState.summary_all_repos,
            AppState.set_workflow_field("summary_all_repos", False),
            "git-branch",
        ),
        _mode_button(
            "Owner",
            AppState.summary_all_repos,
            AppState.set_workflow_field("summary_all_repos", True),
            "building-2",
        ),
        spacing="1",
        padding="1",
        border=f"1px solid {rx.color('gray', 4)}",
        border_radius="8px",
        background_color=rx.color("gray", 2),
        align_items="center",
    )


def _status_block(title: str, text: rx.Var | str) -> rx.Component:
    return rx.vstack(
        rx.text(title, size="2", weight="medium"),
        rx.box(
            rx.text(text, white_space="pre-wrap", size="2"),
            width="100%",
            background_color="white",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="8px",
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
            rx.text(text, white_space="pre-wrap", size="2"),
            width="100%",
            background_color="white",
            border=f"1px solid {rx.color('gray', 4)}",
            border_radius="8px",
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


def _secondary_actions_menu(
    trigger_label: str,
    entries: list[tuple[rx.Var | str, rx.event.EventHandler]],
) -> rx.Component:
    return rx.popover.root(
        rx.popover.trigger(
            rx.button(
                rx.icon("ellipsis", size=14),
                trigger_label,
                size="2",
                variant="soft",
                color_scheme="gray",
            )
        ),
        rx.popover.content(
            rx.vstack(
                *[
                    rx.button(
                        entry_label,
                        on_click=entry_action,
                        size="1",
                        variant="ghost",
                        color_scheme="gray",
                        width="100%",
                        justify="start",
                    )
                    for entry_label, entry_action in entries
                ],
                spacing="1",
                width="220px",
                align_items="stretch",
            ),
            side="bottom",
            align="end",
            padding="0.35rem",
        ),
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
            background=SURFACE_BG,
            border=SURFACE_BORDER,
            box_shadow="0 8px 20px rgba(16, 35, 56, 0.06)",
        ),
        href=href,
        underline="none",
        width="100%",
    )


def _chat_input() -> rx.Component:
    return rx.form(
        rx.box(
            rx.hstack(
                rx.text_area(
                    value=AppState.input_value,
                    on_change=AppState.set_input_value,
                    placeholder="Ask for issue triage, sprint planning, release milestones, or PR support...",
                    disabled=AppState.input_disabled,
                    size="3",
                    flex="1",
                    min_rows=1,
                    max_rows=6,
                    enter_key_submit=True,
                    border_radius="10px",
                    background_color="white",
                ),
                rx.icon_button(
                    rx.icon("send", size=18),
                    disabled=AppState.input_disabled,
                    size="3",
                    color_scheme="indigo",
                    variant="solid",
                    type="submit",
                    border_radius="10px",
                ),
                width="100%",
                spacing="2",
                align_items="end",
            ),
            width="100%",
            padding="0.45rem",
            border_radius="12px",
            background_color="rgba(255, 255, 255, 0.95)",
            border=f"1px solid {rx.color('gray', 4)}",
            box_shadow="0 4px 14px rgba(16, 35, 56, 0.08)",
        ),
        on_submit=lambda _: AppState.send_message,
        reset_on_submit=False,
        width="100%",
    )


def _overview_page() -> rx.Component:
    return _page_shell(
        "Workspace overview",
        "Run planning, execution, and sync workflows without bouncing between tools.",
        rx.vstack(
            _section_card(
                "Recommended flow",
                "Start with guided workflows, then use the agent for analysis and follow-through.",
                rx.vstack(
                    rx.text("1. Configure credentials and default repo in Settings."),
                    rx.text(
                        "2. Use Activity and Agile to baseline current delivery status."
                    ),
                    rx.text(
                        "3. Turn requirements into issue drafts and submit in batch."
                    ),
                    rx.text(
                        "4. Use the agent for follow-up analysis, triage, and PR support."
                    ),
                    spacing="2",
                    width="100%",
                    align_items="start",
                ),
            ),
            _workflow_card(
                "GitHub Agent",
                "Streaming chat with tool calls, reasoning, and approval gates for write actions.",
                "/agent",
            ),
            _workflow_card(
                "Activity summary",
                "Generate concise activity summaries across one repo or an entire owner scope.",
                "/activity",
            ),
            _workflow_card(
                "Milestones",
                "Create release milestones and review roadmap status in one place.",
                "/milestones",
            ),
            _workflow_card(
                "Requirements to issues",
                "Fetch or paste requirements, generate editable issue drafts, then submit.",
                "/requirements",
            ),
            _workflow_card(
                "ServiceNow sync",
                "Preview or apply GitHub-to-ServiceNow sync using shared settings.",
                "/servicenow",
            ),
            _workflow_card(
                "Agile planner",
                "Build dependency-aware sprint plans and apply approved updates back to GitHub.",
                "/agile",
            ),
            spacing="4",
            width="100%",
        ),
    )


def _agent_page() -> rx.Component:
    return _page_shell(
        "GitHub Agent",
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
                height="calc(100dvh - 170px)",
                min_height="0",
            ),
            width="100%",
            border_radius="14px",
            background_color="rgba(248, 250, 252, 0.72)",
            box_shadow="0 8px 24px rgba(16, 35, 56, 0.08)",
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
        content_padding_top="4",
        content_padding_bottom="2",
    )


def _activity_page() -> rx.Component:
    return _page_shell(
        "Activity summary",
        "Generate concise activity summaries with shared repository and credential settings.",
        _section_card(
            "Summary inputs",
            "Review one repository or switch to org mode for an aggregate view.",
            _activity_mode_toggle(),
            rx.hstack(
                _labeled_field(
                    rx.cond(AppState.summary_all_repos, "Owner", "Repository"),
                    rx.input(
                        value=AppState.summary_repo,
                        on_change=lambda value: AppState.set_workflow_field(
                            "summary_repo", value
                        ),
                        placeholder=rx.cond(
                            AppState.summary_all_repos,
                            "owner (owner/repo also works)",
                            "owner/repo",
                        ),
                        width="100%",
                    ),
                    rx.cond(
                        AppState.summary_all_repos,
                        "Owner mode includes all active repositories under the owner.",
                        "Single repo mode expects owner/repo.",
                    ),
                ),
                _repo_shortcut("summary_repo"),
                width="100%",
                align_items="end",
                spacing="3",
            ),
            rx.hstack(
                _labeled_field(
                    "Commit author filter",
                    rx.input(
                        value=AppState.summary_author,
                        on_change=lambda value: AppState.set_workflow_field(
                            "summary_author", value
                        ),
                        placeholder="github-username",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Days back",
                    rx.input(
                        value=AppState.summary_days,
                        on_change=lambda value: AppState.set_workflow_field(
                            "summary_days", value
                        ),
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
                        on_change=lambda value: AppState.set_workflow_field(
                            "summary_since", value
                        ),
                        placeholder="YYYY-MM-DD",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Until",
                    rx.input(
                        value=AppState.summary_until,
                        on_change=lambda value: AppState.set_workflow_field(
                            "summary_until", value
                        ),
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
                    on_change=lambda value: AppState.set_workflow_field(
                        "summary_prompt", value
                    ),
                    min_rows=4,
                    width="100%",
                ),
            ),
            rx.hstack(
                rx.button(
                    rx.icon("sparkles", size=14),
                    "Generate summary",
                    on_click=AppState.generate_summary,
                    color_scheme="indigo",
                ),
                spacing="3",
            ),
        ),
        _status_block("Status", AppState.summary_status),
        _markdown_block("Summary", AppState.summary_output),
    )


def _milestones_page() -> rx.Component:
    return _page_shell(
        "Milestones",
        "Build and publish milestone plans in one structured pass.",
        _section_card(
            "Build the milestone queue",
            "Use the single-milestone form to stage entries, then create everything in one batch.",
            rx.hstack(
                _labeled_field(
                    "Repository",
                    rx.input(
                        value=AppState.milestone_repo,
                        on_change=lambda value: AppState.set_workflow_field(
                            "milestone_repo", value
                        ),
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
                        on_change=lambda value: AppState.set_workflow_field(
                            "milestone_title", value
                        ),
                        placeholder="v1.0 Release",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Due date",
                    rx.input(
                        value=AppState.milestone_due_on,
                        on_change=lambda value: AppState.set_workflow_field(
                            "milestone_due_on", value
                        ),
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
                    on_change=lambda value: AppState.set_workflow_field(
                        "milestone_description", value
                    ),
                    min_rows=3,
                    width="100%",
                ),
            ),
            rx.hstack(
                _choice_button("milestone_state", "open", "Open"),
                _choice_button("milestone_state", "closed", "Closed"),
                rx.button(
                    "Add to queue",
                    on_click=AppState.queue_current_milestone,
                    color_scheme="indigo",
                ),
                rx.button(
                    "Create current only",
                    on_click=AppState.create_milestone_workflow,
                    variant="ghost",
                ),
                spacing="3",
            ),
            _labeled_field(
                "Queued milestones",
                rx.text_area(
                    value=AppState.milestone_queue_text,
                    on_change=lambda value: AppState.set_workflow_field(
                        "milestone_queue_text", value
                    ),
                    min_rows=8,
                    width="100%",
                ),
                "One milestone per line using: title | due_on | state | description.",
            ),
            rx.hstack(
                rx.button(
                    "Create queued milestones",
                    on_click=AppState.create_queued_milestones_workflow,
                    color_scheme="indigo",
                ),
                _secondary_actions_menu(
                    "More",
                    [
                        ("Load defaults from env", AppState.load_default_milestones),
                        ("Clear queue", AppState.clear_milestone_queue),
                    ],
                ),
                spacing="3",
            ),
            _status_block("Queue/default status", AppState.milestone_defaults_status),
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
                        on_change=lambda value: AppState.set_workflow_field(
                            "milestone_list_repo", value
                        ),
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
                rx.button(
                    "List milestones", on_click=AppState.list_milestones_workflow
                ),
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
                    on_change=lambda value: AppState.update_requirement_draft(
                        value, idx, "title"
                    ),
                    width="100%",
                ),
            ),
            _labeled_field(
                "Body",
                rx.text_area(
                    value=draft.body,
                    on_change=lambda value: AppState.update_requirement_draft(
                        value, idx, "body"
                    ),
                    min_rows=8,
                    width="100%",
                ),
            ),
            rx.hstack(
                _labeled_field(
                    "Labels",
                    rx.input(
                        value=draft.labels,
                        on_change=lambda value: AppState.update_requirement_draft(
                            value, idx, "labels"
                        ),
                        placeholder="bug, enhancement",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Assignees",
                    rx.input(
                        value=draft.assignees,
                        on_change=lambda value: AppState.update_requirement_draft(
                            value, idx, "assignees"
                        ),
                        placeholder="alice, bob",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Milestone #",
                    rx.input(
                        value=draft.milestone,
                        on_change=lambda value: AppState.update_requirement_draft(
                            value, idx, "milestone"
                        ),
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
        "Load requirements, generate issue drafts, refine them, and submit without context switching.",
        _section_card(
            "Step 1 · Load requirements",
            "Pull a markdown file from GitHub or paste the source document directly.",
            rx.hstack(
                _labeled_field(
                    "Repository",
                    rx.input(
                        value=AppState.requirements_repo,
                        on_change=lambda value: AppState.set_workflow_field(
                            "requirements_repo", value
                        ),
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
                    on_change=lambda value: AppState.set_workflow_field(
                        "requirements_path", value
                    ),
                    placeholder="docs/requirements.md",
                    width="100%",
                ),
            ),
            rx.hstack(
                rx.button(
                    "Fetch from GitHub", on_click=AppState.fetch_requirements_text
                ),
                _secondary_actions_menu(
                    "More",
                    [
                        (
                            "Copy repo to submit target",
                            AppState.sync_submit_repo_from_requirements,
                        ),
                    ],
                ),
                spacing="3",
            ),
            _status_block("Fetch status", AppState.requirements_fetch_status),
            _labeled_field(
                "Requirements text",
                rx.text_area(
                    value=AppState.requirements_text,
                    on_change=lambda value: AppState.set_workflow_field(
                        "requirements_text", value
                    ),
                    min_rows=12,
                    width="100%",
                ),
            ),
        ),
        _section_card(
            "Step 2 · Seed milestones and generate drafts",
            "Use the shared milestone queue to create the roadmap first, then parse requirements against the live repo milestones.",
            rx.hstack(
                rx.button(
                    rx.cond(
                        AppState.requirements_use_milestones,
                        "Milestone context enabled",
                        "Milestone context disabled",
                    ),
                    size="2",
                    variant=rx.cond(
                        AppState.requirements_use_milestones, "solid", "outline"
                    ),
                    color_scheme=rx.cond(
                        AppState.requirements_use_milestones, "indigo", "gray"
                    ),
                    on_click=AppState.toggle_workflow_flag(
                        "requirements_use_milestones"
                    ),
                ),
                rx.button(
                    "Parse requirements",
                    on_click=AppState.parse_requirements_workflow,
                    color_scheme="indigo",
                ),
                _secondary_actions_menu(
                    "More",
                    [
                        ("Load default milestones", AppState.load_default_milestones),
                        (
                            "Create queued milestones",
                            AppState.seed_requirements_milestones,
                        ),
                        (
                            "Use milestones for parsing",
                            AppState.use_milestones_for_requirements,
                        ),
                        ("Clear drafts", AppState.clear_requirement_drafts),
                    ],
                ),
                spacing="3",
            ),
            _labeled_field(
                "Milestone queue",
                rx.text_area(
                    value=AppState.milestone_queue_text,
                    on_change=lambda value: AppState.set_workflow_field(
                        "milestone_queue_text", value
                    ),
                    min_rows=6,
                    width="100%",
                ),
                "Shared with the Milestones page so you can keep one milestone plan across both workflows.",
            ),
            _labeled_field(
                "Milestones repo",
                rx.input(
                    value=AppState.requirements_milestones_repo,
                    on_change=lambda value: AppState.set_workflow_field(
                        "requirements_milestones_repo", value
                    ),
                    placeholder="owner/repo",
                    width="100%",
                ),
                "Leave blank to reuse the requirements repository.",
            ),
            _status_block(
                "Milestone seed status", AppState.requirements_milestone_status
            ),
            _status_block("Parse status", AppState.requirements_status),
        ),
        _section_card(
            "Step 3 · Review drafts",
            "Review and adjust drafts before publishing them to GitHub.",
            rx.cond(
                AppState.requirement_drafts.length() > 0,
                rx.vstack(
                    rx.foreach(AppState.requirement_drafts, _draft_editor),
                    spacing="4",
                    width="100%",
                ),
                rx.box(
                    rx.text(
                        "No drafts yet. Parse requirements to generate editable issue drafts.",
                        color_scheme="gray",
                    ),
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
                        on_change=lambda value: AppState.set_workflow_field(
                            "submit_repo", value
                        ),
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
                    on_change=lambda value: AppState.set_workflow_field(
                        "submit_milestone_override", value
                    ),
                    placeholder="Optional milestone number",
                    width="100%",
                ),
            ),
            rx.hstack(
                rx.button(
                    "Submit issues",
                    on_click=AppState.submit_requirement_drafts,
                    color_scheme="indigo",
                ),
                _secondary_actions_menu(
                    "More",
                    [
                        ("List open issues", AppState.list_submit_open_issues_workflow),
                    ],
                ),
                spacing="3",
            ),
            _status_block("Submit status", AppState.submit_status),
            _status_block("Open issues status", AppState.submit_open_issues_status),
            _markdown_block("Open issues", AppState.submit_open_issues_markdown),
        ),
    )


def _servicenow_page() -> rx.Component:
    return _page_shell(
        "ServiceNow sync",
        "Sync GitHub data to ServiceNow using the same workspace credentials.",
        _section_card(
            "Sync controls",
            "Use the shared ServiceNow settings and choose whether to preview or apply the sync.",
            rx.hstack(
                _labeled_field(
                    "Repository",
                    rx.input(
                        value=AppState.sync_repo,
                        on_change=lambda value: AppState.set_workflow_field(
                            "sync_repo", value
                        ),
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
                    rx.cond(
                        AppState.sync_dry_run, "Dry run enabled", "Apply mode enabled"
                    ),
                    size="2",
                    variant=rx.cond(AppState.sync_dry_run, "solid", "outline"),
                    color_scheme=rx.cond(AppState.sync_dry_run, "indigo", "gray"),
                    on_click=AppState.toggle_workflow_flag("sync_dry_run"),
                ),
                rx.button(
                    "Run sync",
                    on_click=AppState.run_servicenow_sync,
                    color_scheme="indigo",
                ),
                spacing="3",
            ),
            _labeled_field(
                "Back-sync allowlist",
                rx.input(
                    value=AppState.sync_back_sync_fields,
                    on_change=lambda value: AppState.set_workflow_field(
                        "sync_back_sync_fields", value
                    ),
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
        "Plan dependency-aware sprints, review board status, and apply approved updates.",
        _section_card(
            "Planning inputs",
            "Point the planner at a repository or owner and tune the sprint configuration.",
            rx.hstack(
                _labeled_field(
                    "Repository or owner",
                    rx.input(
                        value=AppState.agile_repo,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_repo", value
                        ),
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
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_capacity", value
                        ),
                        placeholder="10",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Number of sprints",
                    rx.input(
                        value=AppState.agile_sprints,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_sprints", value
                        ),
                        placeholder="3",
                        width="100%",
                    ),
                ),
                width="100%",
            ),
            rx.hstack(
                rx.button(
                    "Generate sprint plan",
                    on_click=AppState.run_agile_workflow,
                    color_scheme="indigo",
                ),
                _secondary_actions_menu(
                    "More",
                    [
                        (
                            "Apply relationships",
                            AppState.apply_agile_relationships_workflow,
                        ),
                        ("Apply labels", AppState.apply_agile_labels_workflow),
                    ],
                ),
                spacing="3",
            ),
            _status_block("Planning status", AppState.agile_status),
            _status_block("Apply status", AppState.agile_apply_status),
        ),
        _section_card(
            "Project status board",
            "Read project statuses for sprint issues and update item status during the sprint.",
            rx.hstack(
                _labeled_field(
                    "Project number",
                    rx.input(
                        value=AppState.agile_project_number,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_project_number", value
                        ),
                        placeholder="12",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Status field",
                    rx.input(
                        value=AppState.agile_project_status_field,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_project_status_field", value
                        ),
                        placeholder="Status",
                        width="100%",
                    ),
                ),
                _labeled_field(
                    "Sprint # filter (optional)",
                    rx.input(
                        value=AppState.agile_project_sprint_number,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_project_sprint_number", value
                        ),
                        placeholder="1",
                        width="100%",
                    ),
                ),
                width="100%",
            ),
            rx.hstack(
                rx.button(
                    "Read board",
                    on_click=AppState.read_agile_project_board_workflow,
                    color_scheme="indigo",
                ),
                _labeled_field(
                    "Issue/PR #",
                    rx.input(
                        value=AppState.agile_project_issue_number,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_project_issue_number", value
                        ),
                        placeholder="123",
                        width="160px",
                    ),
                ),
                _labeled_field(
                    "New status",
                    rx.input(
                        value=AppState.agile_project_status_value,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_project_status_value", value
                        ),
                        placeholder="In Progress",
                        width="220px",
                    ),
                ),
                rx.button(
                    "Update status",
                    on_click=AppState.update_agile_project_status_workflow,
                    color_scheme="indigo",
                    variant="soft",
                ),
                spacing="3",
                align_items="end",
                width="100%",
            ),
            _status_block("Project board status", AppState.agile_project_board_status),
        ),
        _section_card(
            "Repo and project context",
            "List repos and projects, create a project board if needed, and review open issues for sprint execution.",
            rx.hstack(
                _secondary_actions_menu(
                    "Context actions",
                    [
                        (
                            "List owner repositories",
                            AppState.list_agile_repositories_workflow,
                        ),
                        ("List projects", AppState.list_agile_projects_workflow),
                    ],
                ),
                spacing="3",
            ),
            rx.hstack(
                _labeled_field(
                    "New project title",
                    rx.input(
                        value=AppState.agile_new_project_title,
                        on_change=lambda value: AppState.set_workflow_field(
                            "agile_new_project_title", value
                        ),
                        placeholder="Sprint Board",
                        width="100%",
                    ),
                ),
                rx.button(
                    "Create project",
                    on_click=AppState.create_agile_project_workflow,
                    color_scheme="indigo",
                    variant="soft",
                ),
                width="100%",
                align_items="end",
            ),
            rx.button(
                "List open issues for repo",
                on_click=AppState.list_agile_open_issues_workflow,
                variant="soft",
            ),
            _status_block("Context status", AppState.agile_context_status),
            _status_block("Open issues status", AppState.agile_open_issues_status),
        ),
        _markdown_block("Repositories", AppState.agile_repos_markdown),
        _markdown_block("Projects", AppState.agile_projects_markdown),
        _markdown_block("Open issues", AppState.agile_open_issues_markdown),
        _markdown_block("Dependency graph", AppState.agile_dependencies_markdown),
        _markdown_block("Sprint plan", AppState.agile_plan_markdown),
        _markdown_block("Project board", AppState.agile_project_board_markdown),
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
app._api.add_route("/auth/github/login", start_github_oauth_login, methods=["GET"])
app._api.add_route("/auth/github/callback", handle_github_oauth_callback, methods=["GET"])
app._api.add_route("/auth/github/logout", logout_session, methods=["GET"])


def main() -> None:  # pragma: no cover
    """Launch the Reflex app via ``git-review-agent``."""
    app_dir = os.path.dirname(os.path.dirname(__file__))
    print(f"Launching agent app in {app_dir}...")
    subprocess.run(
        [sys.executable, "-m", "reflex", "run"],
        cwd=app_dir,
        check=True,
    )
