"""Settings sidebar component for the git-review agent UI."""

from __future__ import annotations

import reflex as rx

from ..state import AppState


PANEL_BG = "rgba(251, 253, 255, 0.98)"
PANEL_BORDER = "1px solid rgba(20, 53, 89, 0.12)"
PANEL_SHADOW = "-10px 0 28px rgba(16, 35, 56, 0.14)"
SECTION_BG = "rgba(255, 255, 255, 0.96)"
SECTION_BORDER = "1px solid rgba(20, 53, 89, 0.09)"


def _labeled_input(
    label: str,
    value: rx.Var,
    on_change,
    placeholder: str = "",
    password: bool = False,
) -> rx.Component:
    return rx.vstack(
        rx.text(label, size="1", color_scheme="gray", weight="medium"),
        rx.input(
            value=value,
            on_change=on_change,
            placeholder=placeholder,
            type="password" if password else "text",
            size="2",
            width="100%",
        ),
        spacing="1",
        width="100%",
        align_items="start",
    )


def _settings_section(
    title: str,
    section_key: str,
    is_open: rx.Var,
    *children: rx.Component,
) -> rx.Component:
    return rx.box(
        rx.vstack(
            rx.button(
                rx.hstack(
                    rx.text(title, size="2", weight="bold", color=rx.color("blue", 11)),
                    rx.spacer(),
                    rx.icon(rx.cond(is_open, "chevron-up", "chevron-down"), size=15),
                    align_items="center",
                    width="100%",
                    spacing="2",
                ),
                on_click=AppState.toggle_settings_section(section_key),
                variant="ghost",
                color_scheme="gray",
                width="100%",
                justify="start",
                padding="0.2rem 0.1rem",
            ),
            rx.cond(
                is_open,
                rx.vstack(
                    *children,
                    spacing="2",
                    width="100%",
                    align_items="start",
                ),
                rx.fragment(),
            ),
            spacing="2",
            width="100%",
            align_items="start",
        ),
        width="100%",
        padding="0.8rem 0.85rem",
        border_radius="12px",
        background=SECTION_BG,
        border=SECTION_BORDER,
    )


def settings_panel() -> rx.Component:
    """A slide-in settings panel rendered when ``AppState.settings_open`` is True."""
    return rx.cond(
        AppState.settings_open,
        rx.fragment(
            rx.box(
                position="fixed",
                inset="0",
                background="rgba(10, 24, 40, 0.24)",
                z_index="99",
                on_click=AppState.toggle_settings,
            ),
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.vstack(
                            rx.heading("Settings", size="4"),
                            rx.text(
                                "Manage default repository, credentials, and integrations.",
                                size="1",
                                color_scheme="gray",
                            ),
                            spacing="1",
                            align_items="start",
                        ),
                        rx.spacer(),
                        rx.icon_button(
                            rx.icon("x"),
                            variant="ghost",
                            on_click=AppState.toggle_settings,
                            size="2",
                        ),
                        width="100%",
                        align_items="start",
                    ),
                    _settings_section(
                        "Repository",
                        "repo",
                        AppState.settings_repo_open,
                        _labeled_input(
                            "Owner / organization",
                            AppState.owner,
                            AppState.set_owner,
                            placeholder="e.g. my-org",
                        ),
                        _labeled_input(
                            "Default repository",
                            AppState.repo,
                            AppState.set_repo,
                            placeholder="e.g. platform-api",
                        ),
                    ),
                    _settings_section(
                        "Credentials",
                        "credentials",
                        AppState.settings_credentials_open,
                        _labeled_input(
                            "GitHub token",
                            AppState.github_token,
                            AppState.set_github_token,
                            placeholder="ghp_...",
                            password=True,
                        ),
                        _labeled_input(
                            "OpenAI API key",
                            AppState.openai_key,
                            AppState.set_openai_key,
                            placeholder="sk-...",
                            password=True,
                        ),
                        _labeled_input(
                            "OpenAI base URL",
                            AppState.openai_base_url,
                            AppState.set_openai_base_url,
                            placeholder="http://localhost:11434/v1",
                        ),
                    ),
                    _settings_section(
                        "Model",
                        "model",
                        AppState.settings_model_open,
                        _labeled_input(
                            "Agent model",
                            AppState.agent_model,
                            AppState.set_agent_model,
                            placeholder="gpt-4o",
                        ),
                    ),
                    _settings_section(
                        "ServiceNow",
                        "servicenow",
                        AppState.settings_servicenow_open,
                        rx.hstack(
                            rx.text("Integration", size="2", color_scheme="gray"),
                            rx.spacer(),
                            rx.button(
                                rx.cond(
                                    AppState.servicenow_enabled, "Enabled", "Disabled"
                                ),
                                size="1",
                                variant=rx.cond(
                                    AppState.servicenow_enabled, "solid", "outline"
                                ),
                                color_scheme=rx.cond(
                                    AppState.servicenow_enabled, "green", "gray"
                                ),
                                on_click=AppState.toggle_servicenow_enabled,
                            ),
                            width="100%",
                            align_items="center",
                        ),
                        rx.cond(
                            AppState.servicenow_enabled,
                            rx.vstack(
                                _labeled_input(
                                    "ServiceNow URL",
                                    AppState.servicenow_url,
                                    AppState.set_servicenow_url,
                                    placeholder="https://example.service-now.com",
                                ),
                                _labeled_input(
                                    "ServiceNow token",
                                    AppState.servicenow_token,
                                    AppState.set_servicenow_token,
                                    placeholder="Bearer token",
                                    password=True,
                                ),
                                _labeled_input(
                                    "ServiceNow user (optional)",
                                    AppState.servicenow_user,
                                    AppState.set_servicenow_user,
                                    placeholder="username",
                                ),
                                _labeled_input(
                                    "ServiceNow password (optional)",
                                    AppState.servicenow_password,
                                    AppState.set_servicenow_password,
                                    placeholder="password",
                                    password=True,
                                ),
                                _labeled_input(
                                    "Milestone table",
                                    AppState.servicenow_milestone_table,
                                    AppState.set_servicenow_milestone_table,
                                    placeholder="u_github_milestone",
                                ),
                                _labeled_input(
                                    "Issue table",
                                    AppState.servicenow_issue_table,
                                    AppState.set_servicenow_issue_table,
                                    placeholder="u_github_issue",
                                ),
                                _labeled_input(
                                    "Cursor file path",
                                    AppState.servicenow_cursor_path,
                                    AppState.set_servicenow_cursor_path,
                                    placeholder=".git-review-sync-cursor.json",
                                ),
                                spacing="2",
                                width="100%",
                                align_items="start",
                            ),
                            rx.text(
                                "Enable integration to configure ServiceNow connection values.",
                                size="1",
                                color_scheme="gray",
                            ),
                        ),
                    ),
                    spacing="3",
                    padding="4",
                    width="100%",
                    align_items="start",
                ),
                position="fixed",
                top="0",
                right="0",
                height="100vh",
                width=rx.breakpoints(initial="100vw", md="360px"),
                background=PANEL_BG,
                border_left=PANEL_BORDER,
                overflow_y="auto",
                z_index="100",
                box_shadow=PANEL_SHADOW,
                backdrop_filter="blur(8px)",
            ),
        ),
        rx.fragment(),
    )
