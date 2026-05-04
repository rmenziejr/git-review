"""Settings sidebar component for the git-review agent UI."""

from __future__ import annotations

import reflex as rx

from ..state import AppState


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


def settings_panel() -> rx.Component:
    """A slide-in settings panel rendered when ``AppState.settings_open`` is True."""
    return rx.cond(
        AppState.settings_open,
        rx.box(
            rx.vstack(
                rx.hstack(
                    rx.heading("Settings", size="4"),
                    rx.spacer(),
                    rx.icon_button(
                        rx.icon("x"),
                        variant="ghost",
                        on_click=AppState.toggle_settings,
                        size="2",
                    ),
                    width="100%",
                    align_items="center",
                ),
                rx.divider(),
                rx.text("Repository", size="2", weight="bold", color_scheme="blue"),
                _labeled_input(
                    "Owner / Org",
                    AppState.owner,
                    AppState.set_owner,
                    placeholder="e.g. myorg",
                ),
                _labeled_input(
                    "Repository",
                    AppState.repo,
                    AppState.set_repo,
                    placeholder="e.g. myrepo",
                ),
                rx.divider(),
                rx.text("Credentials", size="2", weight="bold", color_scheme="blue"),
                _labeled_input(
                    "GitHub Token",
                    AppState.github_token,
                    AppState.set_github_token,
                    placeholder="ghp_...",
                    password=True,
                ),
                _labeled_input(
                    "OpenAI API Key",
                    AppState.openai_key,
                    AppState.set_openai_key,
                    placeholder="sk-...",
                    password=True,
                ),
                _labeled_input(
                    "OpenAI Base URL",
                    AppState.openai_base_url,
                    AppState.set_openai_base_url,
                    placeholder="http://localhost:11434/v1",
                ),
                rx.divider(),
                rx.text("Model", size="2", weight="bold", color_scheme="blue"),
                _labeled_input(
                    "Agent Model",
                    AppState.agent_model,
                    AppState.set_agent_model,
                    placeholder="gpt-4o",
                ),
                spacing="3",
                padding="4",
                width="320px",
            ),
            position="fixed",
            top="0",
            right="0",
            height="100vh",
            background_color=rx.color("gray", 1),
            border_left=f"1px solid {rx.color('gray', 4)}",
            overflow_y="auto",
            z_index="100",
            box_shadow="-4px 0 16px rgba(0,0,0,0.12)",
        ),
        rx.fragment(),
    )
