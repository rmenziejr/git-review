"""Reflex web application – git-review conversational agent.

Launch
------
::

    # via the installed script:
    git-review-agent

    # or directly (from the git_review/agent_app/ directory):
    reflex run

Settings can be configured via environment variables or a ``.env`` file.
See :class:`~git_review.config.AppSettings` for all available settings.

Environment variables
---------------------
GITHUB_TOKEN
    GitHub personal access token with repo write access.
OPENAI_API_KEY
    OpenAI API key.
OPENAI_BASE_URL
    Custom OpenAI-compatible base URL (optional).
AGENT_MODEL
    LLM model for the agent (default: gpt-4o).
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
from .state import AppState


# ---------------------------------------------------------------------------
# Page layout
# ---------------------------------------------------------------------------


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
                # Submit on Enter (without Shift)
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


def _header() -> rx.Component:
    return rx.hstack(
        rx.hstack(
            rx.icon("git-branch", size=20, color=rx.color("indigo", 10)),
            rx.heading("git-review agent", size="4"),
            spacing="2",
            align_items="center",
        ),
        rx.spacer(),
        rx.hstack(
            rx.button(
                rx.icon("trash-2", size=14),
                "Clear",
                size="2",
                variant="ghost",
                color_scheme="gray",
                on_click=AppState.clear_chat,
            ),
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
        padding_x="4",
        padding_y="3",
        border_bottom=f"1px solid {rx.color('gray', 4)}",
        background_color=rx.color("gray", 1),
        position="sticky",
        top="0",
        z_index="10",
    )


def index() -> rx.Component:
    """Main page component."""
    return rx.box(
        _header(),
        rx.box(
            # Main content area – fills viewport height minus header
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
                    padding_x="4",
                    padding_y="3",
                    border_top=f"1px solid {rx.color('gray', 4)}",
                    background_color=rx.color("gray", 1),
                ),
                spacing="0",
                height="100%",
                width="100%",
            ),
            height="calc(100vh - 57px)",
            display="flex",
            flex_direction="column",
            overflow="hidden",
        ),
        settings_panel(),
        width="100%",
        max_width="900px",
        margin="0 auto",
        height="100vh",
        display="flex",
        flex_direction="column",
        font_family="Inter, system-ui, sans-serif",
    )


# ---------------------------------------------------------------------------
# Reflex app
# ---------------------------------------------------------------------------

app = rx.App(
    theme=rx.theme(
        appearance="light",
        accent_color="indigo",
        radius="medium",
    ),
)
app.add_page(index, route="/", on_load=AppState.on_load)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:  # pragma: no cover
    """Launch the Reflex agent app via ``git-review-agent``."""
    app_dir = os.path.dirname(__file__)
    subprocess.run(
        [sys.executable, "-m", "reflex", "run"],
        cwd=app_dir,
        check=True,
    )
