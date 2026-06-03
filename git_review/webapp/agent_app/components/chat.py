"""Chat thread component – renders the conversation history and live stream."""

from __future__ import annotations

import reflex as rx

from ..state import AppState, ChatMessage, ToolEvent
from .tool_call import tool_call_card, tool_result_card


CHAT_LANE_MAX = "52rem"
NAVY_DARK = "#0f243a"
BLUE_LIGHT = "rgba(219, 235, 251, 0.55)"
SURFACE = "rgba(255, 255, 255, 0.98)"
BORDER = "rgba(22, 50, 79, 0.11)"
SHADOW = "0 8px 20px rgba(16, 35, 56, 0.08)"


# ---------------------------------------------------------------------------
# Individual message renderers
# ---------------------------------------------------------------------------


def _user_bubble(msg: ChatMessage) -> rx.Component:
    return rx.hstack(
        rx.spacer(),
        rx.box(
            rx.text(msg.content, size="2"),
            color=NAVY_DARK,
            background_color=SURFACE,
            border=f"1px solid {BORDER}",
            border_radius="12px",
            padding_x="4",
            padding_y="3",
            max_width="min(80%, 760px)",
            box_shadow=SHADOW,
        ),
        width="100%",
        max_width=CHAT_LANE_MAX,
        margin_inline="auto",
    )


def _reasoning_panel(text: str, thinking_complete: rx.Var | bool) -> rx.Component:
    return rx.cond(
        text != "",
        rx.accordion.root(
            rx.accordion.item(
                header=rx.hstack(
                    rx.hstack(
                        rx.cond(
                            thinking_complete,
                            rx.icon("brain", size=14, color=rx.color("amber", 10)),
                            rx.spinner(size="1", color="amber"),
                        ),
                        rx.text(
                            rx.cond(thinking_complete, "Thoughts", "Thinking"),
                            size="1",
                            weight="medium",
                            color=rx.color("amber", 11),
                        ),
                        spacing="2",
                        align_items="center",
                    ),
                    rx.cond(
                        AppState.processing_tool,
                        rx.text("Using tools", size="1", color_scheme="blue"),
                        rx.fragment(),
                    ),
                    spacing="2",
                    align_items="center",
                    justify="between",
                    width="100%",
                ),
                content=rx.box(
                    rx.text(
                        text,
                        size="1",
                        color_scheme="gray",
                        white_space="pre-wrap",
                        line_height="1.5",
                    ),
                    padding_top="2",
                ),
                value="reasoning",
            ),
            collapsible=True,
            width="100%",
            variant="ghost",
        ),
        rx.fragment(),
    )


def _assistant_bubble(msg: ChatMessage) -> rx.Component:
    has_content = (
        AppState.is_thinking
        | (msg.content != "")
        | (msg.reasoning_text != "")
        | (msg.tool_events.length() > 0)
    )

    return rx.cond(
        has_content,
        rx.box(
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.text(
                            "AI",
                            size="1",
                            weight="bold",
                            color=rx.color("blue", 10),
                        ),
                        rx.cond(
                            AppState.is_thinking & (msg.content == ""),
                            rx.spinner(size="1", color="blue"),
                            rx.fragment(),
                        ),
                        spacing="2",
                        align_items="center",
                    ),
                    _tools_panel(msg.tool_events),
                    _reasoning_panel(msg.reasoning_text, True),
                    rx.cond(
                        msg.content != "",
                        rx.markdown(msg.content),
                        rx.fragment(),
                    ),
                    spacing="3",
                    width="100%",
                    align_items="start",
                ),
                background_color=BLUE_LIGHT,
                border_radius="12px",
                padding_x="4",
                padding_y="3",
                max_width="min(84%, 860px)",
                border=f"1px solid {BORDER}",
                box_shadow=SHADOW,
            ),
            width="100%",
            max_width=CHAT_LANE_MAX,
            margin_inline="auto",
            text_align="left",
        ),
        rx.fragment(),
    )


def _tools_panel(events: list[ToolEvent]) -> rx.Component:
    return rx.cond(
        events.length() > 0,
        rx.accordion.root(
            rx.accordion.item(
                header=rx.hstack(
                    rx.icon("wrench", size=14, color=rx.color("indigo", 10)),
                    rx.text("Tools", size="1", weight="medium", color_scheme="indigo"),
                    rx.badge(events.length(), size="1", color_scheme="indigo", variant="soft"),
                    spacing="2",
                    align_items="center",
                    width="100%",
                ),
                content=rx.vstack(
                    rx.foreach(events, _tool_event_item),
                    spacing="2",
                    width="100%",
                    align_items="stretch",
                    padding_top="2",
                ),
                value="tools",
            ),
            collapsible=True,
            width="100%",
            variant="ghost",
        ),
        rx.fragment(),
    )


def _tool_event_item(event: ToolEvent) -> rx.Component:
    return rx.match(
        event.kind,
        ("tool_call", tool_call_card(event)),
        ("tool_result", tool_result_card(event)),
        ("tool_error", tool_result_card(event)),
        rx.fragment(),
    )


def _reasoning_bubble(msg: ChatMessage) -> rx.Component:
    return rx.accordion.root(
        rx.accordion.item(
            header=rx.hstack(
                rx.icon("brain", size=14, color=rx.color("violet", 10)),
                rx.text("Thinking", size="1", color_scheme="violet"),
                spacing="2",
                align_items="center",
            ),
            content=rx.box(
                rx.text(
                    msg.content,
                    size="1",
                    color_scheme="gray",
                    white_space="pre-wrap",
                ),
                padding_top="2",
            ),
            value="thinking",
        ),
        collapsible=True,
        width="100%",
        variant="ghost",
    )


def _error_bubble(msg: ChatMessage) -> rx.Component:
    return rx.box(
        rx.callout.root(
            rx.callout.icon(rx.icon("circle-alert", size=16)),
            rx.callout.text(msg.content, size="2"),
            color_scheme="red",
            width="100%",
        ),
        width="100%",
        max_width=CHAT_LANE_MAX,
        margin_inline="auto",
    )


def _message_item(msg: ChatMessage) -> rx.Component:
    return rx.match(
        msg.role,
        ("user", _user_bubble(msg)),
        ("assistant", _assistant_bubble(msg)),
        ("reasoning", _reasoning_bubble(msg)),
        ("error", _error_bubble(msg)),
        rx.fragment(),  # fallback
    )


# ---------------------------------------------------------------------------
# Thinking indicator
# ---------------------------------------------------------------------------


def _thinking_indicator() -> rx.Component:
    return rx.cond(
        AppState.is_thinking,
        rx.box(
            rx.box(
                rx.vstack(
                    rx.hstack(
                        rx.text(
                            "AI", size="1", weight="bold", color=rx.color("blue", 10)
                        ),
                        rx.box(
                            rx.hstack(
                                rx.box(
                                    width="8px",
                                    height="8px",
                                    border_radius="50%",
                                    background_color=rx.color("indigo", 8),
                                    animation="bounce 1s infinite",
                                ),
                                rx.box(
                                    width="8px",
                                    height="8px",
                                    border_radius="50%",
                                    background_color=rx.color("indigo", 8),
                                    animation="bounce 1s infinite 0.15s",
                                ),
                                rx.box(
                                    width="8px",
                                    height="8px",
                                    border_radius="50%",
                                    background_color=rx.color("indigo", 8),
                                    animation="bounce 1s infinite 0.3s",
                                ),
                                spacing="1",
                                align_items="center",
                            ),
                        ),
                        spacing="2",
                        align_items="center",
                    ),
                    _reasoning_panel(AppState.reasoning_text, False),
                    spacing="2",
                    width="100%",
                    align_items="start",
                ),
                width="100%",
                background=BLUE_LIGHT,
                border=f"1px solid {BORDER}",
                border_radius="12px",
                padding="0.75rem 0.95rem",
                style={"box-shadow": SHADOW},
                max_width="min(84%, 860px)",
            ),
            width="100%",
            max_width=CHAT_LANE_MAX,
            margin_inline="auto",
        ),
        rx.fragment(),
    )


def _streaming_bubble() -> rx.Component:
    return rx.cond(
        AppState.streaming_text != "",
        rx.box(
            rx.box(
                rx.vstack(
                    rx.text("AI", size="1", weight="bold", color=rx.color("blue", 10)),
                    rx.text(
                        AppState.streaming_text,
                        white_space="pre-wrap",
                        size="2",
                        color=NAVY_DARK,
                    ),
                    spacing="2",
                    width="100%",
                    align_items="start",
                ),
                background=BLUE_LIGHT,
                border_radius="12px",
                padding_x="4",
                padding_y="3",
                max_width="min(84%, 860px)",
                border=f"1px solid {BORDER}",
                style={"box-shadow": SHADOW},
            ),
            width="100%",
            max_width=CHAT_LANE_MAX,
            margin_inline="auto",
        ),
        rx.fragment(),
    )


# ---------------------------------------------------------------------------
# Full chat thread
# ---------------------------------------------------------------------------


def chat_thread() -> rx.Component:
    """Render the full scrollable chat conversation."""
    return rx.box(
        rx.vstack(
            rx.foreach(AppState.messages, _message_item),
            spacing="3",
            width="100%",
            padding_bottom="4",
            min_height="0",
            height="100%",
        ),
        overflow_y="auto",
        flex="1",
        width="100%",
        height="100%",
        min_height="0",
        padding_x="2",
        padding_y="3",
        id="chat-scroll",
    )
