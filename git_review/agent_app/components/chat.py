"""Chat thread component – renders the conversation history and live stream."""

from __future__ import annotations

import reflex as rx

from ..state import AppState, ChatMessage
from .tool_call import tool_call_card, tool_result_card


# ---------------------------------------------------------------------------
# Individual message renderers
# ---------------------------------------------------------------------------


def _user_bubble(msg: ChatMessage) -> rx.Component:
    return rx.hstack(
        rx.spacer(),
        rx.box(
            rx.text(msg.content, size="2"),
            background_color=rx.color("blue", 9),
            color="white",
            border_radius="12px 12px 2px 12px",
            padding="3",
            max_width="70%",
        ),
        width="100%",
    )


def _assistant_bubble(msg: ChatMessage) -> rx.Component:
    return rx.hstack(
        rx.box(
            rx.icon("bot", size=18, color=rx.color("indigo", 10)),
            padding="1",
            flex_shrink="0",
        ),
        rx.box(
            rx.markdown(msg.content),
            background_color=rx.color("gray", 2),
            border_radius="2px 12px 12px 12px",
            padding="3",
            max_width="80%",
            border=f"1px solid {rx.color('gray', 4)}",
        ),
        spacing="2",
        width="100%",
        align_items="start",
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
    return rx.callout.root(
        rx.callout.icon(rx.icon("circle-alert", size=16)),
        rx.callout.text(msg.content, size="2"),
        color_scheme="red",
        width="100%",
    )


def _message_item(msg: ChatMessage) -> rx.Component:
    return rx.match(
        msg.role,
        ("user", _user_bubble(msg)),
        ("assistant", _assistant_bubble(msg)),
        ("reasoning", _reasoning_bubble(msg)),
        ("tool_call", tool_call_card(msg)),
        ("tool_result", tool_result_card(msg)),
        ("error", _error_bubble(msg)),
        rx.fragment(),  # fallback
    )


# ---------------------------------------------------------------------------
# Thinking indicator
# ---------------------------------------------------------------------------


def _thinking_indicator() -> rx.Component:
    return rx.cond(
        AppState.is_thinking,
        rx.hstack(
            rx.box(
                rx.icon("bot", size=18, color=rx.color("indigo", 10)),
                flex_shrink="0",
            ),
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
            spacing="2",
            align_items="center",
        ),
        rx.fragment(),
    )


# ---------------------------------------------------------------------------
# Streaming text live bubble
# ---------------------------------------------------------------------------


def _streaming_bubble() -> rx.Component:
    return rx.cond(
        AppState.streaming_text != "",
        rx.hstack(
            rx.box(
                rx.icon("bot", size=18, color=rx.color("indigo", 10)),
                flex_shrink="0",
            ),
            rx.box(
                rx.markdown(AppState.streaming_text),
                background_color=rx.color("gray", 2),
                border_radius="2px 12px 12px 12px",
                padding="3",
                max_width="80%",
                border=f"1px solid {rx.color('indigo', 4)}",
            ),
            spacing="2",
            width="100%",
            align_items="start",
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
            _streaming_bubble(),
            _thinking_indicator(),
            spacing="3",
            width="100%",
            padding_bottom="4",
        ),
        overflow_y="auto",
        flex="1",
        width="100%",
        padding_x="4",
        padding_y="2",
        id="chat-scroll",
    )
