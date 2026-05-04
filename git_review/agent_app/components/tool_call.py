"""Tool-call card component – a collapsible card showing tool name, args, and result."""

from __future__ import annotations

import reflex as rx

from ..state import ChatMessage


def _badge(text: str, color: str) -> rx.Component:
    return rx.badge(text, color_scheme=color, size="1", variant="soft")


def tool_call_card(msg: ChatMessage) -> rx.Component:
    """Render a collapsible tool-call card."""
    return rx.accordion.root(
        rx.accordion.item(
            header=rx.hstack(
                rx.icon("wrench", size=14, color=rx.color("indigo", 10)),
                _badge("tool call", "indigo"),
                rx.code(msg.tool_name, size="1"),
                spacing="2",
                align_items="center",
            ),
            content=rx.box(
                rx.code_block(
                    msg.args_json,
                    language="json",
                    font_size="0.75rem",
                    width="100%",
                ),
                padding_top="2",
            ),
            value="tool",
        ),
        collapsible=True,
        width="100%",
        variant="ghost",
    )


def tool_result_card(msg: ChatMessage) -> rx.Component:
    """Render a tool result card."""
    return rx.accordion.root(
        rx.accordion.item(
            header=rx.hstack(
                rx.icon("check-circle", size=14, color=rx.color("green", 10)),
                _badge("result", "green"),
                rx.code(msg.tool_name, size="1"),
                spacing="2",
                align_items="center",
            ),
            content=rx.box(
                rx.code_block(
                    msg.content,
                    language="json",
                    font_size="0.75rem",
                    width="100%",
                ),
                padding_top="2",
            ),
            value="result",
        ),
        collapsible=True,
        width="100%",
        variant="ghost",
    )
