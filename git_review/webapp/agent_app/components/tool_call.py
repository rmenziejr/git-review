"""Tool-call card component – a collapsible card showing tool name, args, and result."""

from __future__ import annotations

import reflex as rx

from ..state import ToolEvent


def _event_shell(*children: rx.Component, color: str) -> rx.Component:
    return rx.box(
        *children,
        width="100%",
        border=f"1px solid {rx.color(color, 5)}",
        border_radius="12px",
        background_color=rx.color(color, 2),
        padding_x="3",
        padding_y="3",
        box_shadow="0 8px 20px rgba(16, 35, 56, 0.07)",
    )


def tool_call_card(event: ToolEvent) -> rx.Component:
    """Render a collapsible tool-call card."""
    return _event_shell(
        rx.accordion.root(
            rx.accordion.item(
                header=rx.hstack(
                    rx.icon("wrench", size=14, color=rx.color("indigo", 10)),
                    rx.text(
                        "Tool call", size="1", weight="medium", color_scheme="indigo"
                    ),
                    rx.code(event.tool_name, size="1", color=rx.color("indigo", 11)),
                    spacing="2",
                    align_items="center",
                    width="100%",
                ),
                content=rx.box(
                    rx.code_block(
                        rx.cond(event.args_json != "", event.args_json, "{}"),
                        language="json",
                        font_size="0.75rem",
                        width="100%",
                    ),
                    padding_top="2",
                ),
                value=event.id,
            ),
            collapsible=True,
            width="100%",
            variant="ghost",
        ),
        color="indigo",
    )


def tool_result_card(event: ToolEvent) -> rx.Component:
    """Render a tool result card with optional error state."""
    return rx.cond(
        event.is_error,
        _event_shell(
            rx.accordion.root(
                rx.accordion.item(
                    header=rx.hstack(
                        rx.icon("triangle-alert", size=14, color=rx.color("red", 10)),
                        rx.text(
                            "Tool error", size="1", weight="medium", color_scheme="red"
                        ),
                        rx.code(event.tool_name, size="1", color=rx.color("red", 11)),
                        spacing="2",
                        align_items="center",
                        width="100%",
                    ),
                    content=rx.box(
                        rx.code_block(
                            event.content,
                            language="json",
                            font_size="0.75rem",
                            width="100%",
                        ),
                        padding_top="2",
                    ),
                    value=event.id,
                ),
                collapsible=True,
                width="100%",
                variant="ghost",
            ),
            color="red",
        ),
        _event_shell(
            rx.accordion.root(
                rx.accordion.item(
                    header=rx.hstack(
                        rx.icon("circle-check", size=14, color=rx.color("green", 10)),
                        rx.text(
                            "Tool result",
                            size="1",
                            weight="medium",
                            color_scheme="green",
                        ),
                        rx.code(event.tool_name, size="1", color=rx.color("green", 11)),
                        spacing="2",
                        align_items="center",
                        width="100%",
                    ),
                    content=rx.box(
                        rx.code_block(
                            event.content,
                            language="json",
                            font_size="0.75rem",
                            width="100%",
                        ),
                        padding_top="2",
                    ),
                    value=event.id,
                ),
                collapsible=True,
                width="100%",
                variant="ghost",
            ),
            color="green",
        ),
    )
