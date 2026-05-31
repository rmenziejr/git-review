"""HITL (human-in-the-loop) approval panel component."""

from __future__ import annotations

import reflex as rx

from ..state import AppState, HITLRequest


def _single_approval(item: HITLRequest) -> rx.Component:
    """Render approve / deny controls for a single pending HITL item."""
    return rx.card(
        rx.vstack(
            rx.hstack(
                rx.icon("shield-alert", size=16, color=rx.color("amber", 10)),
                rx.text(item.description, white_space="pre-wrap", size="2"),
                spacing="2",
                align_items="center",
            ),
            rx.text("Requested action:", size="1", color_scheme="gray"),
            rx.hstack(
                rx.badge(item.tool_name, color_scheme="amber", variant="soft"),
                spacing="2",
            ),
            rx.accordion.root(
                rx.accordion.item(
                    header=rx.text("Arguments", size="1", color_scheme="gray"),
                    content=rx.code_block(
                        item.args_json,
                        language="json",
                        font_size="0.7rem",
                        width="100%",
                    ),
                    value="args",
                ),
                collapsible=True,
                variant="ghost",
                width="100%",
            ),
            rx.hstack(
                rx.button(
                    rx.icon("check", size=14),
                    "Approve",
                    color_scheme="green",
                    size="2",
                    on_click=AppState.approve_hitl(item.id),
                ),
                rx.button(
                    rx.icon("x", size=14),
                    "Deny",
                    color_scheme="red",
                    variant="soft",
                    size="2",
                    on_click=AppState.deny_hitl(item.id),
                ),
                spacing="3",
            ),
            spacing="3",
            align_items="start",
            width="100%",
        ),
        width="100%",
        border=f"1px solid {rx.color('amber', 6)}",
        border_radius="12px",
        background_color=rx.color("amber", 2),
        box_shadow="0 8px 22px rgba(137, 83, 14, 0.10)",
        padding="0.2rem",
    )


def hitl_panel() -> rx.Component:
    """Render the HITL approval panel when there are pending requests."""
    return rx.cond(
        AppState.pending_hitl.length() > 0,
        rx.vstack(
            rx.hstack(
                rx.icon("shield-alert", size=18, color=rx.color("amber", 10)),
                rx.text(
                    "Approval required before continuing",
                    size="3",
                    weight="bold",
                    color_scheme="amber",
                ),
                spacing="2",
                align_items="center",
            ),
            rx.foreach(AppState.pending_hitl, _single_approval),
            spacing="4",
            width="100%",
            padding="4",
            border_radius="12px",
            border=f"2px solid {rx.color('amber', 6)}",
            background_color=rx.color("amber", 2),
        ),
        rx.fragment(),
    )
