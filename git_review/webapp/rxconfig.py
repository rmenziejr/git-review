"""Reflex configuration for the git-review agent app."""

import reflex as rx

config = rx.Config(
    app_name="agent_app",
    frontend_port=3000,
    backend_port=8000,
)
