"""Reflex configuration for the git-review agent app."""

import reflex as rx

from git_review.config import AppSettings

settings = AppSettings()

config = rx.Config(
    app_name="agent_app",
    frontend_port=3000,
    backend_port=8000,
    api_url=settings.agent_backend_url,
    deploy_url=settings.agent_frontend_url,
    vite_allowed_hosts=["*", "dev-aitools.wvumedicine.org"],
)
