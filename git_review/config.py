"""Application-level settings for git-review.

Settings are resolved in this priority order (highest first):

1. Values set directly on the :class:`AppSettings` instance.
2. Environment variables.
3. A ``.env`` file in the current working directory (if it exists).
4. Field defaults declared in the model.

Usage
-----
::

    from git_review.config import AppSettings

    settings = AppSettings()             # reads .env + environment
    print(settings.git_review_model)     # "gpt-4o-mini" (or override)

You can also construct with explicit overrides (useful in tests)::

    settings = AppSettings(git_review_model="gpt-4o", github_token="ghp_...")
"""

from __future__ import annotations

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """Centralised configuration for git-review.

    All fields can be overridden via environment variables (case-insensitive)
    or a ``.env`` file.  The mapping between field names and environment
    variable names is:

    ======================  =======================
    Field                   Environment variable
    ======================  =======================
    ``github_token``        ``GITHUB_TOKEN``
    ``openai_api_key``      ``OPENAI_API_KEY``
    ``git_review_model``    ``GIT_REVIEW_MODEL``
    ``openai_base_url``     ``OPENAI_BASE_URL``
    ``agent_model``         ``AGENT_MODEL``
    ``github_oauth_client_id`` ``GITHUB_OAUTH_CLIENT_ID``
    ``github_oauth_client_secret`` ``GITHUB_OAUTH_CLIENT_SECRET``
    ``github_oauth_scopes`` ``GITHUB_OAUTH_SCOPES``
    ``github_oauth_authorize_url`` ``GITHUB_OAUTH_AUTHORIZE_URL``
    ``github_oauth_token_url`` ``GITHUB_OAUTH_TOKEN_URL``
    ``github_oauth_callback_path`` ``GITHUB_OAUTH_CALLBACK_PATH``
    ``agent_session_cookie_name`` ``AGENT_SESSION_COOKIE_NAME``
    ``agent_session_ttl_seconds`` ``AGENT_SESSION_TTL_SECONDS``
    ``agent_cookie_secure`` ``AGENT_COOKIE_SECURE``
    ``agent_cookie_samesite`` ``AGENT_COOKIE_SAMESITE``
    ``agent_user_settings_path`` ``AGENT_USER_SETTINGS_PATH``
    ``servicenow_enabled``  ``SERVICENOW_ENABLED``
    ``servicenow_url``      ``SERVICENOW_URL``
    ``servicenow_user``     ``SERVICENOW_USER``
    ``servicenow_password`` ``SERVICENOW_PASSWORD``
    ``servicenow_token``    ``SERVICENOW_TOKEN``
    ``servicenow_milestone_table`` ``SERVICENOW_MILESTONE_TABLE``
    ``servicenow_issue_table`` ``SERVICENOW_ISSUE_TABLE``
    ``servicenow_cursor_path`` ``SERVICENOW_CURSOR_PATH``
    ``default_milestones_json`` ``DEFAULT_MILESTONES_JSON``
    ``gradio_server_name``  ``GRADIO_SERVER_NAME``
    ``gradio_server_port``  ``GRADIO_SERVER_PORT``
    ======================  =======================
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
        populate_by_name=True,
    )

    github_token: str = Field(
        default="",
        description="GitHub personal access token (PAT) with repo write access.",
    )
    openai_api_key: str = Field(
        default="",
        description="OpenAI API key used for LLM calls.",
    )
    git_review_model: str = Field(
        default="gpt-4o-mini",
        description="LLM model identifier (e.g. 'gpt-4o', 'gpt-4o-mini').",
    )
    openai_base_url: str = Field(
        default="",
        description=(
            "Custom OpenAI-compatible API base URL "
            "(e.g. 'http://localhost:11434/v1' for Ollama)."
        ),
    )
    agent_model: str = Field(
        default="gpt-4o",
        description=(
            "LLM model identifier used by the conversational agent "
            "(e.g. 'gpt-4o', 'gpt-4o-mini')."
        ),
    )
    github_oauth_client_id: str = Field(
        default="",
        description="GitHub OAuth App client ID used for browser sign-in.",
    )
    github_oauth_client_secret: str = Field(
        default="",
        description="GitHub OAuth App client secret used for token exchange.",
    )
    github_oauth_scopes: str = Field(
        default="repo,read:user,read:org",
        description="Comma-separated GitHub OAuth scopes requested during sign-in.",
    )
    github_oauth_authorize_url: str = Field(
        default="https://github.com/login/oauth/authorize",
        description="GitHub OAuth authorize endpoint.",
    )
    github_oauth_token_url: str = Field(
        default="https://github.com/login/oauth/access_token",
        description="GitHub OAuth token endpoint.",
    )
    github_oauth_callback_path: str = Field(
        default="/auth/github/callback",
        description="Callback path used by the OAuth flow.",
    )
    agent_session_cookie_name: str = Field(
        default="git_review_session",
        description="Cookie name used to track authenticated agent sessions.",
    )
    agent_session_ttl_seconds: int = Field(
        default=28_800,
        description="Agent session lifetime in seconds.",
    )
    agent_cookie_secure: bool = Field(
        default=True,
        description="Whether to mark the session cookie as Secure.",
    )
    agent_cookie_samesite: str = Field(
        default="lax",
        description="SameSite policy for the session cookie (lax/strict/none).",
    )
    agent_user_settings_path: str = Field(
        default=".git-review-agent-user-settings.json",
        description="Path to persisted per-user model settings.",
    )
    servicenow_enabled: bool = Field(
        default=False,
        description="Enable ServiceNow integration in UIs and agent tools.",
    )
    servicenow_url: str = Field(
        default="",
        description="ServiceNow instance URL (e.g. https://example.service-now.com).",
    )
    servicenow_user: str = Field(
        default="",
        description="ServiceNow username (when not using token auth).",
    )
    servicenow_password: str = Field(
        default="",
        description="ServiceNow password (when not using token auth).",
    )
    servicenow_token: str = Field(
        default="",
        description="ServiceNow bearer token.",
    )
    servicenow_milestone_table: str = Field(
        default="u_github_milestone",
        description="ServiceNow table for milestone sync records.",
    )
    servicenow_issue_table: str = Field(
        default="u_github_issue",
        description="ServiceNow table for issue/task sync records.",
    )
    servicenow_cursor_path: str = Field(
        default=".git-review-sync-cursor.json",
        description="File path for incremental GitHub→ServiceNow cursor storage.",
    )
    default_milestones_json: str = Field(
        default="",
        description=(
            "Optional JSON array of default milestone definitions for the web UIs. "
            "Each item should include title and may include due_on, state, and description."
        ),
    )
    gradio_server_name: str = Field(
        default="0.0.0.0",
        description=(
            "Hostname or IP address for the Gradio server to bind to "
            "(e.g. '127.0.0.1' or '0.0.0.0')."
        ),
    )
    gradio_server_port: int = Field(
        default=7860,
        description="TCP port for the Gradio server (e.g. 7860).",
    )
