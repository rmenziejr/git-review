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

    ==================  ======================
    Field               Environment variable
    ==================  ======================
    ``github_token``    ``GITHUB_TOKEN``
    ``openai_api_key``  ``OPENAI_API_KEY``
    ``git_review_model`` ``GIT_REVIEW_MODEL``
    ``openai_base_url`` ``OPENAI_BASE_URL``
    ==================  ======================
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
