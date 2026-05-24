"""Tests for AppSettings configuration model."""

from __future__ import annotations

import os

import pytest

from git_review.config import AppSettings


def test_defaults_when_no_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("GIT_REVIEW_MODEL", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("SERVICENOW_ENABLED", raising=False)
    monkeypatch.delenv("SERVICENOW_URL", raising=False)
    monkeypatch.delenv("SERVICENOW_TOKEN", raising=False)
    monkeypatch.delenv("DEFAULT_MILESTONES_JSON", raising=False)
    monkeypatch.delenv("GITHUB_OAUTH_CLIENT_ID", raising=False)
    monkeypatch.delenv("GITHUB_OAUTH_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("GITHUB_OAUTH_SCOPES", raising=False)
    monkeypatch.delenv("AGENT_SESSION_TTL_SECONDS", raising=False)

    settings = AppSettings()

    assert settings.github_token == ""
    assert settings.openai_api_key == ""
    assert settings.git_review_model == "gpt-4o-mini"
    assert settings.openai_base_url == ""
    assert settings.servicenow_enabled is False
    assert settings.servicenow_url == ""
    assert settings.servicenow_token == ""
    assert settings.default_milestones_json == ""
    assert settings.github_oauth_client_id == ""
    assert settings.github_oauth_client_secret == ""
    assert settings.github_oauth_scopes == "repo,read:user,read:org"
    assert settings.agent_session_ttl_seconds == 28_800


def test_reads_from_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test_token")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-key")
    monkeypatch.setenv("GIT_REVIEW_MODEL", "gpt-4o")
    monkeypatch.setenv("OPENAI_BASE_URL", "http://localhost:11434/v1")
    monkeypatch.setenv("SERVICENOW_ENABLED", "true")
    monkeypatch.setenv("SERVICENOW_URL", "https://example.service-now.com")
    monkeypatch.setenv("SERVICENOW_TOKEN", "sn-token")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_ID", "oauth-client-id")
    monkeypatch.setenv("GITHUB_OAUTH_CLIENT_SECRET", "oauth-client-secret")
    monkeypatch.setenv("GITHUB_OAUTH_SCOPES", "repo,read:user")
    monkeypatch.setenv("AGENT_SESSION_TTL_SECONDS", "1200")
    monkeypatch.setenv(
        "DEFAULT_MILESTONES_JSON",
        '[{"title":"Backlog","description":"Shared roadmap"}]',
    )

    settings = AppSettings()

    assert settings.github_token == "ghp_test_token"
    assert settings.openai_api_key == "sk-test-key"
    assert settings.git_review_model == "gpt-4o"
    assert settings.openai_base_url == "http://localhost:11434/v1"
    assert settings.servicenow_enabled is True
    assert settings.servicenow_url == "https://example.service-now.com"
    assert settings.servicenow_token == "sn-token"
    assert settings.default_milestones_json == '[{"title":"Backlog","description":"Shared roadmap"}]'
    assert settings.github_oauth_client_id == "oauth-client-id"
    assert settings.github_oauth_client_secret == "oauth-client-secret"
    assert settings.github_oauth_scopes == "repo,read:user"
    assert settings.agent_session_ttl_seconds == 1200


def test_explicit_construction_overrides_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GIT_REVIEW_MODEL", "gpt-4o")

    settings = AppSettings(git_review_model="gpt-4o-mini")

    assert settings.git_review_model == "gpt-4o-mini"


def test_partial_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GIT_REVIEW_MODEL", raising=False)
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_partial")

    settings = AppSettings()

    assert settings.github_token == "ghp_partial"
    assert settings.git_review_model == "gpt-4o-mini"  # default still applies
