"""Tests for Reflex GitHub OAuth/session helpers."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from urllib.parse import parse_qs, urlparse

from starlette.applications import Starlette
from starlette.testclient import TestClient

from git_review.agent_app import auth
from git_review.agent_app.state import AppState
from git_review.config import AppSettings


def _settings(**overrides: object) -> AppSettings:
    base = {
        "github_oauth_client_id": "client-id",
        "github_oauth_client_secret": "client-secret",
        "agent_cookie_secure": False,
    }
    base.update(overrides)
    return AppSettings(**base)


def test_github_login_route_sets_state_cookie() -> None:
    settings = _settings()
    app = Starlette()
    app.add_route("/auth/github/login", lambda request: auth.start_github_oauth_login(request, settings), methods=["GET"])
    client = TestClient(app)

    response = client.get("/auth/github/login", follow_redirects=False)

    assert response.status_code == 302
    location = response.headers["location"]
    parsed = urlparse(location)
    params = parse_qs(parsed.query)
    assert parsed.netloc == "github.com"
    assert params["client_id"] == ["client-id"]
    assert "state" in params
    assert "git_review_oauth_state" in response.headers.get("set-cookie", "")


def test_github_callback_creates_server_session_and_cookie() -> None:
    settings = _settings(agent_session_ttl_seconds=600)
    app = Starlette()
    app.add_route("/auth/github/login", lambda request: auth.start_github_oauth_login(request, settings), methods=["GET"])
    app.add_route(
        "/auth/github/callback",
        lambda request: auth.handle_github_oauth_callback(request, settings),
        methods=["GET"],
    )
    client = TestClient(app)
    auth._oauth_states.clear()
    auth._sessions.clear()

    login_response = client.get("/auth/github/login", follow_redirects=False)
    state = parse_qs(urlparse(login_response.headers["location"]).query)["state"][0]

    original_exchange = auth._exchange_code_for_token
    original_build = auth._build_auth_session
    try:
        auth._exchange_code_for_token = lambda code, request, cfg: "gho_test_token"
        auth._build_auth_session = lambda token, ttl: auth.AuthSession(
            session_id="session-123",
            github_token=token,
            github_user_id="42",
            github_login="octocat",
            github_name="The Octocat",
            github_orgs=["acme"],
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=ttl),
        )
        response = client.get(f"/auth/github/callback?code=test-code&state={state}", follow_redirects=False)
    finally:
        auth._exchange_code_for_token = original_exchange
        auth._build_auth_session = original_build

    assert response.status_code == 302
    assert response.headers["location"] == "/"
    assert "git_review_session=session-123" in response.headers.get("set-cookie", "")
    assert auth._sessions["session-123"].github_login == "octocat"


def test_get_session_by_cookie_expires_stale_session() -> None:
    settings = _settings()
    auth._sessions.clear()
    auth._sessions["expired"] = auth.AuthSession(
        session_id="expired",
        github_token="gho_token",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) - timedelta(seconds=1),
    )

    session = auth.get_session_by_cookie("git_review_session=expired", settings)

    assert session is None
    assert "expired" not in auth._sessions


def test_user_model_settings_are_isolated_per_user(tmp_path) -> None:
    settings = _settings(agent_user_settings_path=str(tmp_path / "user-settings.json"))
    auth.save_user_model_settings(
        "u1",
        {"openai_key": "sk-user-1", "openai_base_url": "https://one", "agent_model": "gpt-4o-mini"},
        settings,
    )
    auth.save_user_model_settings(
        "u2",
        {"openai_key": "sk-user-2", "openai_base_url": "https://two", "agent_model": "gpt-4o"},
        settings,
    )

    user1 = auth.load_user_model_settings("u1", settings)
    user2 = auth.load_user_model_settings("u2", settings)

    assert user1["openai_key"] == "sk-user-1"
    assert user2["openai_key"] == "sk-user-2"
    assert user1["agent_model"] == "gpt-4o-mini"
    assert user2["agent_model"] == "gpt-4o"


def test_state_requires_authentication_when_no_session() -> None:
    state = AppState(_reflex_internal_init=True)

    token = state._require_github_token()

    assert token is None
    assert "Sign in with GitHub" in state.auth_status
