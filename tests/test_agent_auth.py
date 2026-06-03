"""Tests for Reflex GitHub OAuth/session helpers."""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from starlette.applications import Starlette
from starlette.testclient import TestClient

from git_review.webapp.agent_app import auth
from git_review.webapp.agent_app.state import AppState
from git_review.config import AppSettings


def _settings(**overrides: object) -> AppSettings:
    base = {
        "github_oauth_client_id": "client-id",
        "github_oauth_client_secret": "client-secret",
        "agent_cookie_secure": False,
        "agent_session_store_path": str(Path(tempfile.gettempdir()) / "git-review-test-sessions.json"),
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


def test_settings_auth_links_target_backend_url() -> None:
    from git_review.webapp.agent_app.components.settings import _auth_href

    settings = _settings(agent_backend_url="http://localhost:3333/")

    assert _auth_href("/auth/github/login", settings) == "http://localhost:3333/auth/github/login"
    assert _auth_href("auth/github/logout", settings) == "http://localhost:3333/auth/github/logout"


def test_github_callback_creates_server_session_and_cookie() -> None:
    settings = _settings(agent_session_ttl_seconds=600, agent_frontend_url="http://localhost:3334")
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

    with patch.object(auth, "_exchange_code_for_token", return_value=("gho_test_token", ["read:user", "repo"])), patch.object(
        auth,
        "_build_auth_session",
        return_value=auth.AuthSession(
            session_id="session-123",
            github_token="gho_test_token",
            github_user_id="42",
            github_login="octocat",
            github_name="The Octocat",
            github_orgs=["acme"],
            github_scopes=["read:user", "repo"],
            expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
        ),
    ):
        response = client.get(f"/auth/github/callback?code=test-code&state={state}", follow_redirects=False)

    assert response.status_code == 302
    redirect = urlparse(response.headers["location"])
    params = parse_qs(redirect.query)
    assert f"{redirect.scheme}://{redirect.netloc}{redirect.path}" == "http://localhost:3334"
    assert params == {"auth": ["signed-in"]}
    set_cookie = response.headers.get("set-cookie", "")
    assert "git_review_session=session-123" in set_cookie
    assert "git_review_session_reflex=session-123" in set_cookie
    assert auth._sessions["session-123"].github_login == "octocat"
    assert auth._sessions["session-123"].github_scopes == ["read:user", "repo"]


def test_session_cookie_resolves_after_memory_store_is_empty(tmp_path: Path) -> None:
    settings = _settings(agent_session_store_path=str(tmp_path / "sessions.json"))
    auth._sessions.clear()
    session = auth.AuthSession(
        session_id="persisted-session",
        github_token="gho_persisted",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=["acme"],
        github_scopes=["repo", "read:user"],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )

    auth.save_auth_session(session, settings)
    auth._sessions.clear()

    resolved = auth.get_session_by_cookie("git_review_session=persisted-session", settings)

    assert resolved is not None
    assert resolved.github_token == "gho_persisted"
    assert resolved.github_login == "alice"
    assert resolved.github_scopes == ["repo", "read:user"]
    assert auth._sessions["persisted-session"].github_login == "alice"


def test_logout_removes_persisted_session(tmp_path: Path) -> None:
    settings = _settings(
        agent_frontend_url="http://localhost:3334/",
        agent_session_store_path=str(tmp_path / "sessions.json"),
    )
    session = auth.AuthSession(
        session_id="logout-persisted",
        github_token="gho_logout",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    auth.save_auth_session(session, settings)
    auth._sessions.clear()
    app = Starlette()
    app.add_route(
        "/auth/github/logout",
        lambda request: auth.logout_session(request, settings),
        methods=["GET"],
    )
    client = TestClient(app)

    client.get(
        "/auth/github/logout",
        headers={"cookie": "git_review_session=logout-persisted"},
        follow_redirects=False,
    )

    assert auth.get_session_by_cookie("git_review_session=logout-persisted", settings) is None


def test_logout_redirects_to_frontend_url() -> None:
    settings = _settings(agent_frontend_url="http://localhost:3334/")
    app = Starlette()
    app.add_route(
        "/auth/github/logout",
        lambda request: auth.logout_session(request, settings),
        methods=["GET"],
    )
    client = TestClient(app)

    response = client.get(
        "/auth/github/logout",
        headers={"cookie": "git_review_session=session-123"},
        follow_redirects=False,
    )

    assert response.status_code == 302
    assert response.headers["location"] == "http://localhost:3334?auth=signed-out"


def test_auth_session_status_reports_cookie_and_session() -> None:
    settings = _settings()
    auth._sessions.clear()
    auth._sessions["active"] = auth.AuthSession(
        session_id="active",
        github_token="gho_active",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    app = Starlette()
    app.add_route(
        "/auth/github/session",
        lambda request: auth.auth_session_status(request, settings),
        methods=["GET"],
    )
    client = TestClient(app)

    response = client.get(
        "/auth/github/session",
        headers={"cookie": "git_review_session=active"},
    )

    assert response.status_code == 200
    assert response.json()["cookie_present"] is True
    assert response.json()["session_present"] is True
    assert response.json()["github_login"] == "alice"


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


def test_user_model_settings_are_isolated_per_user(tmp_path: Path) -> None:
    settings = _settings(agent_user_settings_path=str(tmp_path / "user-settings.json"))
    auth.save_user_model_settings(
        "u1",
        {
            "openai_key": "sk-user-1",
            "openai_base_url": "https://one",
            "agent_model": "gpt-4o-mini",
            "org_access_token": "gho_org_user_1",
        },
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
    assert user1["org_access_token"] == "gho_org_user_1"
    assert user2["agent_model"] == "gpt-4o"
    assert "org_access_token" not in user2 or user2["org_access_token"] == ""




def test_on_load_prompts_for_sign_in_without_session() -> None:
    auth._sessions.clear()
    state = AppState(_reflex_internal_init=True)

    state.on_load()

    assert state.authenticated is False
    assert state.auth_prompt_open is True


def test_on_load_hydrates_from_cookie_without_handoff() -> None:
    auth._sessions.clear()
    auth._sessions["cookie-session"] = auth.AuthSession(
        session_id="cookie-session",
        github_token="gho_cookie",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=["acme"],
        github_scopes=["repo", "read:user"],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    state = AppState(_reflex_internal_init=True)
    object.__setattr__(
        state.router,
        "headers",
        replace(state.router.headers, cookie="git_review_session=cookie-session"),
    )
    object.__setattr__(state.router, "url", "/?auth=signed-in")

    state.on_load()

    assert state.authenticated is True
    assert state.github_login == "alice"
    assert state.github_scopes == "repo, read:user"
    assert state._require_github_token() == "gho_cookie"
    assert state.active_session_id == "cookie-session"
    assert state.auth_prompt_open is False


def test_auth_session_loads_persisted_org_access_token(tmp_path: Path) -> None:
    settings = _settings(agent_user_settings_path=str(tmp_path / "user-settings.json"))
    auth.save_user_model_settings(
        "1",
        {"org_access_token": "gho_org_access", "agent_model": "gpt-4o"},
        settings,
    )
    session = auth.AuthSession(
        session_id="org-token-session",
        github_token="gho_oauth",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    state = AppState(_reflex_internal_init=True)

    state._apply_auth_session(session, settings)

    assert state.org_access_token == "gho_org_access"
    assert state._require_github_token() == "gho_org_access"


def test_require_github_token_falls_back_to_oauth_token_without_org_token() -> None:
    state = AppState(_reflex_internal_init=True)
    state._github_token = "gho_oauth"
    state.org_access_token = ""

    assert state._require_github_token() == "gho_oauth"



def test_require_github_token_uses_active_session_id_after_cookie_hydration() -> None:
    auth._sessions.clear()
    auth._sessions["active-state"] = auth.AuthSession(
        session_id="active-state",
        github_token="gho_active_state",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    state = AppState(_reflex_internal_init=True)
    state.active_session_id = "active-state"

    token = state._require_github_token()

    assert token == "gho_active_state"
    assert state.authenticated is True


def test_require_github_token_rehydrates_from_cookie_session() -> None:
    auth._sessions.clear()
    auth._sessions["active"] = auth.AuthSession(
        session_id="active",
        github_token="gho_active",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    state = AppState(_reflex_internal_init=True)
    object.__setattr__(
        state.router,
        "headers",
        replace(state.router.headers, cookie="git_review_session=active"),
    )

    token = state._require_github_token()

    assert token == "gho_active"
    assert state.authenticated is True




def test_require_github_token_rehydrates_from_reflex_cookie_value() -> None:
    auth._sessions.clear()
    auth._sessions["reflex"] = auth.AuthSession(
        session_id="reflex",
        github_token="gho_reflex",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    state = AppState(_reflex_internal_init=True)
    state.github_session_id = "reflex"

    token = state._require_github_token()

    assert token == "gho_reflex"
    assert state.authenticated is True




def test_require_github_token_uses_reflex_cookie_when_headers_have_other_cookies() -> None:
    auth._sessions.clear()
    auth._sessions["reflex-other"] = auth.AuthSession(
        session_id="reflex-other",
        github_token="gho_reflex_other",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=[],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    state = AppState(_reflex_internal_init=True)
    state.github_session_id = "reflex-other"
    object.__setattr__(
        state.router,
        "headers",
        replace(state.router.headers, cookie="some_other_cookie=value"),
    )

    token = state._require_github_token()

    assert token == "gho_reflex_other"
    assert state.authenticated is True




def test_opening_settings_refreshes_auth_session() -> None:
    auth._sessions.clear()
    auth._sessions["drawer"] = auth.AuthSession(
        session_id="drawer",
        github_token="gho_drawer",
        github_user_id="1",
        github_login="alice",
        github_name="Alice",
        github_orgs=["acme"],
        expires_at=datetime.now(timezone.utc) + timedelta(seconds=600),
    )
    state = AppState(_reflex_internal_init=True)
    object.__setattr__(
        state.router,
        "headers",
        replace(state.router.headers, cookie="git_review_session=drawer"),
    )

    state.toggle_settings()

    assert state.settings_open is True
    assert state.authenticated is True
    assert state.github_login == "alice"
    assert state.auth_status == "Authenticated"
    assert state.auth_prompt_open is False


def test_state_requires_authentication_when_no_session() -> None:
    state = AppState(_reflex_internal_init=True)
    state.openai_key = "sk-temp"
    state.openai_base_url = "https://example"
    state._hydrate_auth_session(_settings())

    token = state._require_github_token()

    assert token is None
    assert "Sign in with GitHub" in state.auth_status
    assert state.openai_key == ""
    assert state.openai_base_url == ""
