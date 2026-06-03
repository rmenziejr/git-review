"""GitHub OAuth and session persistence helpers for the Reflex app."""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode, urlparse

import requests
from starlette.requests import Request
from starlette.responses import JSONResponse, PlainTextResponse, RedirectResponse, Response

from git_review.config import AppSettings
from git_review.github_client import GitHubClient

_STATE_LOCK = threading.Lock()
_SESSION_LOCK = threading.RLock()
_SETTINGS_LOCK = threading.Lock()

_OAUTH_STATE_TTL_SECONDS = 600
_OAUTH_STATE_COOKIE = "git_review_oauth_state"
_REFLEX_SESSION_COOKIE = "git_review_session_reflex"
_MINIMUM_SCOPES = frozenset({"read:user"})

_oauth_states: dict[str, datetime] = {}
_sessions: dict[str, "AuthSession"] = {}


@dataclass
class AuthSession:
    """Server-side session payload for an authenticated user."""

    session_id: str
    github_token: str
    github_user_id: str
    github_login: str
    github_name: str
    github_orgs: list[str]
    expires_at: datetime
    github_scopes: list[str] = field(default_factory=list)

    def is_expired(self, now: Optional[datetime] = None) -> bool:
        current = now or datetime.now(timezone.utc)
        return current >= self.expires_at


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _to_utc(raw: datetime) -> datetime:
    if raw.tzinfo is None:
        return raw.replace(tzinfo=timezone.utc)
    return raw.astimezone(timezone.utc)


def _cookie_secure_value(settings: AppSettings) -> bool:
    return bool(settings.agent_cookie_secure)


def _cookie_samesite_value(settings: AppSettings) -> str:
    value = (settings.agent_cookie_samesite or "lax").strip().lower()
    return value if value in {"lax", "strict", "none"} else "lax"


def _parse_cookie_header(cookie_header: str) -> dict[str, str]:
    cookies: dict[str, str] = {}
    for part in cookie_header.split(";"):
        item = part.strip()
        if not item or "=" not in item:
            continue
        key, value = item.split("=", 1)
        cookies[key.strip()] = value.strip()
    return cookies


def get_cookie_value(cookie_header: str, cookie_name: str) -> str:
    """Extract a cookie value from a Cookie header string."""
    if not cookie_header:
        return ""
    return _parse_cookie_header(cookie_header).get(cookie_name, "")


def _build_callback_url(request: Request, settings: AppSettings) -> str:
    callback_path = settings.github_oauth_callback_path or "/auth/github/callback"
    if callback_path.startswith("http://") or callback_path.startswith("https://"):
        configured = urlparse(callback_path)
        request_port = request.url.port or (443 if request.url.scheme == "https" else 80)
        configured_port = configured.port or (443 if configured.scheme == "https" else 80)
        if configured.hostname != request.url.hostname or configured_port != request_port:
            raise ValueError("GITHUB_OAUTH_CALLBACK_PATH host must match the current app host.")
        return callback_path
    if not callback_path.startswith("/"):
        callback_path = "/" + callback_path
    return str(request.base_url).rstrip("/") + callback_path


def _validate_oauth_settings(settings: AppSettings) -> Optional[Response]:
    if not settings.github_oauth_client_id or not settings.github_oauth_client_secret:
        return PlainTextResponse(
            "GitHub OAuth is not configured. Set GITHUB_OAUTH_CLIENT_ID and GITHUB_OAUTH_CLIENT_SECRET.",
            status_code=500,
        )
    requested_scopes = {
        scope.strip()
        for scope in (settings.github_oauth_scopes or "").split(",")
        if scope.strip()
    }
    if not _MINIMUM_SCOPES.issubset(requested_scopes):
        return PlainTextResponse(
            "Invalid OAuth scope configuration. GITHUB_OAUTH_SCOPES must include read:user.",
            status_code=500,
        )
    return None


def _frontend_redirect_url(settings: AppSettings, **query_params: str) -> str:
    url = (settings.agent_frontend_url or "/").strip().rstrip("/") or "/"
    query = urlencode({key: value for key, value in query_params.items() if value})
    if not query:
        return url
    separator = "&" if "?" in url else "?"
    return f"{url}{separator}{query}"


def _session_store_path(settings: AppSettings) -> Path:
    return Path(settings.agent_session_store_path).expanduser().resolve()


def _auth_session_from_dict(payload: dict[str, Any]) -> Optional[AuthSession]:
    try:
        expires_at = _to_utc(datetime.fromisoformat(str(payload.get("expires_at") or "")))
    except ValueError:
        return None
    orgs = payload.get("github_orgs", [])
    if not isinstance(orgs, list):
        orgs = []
    scopes = payload.get("github_scopes", [])
    if isinstance(scopes, str):
        scopes = [item.strip() for item in scopes.split(",") if item.strip()]
    if not isinstance(scopes, list):
        scopes = []
    session_id = str(payload.get("session_id") or "").strip()
    github_token = str(payload.get("github_token") or "").strip()
    github_user_id = str(payload.get("github_user_id") or "").strip()
    github_login = str(payload.get("github_login") or "").strip()
    if not session_id or not github_token or not github_login:
        return None
    return AuthSession(
        session_id=session_id,
        github_token=github_token,
        github_user_id=github_user_id,
        github_login=github_login,
        github_name=str(payload.get("github_name") or github_login),
        github_orgs=[str(item) for item in orgs if str(item).strip()],
        expires_at=expires_at,
        github_scopes=[str(item) for item in scopes if str(item).strip()],
    )


def _read_session_store(settings: AppSettings) -> dict[str, Any]:
    path = _session_store_path(settings)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _write_session_store(settings: AppSettings, payload: dict[str, Any]) -> None:
    path = _session_store_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    tmp_path.chmod(0o600)
    tmp_path.replace(path)


def save_auth_session(session: AuthSession, settings: Optional[AppSettings] = None) -> None:
    """Persist an auth session so cookie lookup survives worker boundaries."""
    settings = settings or AppSettings()
    with _SESSION_LOCK:
        _sessions[session.session_id] = session
        payload = _read_session_store(settings)
        payload[session.session_id] = session_as_dict(session)
        _write_session_store(settings, payload)


def load_auth_session(session_id: str, settings: Optional[AppSettings] = None) -> Optional[AuthSession]:
    """Load a persisted auth session by id."""
    settings = settings or AppSettings()
    session_id = session_id.strip()
    if not session_id:
        return None
    payload = _read_session_store(settings).get(session_id)
    if not isinstance(payload, dict):
        return None
    session = _auth_session_from_dict(payload)
    if session is None:
        return None
    if session.is_expired():
        delete_auth_session(session_id, settings)
        return None
    with _SESSION_LOCK:
        _sessions[session.session_id] = session
    return session


def delete_auth_session(session_id: str, settings: Optional[AppSettings] = None) -> None:
    """Delete an auth session from memory and disk."""
    settings = settings or AppSettings()
    session_id = session_id.strip()
    if not session_id:
        return
    with _SESSION_LOCK:
        _sessions.pop(session_id, None)
        payload = _read_session_store(settings)
        if session_id in payload:
            payload.pop(session_id, None)
            _write_session_store(settings, payload)


def start_github_oauth_login(request: Request, settings: Optional[AppSettings] = None) -> Response:
    """Start OAuth login by redirecting to GitHub authorize endpoint."""
    settings = settings or AppSettings()
    error_response = _validate_oauth_settings(settings)
    if error_response:
        return error_response

    state = secrets.token_urlsafe(32)
    expires_at = _now_utc() + timedelta(seconds=_OAUTH_STATE_TTL_SECONDS)
    with _STATE_LOCK:
        _oauth_states[state] = expires_at

    callback_url = _build_callback_url(request, settings)
    query = urlencode(
        {
            "client_id": settings.github_oauth_client_id,
            "redirect_uri": callback_url,
            "scope": settings.github_oauth_scopes,
            "state": state,
        }
    )
    authorize_url = f"{settings.github_oauth_authorize_url}?{query}"
    response = RedirectResponse(url=authorize_url, status_code=302)
    response.set_cookie(
        _OAUTH_STATE_COOKIE,
        state,
        httponly=True,
        secure=_cookie_secure_value(settings),
        samesite=_cookie_samesite_value(settings),
        path="/",
        max_age=_OAUTH_STATE_TTL_SECONDS,
    )
    return response


def _exchange_code_for_token(
    code: str, request: Request, settings: AppSettings
) -> tuple[str, list[str]]:
    callback_url = _build_callback_url(request, settings)
    response = requests.post(
        settings.github_oauth_token_url,
        headers={"Accept": "application/json"},
        data={
            "client_id": settings.github_oauth_client_id,
            "client_secret": settings.github_oauth_client_secret,
            "code": code,
            "redirect_uri": callback_url,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    token = str(payload.get("access_token") or "").strip()
    if not token:
        raise ValueError("GitHub OAuth token exchange did not return an access token.")
    raw_scope = str(payload.get("scope") or "").strip()
    scopes = [item.strip() for item in raw_scope.split(",") if item.strip()]
    return token, scopes


def _build_auth_session(
    github_token: str, ttl_seconds: int, github_scopes: list[str] | None = None
) -> AuthSession:
    gh = GitHubClient(token=github_token)
    user = gh.get_authenticated_user()
    orgs = gh.get_user_orgs()
    expires_at = _now_utc() + timedelta(seconds=max(ttl_seconds, 1))
    return AuthSession(
        session_id=secrets.token_urlsafe(32),
        github_token=github_token,
        github_user_id=str(user.get("id") or user.get("login") or ""),
        github_login=str(user.get("login") or ""),
        github_name=str(user.get("name") or user.get("login") or ""),
        github_orgs=sorted(
            {
                str(item.get("login") or "").strip()
                for item in orgs
                if isinstance(item, dict) and str(item.get("login") or "").strip()
            }
        ),
        expires_at=expires_at,
        github_scopes=github_scopes or [],
    )


def handle_github_oauth_callback(request: Request, settings: Optional[AppSettings] = None) -> Response:
    """Handle OAuth callback, create server-side session, and set cookie."""
    settings = settings or AppSettings()
    error_response = _validate_oauth_settings(settings)
    if error_response:
        return error_response

    query_state = str(request.query_params.get("state") or "").strip()
    code = str(request.query_params.get("code") or "").strip()
    cookie_state = request.cookies.get(_OAUTH_STATE_COOKIE, "")
    if not code:
        return PlainTextResponse("Missing OAuth code.", status_code=400)
    if not query_state:
        return PlainTextResponse("Missing OAuth state.", status_code=400)
    if not cookie_state:
        return PlainTextResponse("Missing OAuth state cookie.", status_code=400)
    if cookie_state != query_state:
        return PlainTextResponse("OAuth state mismatch.", status_code=400)

    with _STATE_LOCK:
        state_expiry = _oauth_states.pop(query_state, None)
    if state_expiry is None or _to_utc(state_expiry) <= _now_utc():
        return PlainTextResponse("OAuth state expired. Please try signing in again.", status_code=400)

    try:
        github_token, github_scopes = _exchange_code_for_token(code, request, settings)
        session = _build_auth_session(
            github_token, settings.agent_session_ttl_seconds, github_scopes
        )
    except (requests.RequestException, ValueError) as exc:
        return PlainTextResponse(f"OAuth login failed: {exc}", status_code=400)

    save_auth_session(session, settings)

    response = RedirectResponse(
        url=_frontend_redirect_url(settings, auth="signed-in"),
        status_code=302,
    )
    cookie_max_age = max(settings.agent_session_ttl_seconds, 1)
    response.set_cookie(
        settings.agent_session_cookie_name,
        session.session_id,
        httponly=True,
        secure=_cookie_secure_value(settings),
        samesite=_cookie_samesite_value(settings),
        path="/",
        max_age=cookie_max_age,
    )
    response.set_cookie(
        _REFLEX_SESSION_COOKIE,
        session.session_id,
        httponly=False,
        secure=_cookie_secure_value(settings),
        samesite=_cookie_samesite_value(settings),
        path="/",
        max_age=cookie_max_age,
    )
    response.delete_cookie(_OAUTH_STATE_COOKIE, path="/")
    return response


def logout_session(request: Request, settings: Optional[AppSettings] = None) -> Response:
    """Invalidate the current server-side session and clear the cookie."""
    settings = settings or AppSettings()
    session_id = request.cookies.get(settings.agent_session_cookie_name, "")
    if session_id:
        delete_auth_session(session_id, settings)
    response = RedirectResponse(url=_frontend_redirect_url(settings, auth="signed-out"), status_code=302)
    response.delete_cookie(settings.agent_session_cookie_name, path="/")
    response.delete_cookie(_REFLEX_SESSION_COOKIE, path="/")
    response.delete_cookie(_OAUTH_STATE_COOKIE, path="/")
    return response


def get_session_by_cookie(cookie_header: str, settings: Optional[AppSettings] = None) -> Optional[AuthSession]:
    """Resolve an active session from the request Cookie header."""
    settings = settings or AppSettings()
    session_id = get_cookie_value(cookie_header, settings.agent_session_cookie_name)
    if not session_id:
        return None
    with _SESSION_LOCK:
        session = _sessions.get(session_id)
        if session and session.is_expired():
            delete_auth_session(session_id, settings)
            return None
        if session:
            return session
    return load_auth_session(session_id, settings)


def auth_session_status(request: Request, settings: Optional[AppSettings] = None) -> Response:
    """Return non-sensitive auth session state for browser debugging."""
    settings = settings or AppSettings()
    session_id = request.cookies.get(settings.agent_session_cookie_name, "")
    session = get_session_by_cookie(request.headers.get("cookie", ""), settings)
    return JSONResponse(
        {
            "cookie_present": bool(session_id),
            "session_present": session is not None,
            "github_login": session.github_login if session else "",
            "github_scopes": session.github_scopes if session else [],
            "session_expires_at": session.expires_at.isoformat() if session else "",
        }
    )


def _settings_file_path(settings: AppSettings) -> Path:
    return Path(settings.agent_user_settings_path).expanduser().resolve()


def load_user_model_settings(user_id: str, settings: Optional[AppSettings] = None) -> dict[str, str]:
    """Load persisted model settings for one authenticated user."""
    settings = settings or AppSettings()
    if not user_id:
        return {}
    path = _settings_file_path(settings)
    with _SETTINGS_LOCK:
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return {}
    user_payload = payload.get(str(user_id), {}) if isinstance(payload, dict) else {}
    if not isinstance(user_payload, dict):
        return {}
    result: dict[str, str] = {}
    for key in ("openai_key", "openai_base_url", "agent_model", "org_access_token"):
        value = user_payload.get(key)
        if isinstance(value, str):
            result[key] = value
    return result


def save_user_model_settings(
    user_id: str,
    settings_payload: dict[str, str],
    settings: Optional[AppSettings] = None,
) -> None:
    """Persist model settings for an authenticated user."""
    settings = settings or AppSettings()
    if not user_id:
        return
    path = _settings_file_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _SETTINGS_LOCK:
        data: dict[str, Any] = {}
        if path.exists():
            try:
                raw = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    data = raw
            except (json.JSONDecodeError, OSError):
                data = {}
        data[str(user_id)] = {
            "openai_key": str(settings_payload.get("openai_key") or ""),
            "openai_base_url": str(settings_payload.get("openai_base_url") or ""),
            "agent_model": str(settings_payload.get("agent_model") or ""),
            "org_access_token": str(settings_payload.get("org_access_token") or ""),
        }
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp_path.chmod(0o600)
        tmp_path.replace(path)


def session_as_dict(session: AuthSession) -> dict[str, Any]:
    """Serialize a session for testing/debugging."""
    payload = asdict(session)
    payload["expires_at"] = session.expires_at.isoformat()
    return payload
