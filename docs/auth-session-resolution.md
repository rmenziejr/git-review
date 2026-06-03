# GitHub OAuth Session Resolution

## Symptom

After GitHub OAuth completed successfully, the browser redirected back to the Reflex frontend and both cookies were present, but workflow actions still reported that the user needed to sign in.

The temporary Settings debug panel showed the important state:

```text
authenticated=no; token=no; header_cookie=yes; header_has_session=yes; reflex_cookie=yes; session_source=none
```

That meant the browser was sending the session cookie correctly. The failure was server-side session lookup, not cookie transport.

## Root Cause

The OAuth callback stored the authenticated session only in an in-memory module dictionary:

```python
_sessions: dict[str, AuthSession] = {}

with _SESSION_LOCK:
    _sessions[session.session_id] = session
```

That worked only while the same Python process handled both the OAuth callback and later Reflex state events. In dev/deployed Reflex setups, the callback and frontend state event can be handled after a reload or by a different worker process. The cookie still contained the session id, but `_sessions` was empty in the process resolving it.

## Final Fix

The browser still receives only an opaque session id cookie. The GitHub token stays server-side. The server now persists the session payload to a private JSON file and falls back to that file when memory is empty.

```python
def save_auth_session(session: AuthSession, settings: AppSettings | None = None) -> None:
    settings = settings or AppSettings()
    with _SESSION_LOCK:
        _sessions[session.session_id] = session
        payload = _read_session_store(settings)
        payload[session.session_id] = session_as_dict(session)
        _write_session_store(settings, payload)
```

The callback stores the session through that helper:

```python
github_token = _exchange_code_for_token(code, request, settings)
session = _build_auth_session(github_token, settings.agent_session_ttl_seconds)
save_auth_session(session, settings)
```

Cookie lookup first checks memory and then loads from the persisted store:

```python
def get_session_by_cookie(cookie_header: str, settings: AppSettings | None = None) -> AuthSession | None:
    settings = settings or AppSettings()
    session_id = get_cookie_value(cookie_header, settings.agent_session_cookie_name)
    if not session_id:
        return None
    with _SESSION_LOCK:
        session = _sessions.get(session_id)
        if session:
            return session
    return load_auth_session(session_id, settings)
```

Logout deletes both copies:

```python
session_id = request.cookies.get(settings.agent_session_cookie_name, "")
if session_id:
    delete_auth_session(session_id, settings)
```

## Redirect Shape

The callback now redirects with only a non-sensitive status flag:

```text
http://localhost:3334?auth=signed-in
```

The older `handoff=...` query parameter is no longer needed. Hydration is handled by the session cookie plus the server-side session store.

## Page Load Behavior

Every Reflex page calls `AppState.on_load`. On load, the app tries to hydrate the session from the request cookie:

```python
self._hydrate_auth_session(settings)
self.auth_prompt_open = not self.authenticated
```

If no valid session exists, the shared page shell opens a GitHub sign-in modal. After OAuth succeeds and the callback redirects to the frontend, the cookie resolves to a persisted session and the modal stays closed.

## Configuration

The session store path is configurable:

```env
AGENT_SESSION_STORE_PATH=.git-review-agent-sessions.json
```

This file contains GitHub OAuth tokens and must stay private. The default path is ignored by git.
