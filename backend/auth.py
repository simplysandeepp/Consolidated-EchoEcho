"""Lightweight file-backed auth for EchoEcho.

Stores users in backend/users.json with SHA-256 password hashes and is seeded
with the demo account (test@echo.com / test@123). Tokens are opaque random
strings — enough for the demo's localStorage-based session handling.
"""

from __future__ import annotations

import hashlib
import json
import base64
import secrets
from pathlib import Path
from threading import Lock

BASE_DIR = Path(__file__).resolve().parent
USERS_FILE = BASE_DIR / "users.json"
SESSIONS_FILE = BASE_DIR / "sessions.json"

DEMO_EMAIL = "test@echo.com"
DEMO_PASSWORD = "test@123"
DEMO_NAME = "Echo Tester"

_users_lock = Lock()
_sessions_lock = Lock()


class AuthError(ValueError):
    pass


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode("utf-8")).hexdigest()


def _seed_users() -> dict[str, dict[str, str]]:
    return {DEMO_EMAIL: {"name": DEMO_NAME, "password_hash": _hash_password(DEMO_PASSWORD)}}


def _read_users() -> dict[str, dict[str, str]]:
    if not USERS_FILE.exists():
        users = _seed_users()
        USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")
        return users
    try:
        data = json.loads(USERS_FILE.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError
    except (json.JSONDecodeError, ValueError):
        data = {}
    if DEMO_EMAIL not in data:
        data.update(_seed_users())
        USERS_FILE.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _read_sessions() -> dict[str, str]:
    if not SESSIONS_FILE.exists():
        SESSIONS_FILE.write_text("{}", encoding="utf-8")
        return {}
    try:
        data = json.loads(SESSIONS_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def _write_sessions(sessions: dict[str, str]) -> None:
    SESSIONS_FILE.write_text(json.dumps(sessions, indent=2), encoding="utf-8")


def _issue_session(email: str, name: str) -> dict[str, object]:
    token = secrets.token_hex(24)
    with _sessions_lock:
        sessions = _read_sessions()
        sessions[token] = email
        _write_sessions(sessions)
    return {
        "ok": True,
        "token": token,
        "user": {"name": name, "email": email},
    }


def user_email_for_token(token: str) -> str | None:
    cleaned = token.strip()
    if not cleaned:
        return None
    with _sessions_lock:
        email = _read_sessions().get(cleaned)
    if email:
        return email.strip().lower()
    claims = firebase_claims_for_token(cleaned)
    email = str(claims.get("email") or "").strip().lower()
    return email or None


def firebase_claims_for_token(token: str) -> dict[str, object]:
    """Decode non-sensitive Firebase ID token claims for local user routing.

    Firebase verifies the sign-in in the browser. The backend only needs the
    stable user key so JSON files are isolated per account; malformed tokens
    simply behave like an anonymous/default session.
    """
    parts = token.split(".")
    if len(parts) < 2:
        return {}
    payload = parts[1]
    payload += "=" * (-len(payload) % 4)
    try:
        decoded = base64.urlsafe_b64decode(payload.encode("ascii"))
        data = json.loads(decoded.decode("utf-8"))
    except (ValueError, json.JSONDecodeError):
        return {}
    return data if isinstance(data, dict) else {}


def user_name_for_token(token: str) -> str:
    claims = firebase_claims_for_token(token.strip())
    name = str(claims.get("name") or "").strip()
    if name:
        return name
    email = str(claims.get("email") or "").strip().lower()
    return user_name_for_email(email) if email else ""


def user_name_for_email(email: str) -> str:
    normalized = email.strip().lower()
    if not normalized:
        return ""
    with _users_lock:
        user = _read_users().get(normalized) or {}
    return str(user.get("name") or normalized.split("@")[0].replace(".", " ").title())


def login(email: str, password: str) -> dict[str, object]:
    normalized = email.strip().lower()
    with _users_lock:
        user = _read_users().get(normalized)
    if not user or user.get("password_hash") != _hash_password(password):
        raise AuthError("Invalid email or password.")
    return _issue_session(normalized, str(user.get("name") or normalized.split("@")[0].title()))


def signup(name: str, email: str, password: str) -> dict[str, object]:
    normalized = email.strip().lower()
    if "@" not in normalized or "." not in normalized.split("@")[-1]:
        raise AuthError("Please enter a valid email address.")
    if len(password) < 6:
        raise AuthError("Password must be at least 6 characters.")
    display_name = name.strip() or normalized.split("@")[0].replace(".", " ").title()
    with _users_lock:
        users = _read_users()
        if normalized in users:
            raise AuthError("An account with this email already exists.")
        users[normalized] = {"name": display_name, "password_hash": _hash_password(password)}
        USERS_FILE.write_text(json.dumps(users, indent=2), encoding="utf-8")
    return _issue_session(normalized, display_name)
