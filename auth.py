"""Прості ролі користувачів (демонстраційний рівень захисту)."""
import hashlib
import hmac
import os
import secrets

import db

ROLES = {"admin": "Адміністратор", "manager": "Енергоменеджер"}


def _hash(password, salt):
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), bytes.fromhex(salt), 120_000).hex()


def create_user(username, password, role):
    if role not in ROLES or not username or len(password) < 6:
        raise ValueError("Некоректні дані користувача (пароль не коротший за 6 символів).")
    salt = secrets.token_hex(16)
    with db.connect() as conn:
        conn.execute("INSERT OR REPLACE INTO App_User(username, role, salt, pwd_hash) VALUES (?,?,?,?)",
                     (username.strip(), role, salt, _hash(password, salt)))


def ensure_default_users():
    with db.connect() as conn:
        n = conn.execute("SELECT COUNT(*) FROM App_User").fetchone()[0]
    if n == 0:
        create_user("admin", os.environ.get("ENERGYPULSE_ADMIN_PASSWORD", "admin123"), "admin")
        create_user("manager", os.environ.get("ENERGYPULSE_MANAGER_PASSWORD", "manager123"), "manager")


def verify(username, password):
    with db.connect() as conn:
        r = conn.execute("SELECT role, salt, pwd_hash FROM App_User WHERE username=?", (username.strip(),)).fetchone()
    if r is None:
        return None
    role, salt, h = r
    return role if hmac.compare_digest(_hash(password, salt), h) else None


def list_users():
    with db.connect() as conn:
        return [(u, r) for u, r in conn.execute("SELECT username, role FROM App_User ORDER BY username")]
