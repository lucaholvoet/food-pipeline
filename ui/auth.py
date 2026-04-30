import sqlite3
import os
import hashlib
import secrets
from contextlib import contextmanager

DB_PATH = os.environ.get("DB_PATH", "/app/meals.db")

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_auth_db():
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                salt TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
        """)

def _hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode()).hexdigest()

def register_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    if len(username) < 3:
        return False, "Username must be at least 3 characters."
    if len(password) < 6:
        return False, "Password must be at least 6 characters."
    
    salt = secrets.token_hex(16)
    password_hash = _hash_password(password, salt)
    
    try:
        from datetime import datetime
        with get_conn() as conn:
            conn.execute("""
                INSERT INTO users (username, password_hash, salt, created_at)
                VALUES (?, ?, ?, ?)
            """, (username, password_hash, salt, datetime.now().isoformat()))
        return True, username
    except sqlite3.IntegrityError:
        return False, "Username already taken."

def login_user(username: str, password: str) -> tuple[bool, str]:
    username = username.strip().lower()
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    
    if not row:
        return False, "Username not found."
    
    expected = _hash_password(password, row["salt"])
    if expected != row["password_hash"]:
        return False, "Incorrect password."
    
    return True, username