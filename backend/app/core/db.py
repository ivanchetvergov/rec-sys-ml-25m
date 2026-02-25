"""Sync psycopg2 connection helper for use in FastAPI route handlers."""

from contextlib import contextmanager
from typing import Generator

import psycopg2
import psycopg2.extras

from app.database import _SYNC_URL


@contextmanager
def get_connection() -> Generator:
    """Context manager that yields a psycopg2 connection with RealDictCursor."""
    conn = psycopg2.connect(_SYNC_URL, connect_timeout=10)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def get_user_by_login(conn, login: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE login = %s", (login,))
        return cur.fetchone()


def get_user_by_email(conn, email: str) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        return cur.fetchone()


def get_user_by_id(conn, user_id: int) -> dict | None:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute("SELECT * FROM users WHERE id = %s", (user_id,))
        return cur.fetchone()


def create_user(conn, login: str, email: str, password_hash: str) -> dict:
    with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
        cur.execute(
            """
            INSERT INTO users (login, email, password_hash, role)
            VALUES (%s, %s, %s, 'user')
            RETURNING id, login, email, role, created_at
            """,
            (login, email, password_hash),
        )
        return dict(cur.fetchone())
