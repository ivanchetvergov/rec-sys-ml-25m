"""Admin dashboard router — metrics from existing tables.

All endpoints require admin role.
Data is read directly via SQL aggregations — no separate event tracking needed
since users/watched/reviews/watchlist tables have timestamps.
"""
from __future__ import annotations

import logging
from typing import Any

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Query

from app.core.db import get_connection
from app.routers.auth import get_current_user

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/admin", tags=["admin"])


# ── Auth guard ────────────────────────────────────────────────────────────────

def require_admin(current_user: dict = Depends(get_current_user)) -> dict:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return current_user


# ── DB helpers ────────────────────────────────────────────────────────────────

def _query(sql: str, params: tuple = ()) -> list[dict]:
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params)
            return [dict(r) for r in cur.fetchall()]


def _scalar(sql: str, params: tuple = ()) -> Any:
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            row = cur.fetchone()
            return row[0] if row else None


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("/stats/overview")
def stats_overview(_admin: dict = Depends(require_admin)):
    """Top-level counters for the dashboard stat cards."""
    return {
        "total_users":     _scalar("SELECT COUNT(*) FROM users"),
        "total_watched":   _scalar("SELECT COUNT(*) FROM watched"),
        "total_reviews":   _scalar("SELECT COUNT(*) FROM reviews"),
        "total_watchlist": _scalar("SELECT COUNT(*) FROM watchlist"),
        "avg_rating":      _scalar(
            "SELECT ROUND(AVG(rating)::numeric, 2) FROM reviews"
        ),
        "users_today": _scalar(
            "SELECT COUNT(*) FROM users WHERE created_at::date = CURRENT_DATE"
        ),
        "active_today": _scalar("""
            SELECT COUNT(DISTINCT user_id) FROM (
                SELECT user_id FROM watched   WHERE watched_at::date = CURRENT_DATE
                UNION ALL
                SELECT user_id FROM reviews   WHERE created_at::date = CURRENT_DATE
                UNION ALL
                SELECT user_id FROM watchlist WHERE added_at::date   = CURRENT_DATE
            ) t
        """),
    }


@router.get("/stats/daily")
def stats_daily(
    days: int = Query(30, ge=7, le=90),
    _admin: dict = Depends(require_admin),
):
    """Per-day activity for the last N days (for line chart)."""
    rows = _query("""
        WITH dates AS (
            SELECT generate_series(
                CURRENT_DATE - (%s - 1) * INTERVAL '1 day',
                CURRENT_DATE,
                INTERVAL '1 day'
            )::date AS day
        )
        SELECT
            d.day::text                AS date,
            COUNT(DISTINCT u.id)       AS new_users,
            COUNT(DISTINCT w.id)       AS new_watched,
            COUNT(DISTINCT r.id)       AS new_reviews,
            COUNT(DISTINCT wl.id)      AS new_watchlist
        FROM dates d
        LEFT JOIN users     u  ON u.created_at::date  = d.day
        LEFT JOIN watched   w  ON w.watched_at::date  = d.day
        LEFT JOIN reviews   r  ON r.created_at::date  = d.day
        LEFT JOIN watchlist wl ON wl.added_at::date   = d.day
        GROUP BY d.day
        ORDER BY d.day
    """, (days,))
    return rows


@router.get("/stats/top-movies")
def stats_top_movies(
    limit: int = Query(8, ge=1, le=20),
    _admin: dict = Depends(require_admin),
):
    """Most watched and highest-rated movies."""
    most_watched = _query("""
        SELECT movie_id, title, COUNT(*) AS watch_count
        FROM watched
        WHERE title IS NOT NULL
        GROUP BY movie_id, title
        ORDER BY watch_count DESC
        LIMIT %s
    """, (limit,))

    top_rated = _query("""
        SELECT movie_id, title,
               ROUND(AVG(rating)::numeric, 2) AS avg_rating,
               COUNT(*) AS review_count
        FROM reviews
        WHERE title IS NOT NULL
        GROUP BY movie_id, title
        HAVING COUNT(*) >= 1
        ORDER BY avg_rating DESC, review_count DESC
        LIMIT %s
    """, (limit,))

    return {"most_watched": most_watched, "top_rated": top_rated}


@router.get("/stats/rating-distribution")
def stats_rating_dist(_admin: dict = Depends(require_admin)):
    """Count of reviews per star rating (1–5)."""
    rows = _query("""
        SELECT rating, COUNT(*) AS count
        FROM reviews
        GROUP BY rating
        ORDER BY rating
    """)
    dist = {int(r["rating"]): int(r["count"]) for r in rows}
    return [{"rating": i, "count": dist.get(i, 0)} for i in range(1, 6)]


@router.get("/stats/users")
def stats_users(
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    _admin: dict = Depends(require_admin),
):
    """Paginated user list with per-user activity counters."""
    users = _query("""
        SELECT
            u.id,
            u.login,
            u.email,
            u.role,
            u.created_at::text                  AS created_at,
            COUNT(DISTINCT w.id)                AS watched_count,
            COUNT(DISTINCT r.id)                AS review_count,
            COUNT(DISTINCT wl.id)               AS watchlist_count
        FROM users u
        LEFT JOIN watched   w  ON w.user_id  = u.id
        LEFT JOIN reviews   r  ON r.user_id  = u.id
        LEFT JOIN watchlist wl ON wl.user_id = u.id
        GROUP BY u.id, u.login, u.email, u.role, u.created_at
        ORDER BY u.created_at DESC
        LIMIT %s OFFSET %s
    """, (limit, offset))
    total = _scalar("SELECT COUNT(*) FROM users")
    return {"total": total, "users": users}
