"""Watched router — CRUD for the authenticated user's watched movies.

Stores every movie a user has marked as watched. This table is the
primary data source for downstream model training (watched + review).
"""

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.db import get_connection
from app.routers.auth import get_current_user

router = APIRouter(prefix="/watched", tags=["watched"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class WatchedAddRequest(BaseModel):
    movie_id: int
    title: str
    genres: str | None = None
    year: int | None = None
    avg_rating: float | None = None
    num_ratings: int | None = None
    popularity_score: float | None = None
    tmdb_id: int | None = None
    imdb_id: str | None = None


class WatchedItem(BaseModel):
    id: int
    user_id: int
    movie_id: int
    title: str
    genres: str | None
    year: int | None
    avg_rating: float | None
    num_ratings: int | None
    popularity_score: float | None
    tmdb_id: int | None
    imdb_id: str | None
    watched_at: str

    @classmethod
    def from_row(cls, row: dict) -> "WatchedItem":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            movie_id=row["movie_id"],
            title=row["title"],
            genres=row.get("genres"),
            year=row.get("year"),
            avg_rating=row.get("avg_rating"),
            num_ratings=row.get("num_ratings"),
            popularity_score=row.get("popularity_score"),
            tmdb_id=row.get("tmdb_id"),
            imdb_id=row.get("imdb_id"),
            watched_at=str(row["watched_at"]),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[WatchedItem])
def get_watched(user: dict = Depends(get_current_user)):
    """Return all movies the current user has marked as watched."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM watched WHERE user_id = %s ORDER BY watched_at DESC",
                (user["id"],),
            )
            rows = cur.fetchall()
    return [WatchedItem.from_row(dict(r)) for r in rows]


@router.post("", response_model=WatchedItem, status_code=status.HTTP_201_CREATED)
def add_watched(body: WatchedAddRequest, user: dict = Depends(get_current_user)):
    """Mark a movie as watched (idempotent — upserts on conflict)."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO watched
                    (user_id, movie_id, title, genres, year, avg_rating,
                     num_ratings, popularity_score, tmdb_id, imdb_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (user_id, movie_id) DO UPDATE
                    SET watched_at = now()
                RETURNING *
                """,
                (
                    user["id"],
                    body.movie_id,
                    body.title,
                    body.genres,
                    body.year,
                    body.avg_rating,
                    body.num_ratings,
                    body.popularity_score,
                    body.tmdb_id,
                    body.imdb_id,
                ),
            )
            row = cur.fetchone()
    return WatchedItem.from_row(dict(row))


@router.delete("/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_watched(movie_id: int, user: dict = Depends(get_current_user)):
    """Remove a movie from the watched list."""
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM watched WHERE user_id = %s AND movie_id = %s",
                (user["id"], movie_id),
            )
            deleted = cur.rowcount
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Movie not in watched list")


# ── ML export endpoint ────────────────────────────────────────────────────────

@router.get("/export", tags=["ml"])
def export_watched_with_reviews(user: dict = Depends(get_current_user)):
    """
    Return joined watched + reviews for the current user.
    Intended for ML training data export.

    Response shape:
    [
      {
        "user_id": 1,
        "movie_id": 42,
        "title": "...",
        "genres": "...",
        "watched_at": "...",
        "rating": 4,           # null if not reviewed
        "review_text": "...",  # null if not reviewed
        "reviewed_at": "..."   # null if not reviewed
      }
    ]
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    w.user_id,
                    w.movie_id,
                    w.title,
                    w.genres,
                    w.year,
                    w.avg_rating       AS dataset_avg_rating,
                    w.tmdb_id,
                    w.watched_at,
                    r.rating,
                    r.review_text,
                    r.created_at       AS reviewed_at
                FROM watched w
                LEFT JOIN reviews r
                    ON r.user_id = w.user_id AND r.movie_id = w.movie_id
                WHERE w.user_id = %s
                ORDER BY w.watched_at DESC
                """,
                (user["id"],),
            )
            rows = cur.fetchall()
    return [dict(r) for r in rows]
