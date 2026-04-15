"""Watched router — CRUD for the authenticated user's watched movies.

Movie metadata is NOT stored in the DB — it is looked up on-the-fly from
PopularityService (movies.parquet). The table stores only the user–movie
relationship and watched_at timestamp. This table is also the primary data
source for downstream model training.
"""

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.db import get_connection
from app.routers.auth import get_current_user
from app.services.popularity_service import PopularityService, get_popularity_service

router = APIRouter(prefix="/watched", tags=["watched"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class WatchedAddRequest(BaseModel):
    movie_id: int


class WatchedItem(BaseModel):
    id: int
    user_id: int
    movie_id: int
    title: str | None
    genres: str | None
    year: int | None
    avg_rating: float | None
    num_ratings: int | None
    popularity_score: float | None
    tmdb_id: int | None
    imdb_id: str | None
    watched_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich(row: dict, pop_svc: PopularityService) -> WatchedItem:
    movie = pop_svc.get_movie(row["movie_id"]) or {}
    return WatchedItem(
        id=row["id"],
        user_id=row["user_id"],
        movie_id=row["movie_id"],
        title=movie.get("title"),
        genres=movie.get("genres"),
        year=movie.get("year"),
        avg_rating=movie.get("avg_rating"),
        num_ratings=movie.get("num_ratings"),
        popularity_score=movie.get("popularity_score"),
        tmdb_id=movie.get("tmdb_id"),
        imdb_id=movie.get("imdb_id"),
        watched_at=str(row["watched_at"]),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[WatchedItem])
def get_watched(
    user: dict = Depends(get_current_user),
    pop_svc: PopularityService = Depends(get_popularity_service),
):
    """Return all movies the current user has marked as watched."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, user_id, movie_id, watched_at FROM watched WHERE user_id = %s ORDER BY watched_at DESC",
                (user["id"],),
            )
            rows = cur.fetchall()
    return [_enrich(dict(r), pop_svc) for r in rows]


@router.post("", response_model=WatchedItem, status_code=status.HTTP_201_CREATED)
def add_watched(
    body: WatchedAddRequest,
    user: dict = Depends(get_current_user),
    pop_svc: PopularityService = Depends(get_popularity_service),
):
    """Mark a movie as watched (idempotent — upserts on conflict)."""
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO watched (user_id, movie_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, movie_id) DO UPDATE SET watched_at = now()
                RETURNING id, user_id, movie_id, watched_at
                """,
                (user["id"], body.movie_id),
            )
            row = cur.fetchone()
    return _enrich(dict(row), pop_svc)


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
def export_watched_with_reviews(
    user: dict = Depends(get_current_user),
    pop_svc: PopularityService = Depends(get_popularity_service),
):
    """
    Return joined watched + reviews for the current user, enriched with
    movie metadata from the feature store. Intended for ML training data export.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    w.user_id,
                    w.movie_id,
                    w.watched_at,
                    r.rating,
                    r.review_text,
                    r.created_at AS reviewed_at
                FROM watched w
                LEFT JOIN reviews r
                    ON r.user_id = w.user_id AND r.movie_id = w.movie_id
                WHERE w.user_id = %s
                ORDER BY w.watched_at DESC
                """,
                (user["id"],),
            )
            rows = cur.fetchall()

    result = []
    for row in rows:
        r = dict(row)
        movie = pop_svc.get_movie(r["movie_id"]) or {}
        result.append({
            "user_id": r["user_id"],
            "movie_id": r["movie_id"],
            "title": movie.get("title"),
            "genres": movie.get("genres"),
            "year": movie.get("year"),
            "dataset_avg_rating": movie.get("avg_rating"),
            "tmdb_id": movie.get("tmdb_id"),
            "watched_at": str(r["watched_at"]),
            "rating": r["rating"],
            "review_text": r["review_text"],
            "reviewed_at": str(r["reviewed_at"]) if r["reviewed_at"] else None,
        })
    return result
