"""Watchlist router — CRUD for the authenticated user's watchlist.

Movie metadata (title, genres, year, …) is NOT stored in the DB.
It is looked up on-the-fly from PopularityService (movies.parquet).
The table stores only the user–movie relationship and the timestamp.
"""

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.db import get_connection
from app.routers.auth import get_current_user
from app.services.popularity_service import PopularityService, get_popularity_service

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class WatchlistAddRequest(BaseModel):
    movie_id: int


class WatchlistItem(BaseModel):
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
    added_at: str


# ── Helpers ───────────────────────────────────────────────────────────────────

def _enrich(row: dict, pop_svc: PopularityService) -> WatchlistItem:
    movie = pop_svc.get_movie(row["movie_id"]) or {}
    return WatchlistItem(
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
        added_at=str(row["added_at"]),
    )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[WatchlistItem])
def get_watchlist(
    user: dict = Depends(get_current_user),
    pop_svc: PopularityService = Depends(get_popularity_service),
):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, user_id, movie_id, added_at FROM watchlist WHERE user_id = %s ORDER BY added_at DESC",
                (user["id"],),
            )
            rows = cur.fetchall()
    return [_enrich(dict(r), pop_svc) for r in rows]


@router.post("", response_model=WatchlistItem, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(
    body: WatchlistAddRequest,
    user: dict = Depends(get_current_user),
    pop_svc: PopularityService = Depends(get_popularity_service),
):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO watchlist (user_id, movie_id)
                VALUES (%s, %s)
                ON CONFLICT (user_id, movie_id) DO UPDATE SET added_at = now()
                RETURNING id, user_id, movie_id, added_at
                """,
                (user["id"], body.movie_id),
            )
            row = cur.fetchone()
    return _enrich(dict(row), pop_svc)


@router.delete("/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def remove_from_watchlist(movie_id: int, user: dict = Depends(get_current_user)):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM watchlist WHERE user_id = %s AND movie_id = %s",
                (user["id"], movie_id),
            )
            deleted = cur.rowcount
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Item not found in watchlist")
