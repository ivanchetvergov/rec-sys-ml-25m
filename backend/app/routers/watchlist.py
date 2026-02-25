"""Watchlist router — CRUD for the authenticated user's watchlist."""

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel

from app.core.db import get_connection
from app.routers.auth import get_current_user

router = APIRouter(prefix="/watchlist", tags=["watchlist"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class WatchlistAddRequest(BaseModel):
    movie_id: int
    title: str
    genres: str | None = None
    year: int | None = None
    avg_rating: float | None = None
    num_ratings: int | None = None
    popularity_score: float | None = None
    tmdb_id: int | None = None
    imdb_id: str | None = None


class WatchlistItem(BaseModel):
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
    added_at: str

    @classmethod
    def from_row(cls, row: dict) -> "WatchlistItem":
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
            added_at=str(row["added_at"]),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[WatchlistItem])
def get_watchlist(user: dict = Depends(get_current_user)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM watchlist WHERE user_id = %s ORDER BY added_at DESC",
                (user["id"],),
            )
            rows = cur.fetchall()
    return [WatchlistItem.from_row(dict(r)) for r in rows]


@router.post("", response_model=WatchlistItem, status_code=status.HTTP_201_CREATED)
def add_to_watchlist(body: WatchlistAddRequest, user: dict = Depends(get_current_user)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO watchlist
                    (user_id, movie_id, title, genres, year, avg_rating,
                     num_ratings, popularity_score, tmdb_id, imdb_id)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
                ON CONFLICT (user_id, movie_id) DO UPDATE
                    SET title            = EXCLUDED.title,
                        genres           = EXCLUDED.genres,
                        year             = EXCLUDED.year,
                        avg_rating       = EXCLUDED.avg_rating,
                        num_ratings      = EXCLUDED.num_ratings,
                        popularity_score = EXCLUDED.popularity_score,
                        tmdb_id          = EXCLUDED.tmdb_id,
                        imdb_id          = EXCLUDED.imdb_id
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
    return WatchlistItem.from_row(dict(row))


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
