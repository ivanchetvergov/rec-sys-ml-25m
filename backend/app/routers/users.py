"""Public users router — read-only profile data for user pages.

Movie metadata is NOT stored in interaction tables (migration 009).
It is enriched on-the-fly from PopularityService (movies.parquet).
"""

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.core.db import get_connection
from app.routers.auth import get_current_user
from app.services.popularity_service import PopularityService, get_popularity_service

router = APIRouter(prefix="/users", tags=["users"])

_DEFAULT_AVATARS: tuple[str, ...] = ("cat", "fox", "owl", "panda", "koala")


def _default_avatar_id(user_id: int) -> str:
    return _DEFAULT_AVATARS[abs(int(user_id)) % len(_DEFAULT_AVATARS)]


class PublicWatchedItem(BaseModel):
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


class PublicWatchlistItem(BaseModel):
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


class PublicUserReview(BaseModel):
    movie_id: int
    title: str | None
    rating: int
    review_text: str | None
    created_at: str


class PublicUserProfile(BaseModel):
    user_id: int
    user_login: str
    user_avatar_id: str
    watched_count: int
    watchlist_count: int
    reviews_count: int
    avg_rating: float | None
    watched: list[PublicWatchedItem]
    watchlist: list[PublicWatchlistItem]
    reviews: list[PublicUserReview]


class ProfilePrivacyOut(BaseModel):
    is_profile_private: bool


class ProfilePrivacyUpdateRequest(BaseModel):
    is_profile_private: bool


@router.get("/{user_id}/profile", response_model=PublicUserProfile)
def get_public_user_profile(
    user_id: int,
    items_limit: int = Query(30, ge=1, le=200),
    pop_svc: PopularityService = Depends(get_popularity_service),
):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT id, login, is_profile_private FROM users WHERE id = %s",
                (user_id,),
            )
            user_row = cur.fetchone()
            if not user_row:
                raise HTTPException(status_code=404, detail="User not found")
            if bool(user_row.get("is_profile_private", False)):
                raise HTTPException(status_code=403, detail="Profile is private")

            cur.execute("SELECT COUNT(*) AS c FROM watched WHERE user_id = %s", (user_id,))
            watched_count = int(cur.fetchone()["c"])

            cur.execute("SELECT COUNT(*) AS c FROM watchlist WHERE user_id = %s", (user_id,))
            watchlist_count = int(cur.fetchone()["c"])

            cur.execute("SELECT COUNT(*) AS c FROM reviews WHERE user_id = %s", (user_id,))
            reviews_count = int(cur.fetchone()["c"])

            cur.execute("SELECT AVG(rating) AS v FROM reviews WHERE user_id = %s", (user_id,))
            avg_rating = cur.fetchone()["v"]
            avg_rating = float(avg_rating) if avg_rating is not None else None

            cur.execute(
                "SELECT movie_id, watched_at FROM watched WHERE user_id = %s ORDER BY watched_at DESC LIMIT %s",
                (user_id, items_limit),
            )
            watched_rows = cur.fetchall()

            cur.execute(
                "SELECT movie_id, added_at FROM watchlist WHERE user_id = %s ORDER BY added_at DESC LIMIT %s",
                (user_id, items_limit),
            )
            watchlist_rows = cur.fetchall()

            cur.execute(
                "SELECT movie_id, rating, review_text, created_at FROM reviews WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                (user_id, items_limit),
            )
            review_rows = cur.fetchall()

    def _movie(movie_id: int) -> dict:
        return pop_svc.get_movie(movie_id) or {}

    watched = [
        PublicWatchedItem(
            movie_id=row["movie_id"],
            watched_at=str(row["watched_at"]),
            **{k: _movie(row["movie_id"]).get(k) for k in
               ("title", "genres", "year", "avg_rating", "num_ratings", "popularity_score", "tmdb_id", "imdb_id")},
        )
        for row in watched_rows
    ]

    watchlist = [
        PublicWatchlistItem(
            movie_id=row["movie_id"],
            added_at=str(row["added_at"]),
            **{k: _movie(row["movie_id"]).get(k) for k in
               ("title", "genres", "year", "avg_rating", "num_ratings", "popularity_score", "tmdb_id", "imdb_id")},
        )
        for row in watchlist_rows
    ]

    reviews = [
        PublicUserReview(
            movie_id=row["movie_id"],
            title=_movie(row["movie_id"]).get("title"),
            rating=row["rating"],
            review_text=row.get("review_text"),
            created_at=str(row["created_at"]),
        )
        for row in review_rows
    ]

    return PublicUserProfile(
        user_id=user_row["id"],
        user_login=user_row["login"],
        user_avatar_id=_default_avatar_id(user_row["id"]),
        watched_count=watched_count,
        watchlist_count=watchlist_count,
        reviews_count=reviews_count,
        avg_rating=avg_rating,
        watched=watched,
        watchlist=watchlist,
        reviews=reviews,
    )


@router.get("/me/privacy", response_model=ProfilePrivacyOut)
def get_my_profile_privacy(user: dict = Depends(get_current_user)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            cur.execute("SELECT is_profile_private FROM users WHERE id = %s", (user["id"],))
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
    return ProfilePrivacyOut(is_profile_private=bool(row["is_profile_private"]))


@router.put("/me/privacy", response_model=ProfilePrivacyOut)
def update_my_profile_privacy(
    body: ProfilePrivacyUpdateRequest,
    user: dict = Depends(get_current_user),
):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            cur.execute(
                """
                UPDATE users
                SET is_profile_private = %s
                WHERE id = %s
                RETURNING is_profile_private
                """,
                (body.is_profile_private, user["id"]),
            )
            row = cur.fetchone()
            if not row:
                raise HTTPException(status_code=404, detail="User not found")
    return ProfilePrivacyOut(is_profile_private=bool(row["is_profile_private"]))
