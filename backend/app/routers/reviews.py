"""Reviews router — CRUD for the authenticated user's reviews."""

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, Query, status
from pydantic import BaseModel, Field

from app.core.db import get_connection
from app.routers.auth import get_current_user

router = APIRouter(prefix="/reviews", tags=["reviews"])

_DEFAULT_AVATARS: tuple[str, ...] = ("cat", "fox", "owl", "panda", "koala")


def _default_avatar_id(user_id: int) -> str:
    return _DEFAULT_AVATARS[abs(int(user_id)) % len(_DEFAULT_AVATARS)]


# ── Schemas ───────────────────────────────────────────────────────────────────

class ReviewUpsertRequest(BaseModel):
    movie_id: int
    title: str
    rating: int = Field(..., ge=1, le=5)
    review_text: str | None = None


class ReviewOut(BaseModel):
    id: int
    user_id: int
    movie_id: int
    title: str
    rating: int
    review_text: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: dict) -> "ReviewOut":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            movie_id=row["movie_id"],
            title=row["title"],
            rating=row["rating"],
            review_text=row.get("review_text"),
            created_at=str(row["created_at"]),
        )


class MovieReviewOut(BaseModel):
    id: int
    user_id: int
    user_login: str
    user_avatar_id: str
    movie_id: int
    rating: int
    review_text: str | None
    created_at: str

    @classmethod
    def from_row(cls, row: dict) -> "MovieReviewOut":
        return cls(
            id=row["id"],
            user_id=row["user_id"],
            user_login=row["user_login"],
            user_avatar_id=row.get("user_avatar_id") or _default_avatar_id(row["user_id"]),
            movie_id=row["movie_id"],
            rating=row["rating"],
            review_text=row.get("review_text"),
            created_at=str(row["created_at"]),
        )


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("", response_model=list[ReviewOut])
def get_reviews(user: dict = Depends(get_current_user)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                "SELECT * FROM reviews WHERE user_id = %s ORDER BY created_at DESC",
                (user["id"],),
            )
            rows = cur.fetchall()
    return [ReviewOut.from_row(dict(r)) for r in rows]


@router.post("", response_model=ReviewOut, status_code=status.HTTP_201_CREATED)
def upsert_review(body: ReviewUpsertRequest, user: dict = Depends(get_current_user)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO reviews (user_id, movie_id, title, rating, review_text)
                VALUES (%s, %s, %s, %s, %s)
                ON CONFLICT (user_id, movie_id) DO UPDATE
                    SET title       = EXCLUDED.title,
                        rating      = EXCLUDED.rating,
                        review_text = EXCLUDED.review_text,
                        created_at  = now()
                RETURNING *
                """,
                (user["id"], body.movie_id, body.title, body.rating, body.review_text),
            )
            row = cur.fetchone()
    return ReviewOut.from_row(dict(row))


@router.get("/movie/{movie_id}", response_model=list[MovieReviewOut])
def get_movie_reviews(
    movie_id: int,
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Public reviews for a movie with review author metadata.

    Returns reviews that contain either text or a rating.
    """
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                SELECT
                    r.id,
                    r.user_id,
                    u.login AS user_login,
                    r.movie_id,
                    r.rating,
                    r.review_text,
                    r.created_at
                FROM reviews r
                JOIN users u ON u.id = r.user_id
                WHERE r.movie_id = %s
                  AND (r.review_text IS NOT NULL OR r.rating IS NOT NULL)
                ORDER BY r.created_at DESC
                LIMIT %s OFFSET %s
                """,
                (movie_id, limit, offset),
            )
            rows = cur.fetchall()

    return [MovieReviewOut.from_row(dict(r)) for r in rows]


@router.delete("/{movie_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_review(movie_id: int, user: dict = Depends(get_current_user)):
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM reviews WHERE user_id = %s AND movie_id = %s",
                (user["id"], movie_id),
            )
            deleted = cur.rowcount
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Review not found")
