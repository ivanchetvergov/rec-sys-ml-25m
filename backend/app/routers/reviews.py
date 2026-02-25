"""Reviews router — CRUD for the authenticated user's reviews."""

import psycopg2.extras
from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from app.core.db import get_connection
from app.routers.auth import get_current_user

router = APIRouter(prefix="/reviews", tags=["reviews"])


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
