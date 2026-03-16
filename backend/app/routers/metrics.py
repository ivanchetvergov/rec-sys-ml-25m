"""Metrics router — lightweight event ingestion for product KPI tracking."""

import psycopg2.extras
from fastapi import APIRouter, Depends
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from app.core.db import get_connection, get_user_by_id
from app.core.security import decode_access_token

router = APIRouter(prefix="/metrics", tags=["metrics"])
_bearer = HTTPBearer(auto_error=False)


class KpiEventIn(BaseModel):
    session_id: str = Field(..., min_length=6, max_length=128)
    event_type: str = Field(..., min_length=2, max_length=64)
    block: str | None = Field(default=None, max_length=64)
    movie_id: int | None = None


def _resolve_user_id(
    creds: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> int | None:
    if creds is None:
        return None
    payload = decode_access_token(creds.credentials)
    if not payload:
        return None
    sub = payload.get("sub")
    if sub is None:
        return None
    try:
        uid = int(sub)
    except Exception:
        return None
    with get_connection() as conn:
        user = get_user_by_id(conn, uid)
    return uid if user else None


@router.post("/event", status_code=202)
def track_event(body: KpiEventIn, user_id: int | None = Depends(_resolve_user_id)):
    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(
                """
                INSERT INTO rec_events (session_id, user_id, event_type, block, movie_id)
                VALUES (%s, %s, %s, %s, %s)
                """,
                (body.session_id, user_id, body.event_type, body.block, body.movie_id),
            )
    return {"accepted": True}
