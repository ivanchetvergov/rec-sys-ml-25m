"""
TMDBService — thin async client for The Movie Database API.

Fetches movie metadata (overview, poster, runtime, etc.) and caches
results in-process so the external API is hit only once per unique tmdb_id.

Requires:
    TMDB_API_KEY env var  (free key at https://www.themoviedb.org/settings/api)

If the key is missing or the request fails, returns None gracefully — the
frontend falls back to the gradient poster placeholder.
"""
import logging
import os
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

_TMDB_BASE = "https://api.themoviedb.org/3"
_POSTER_BASE = "https://image.tmdb.org/t/p/w500"
_BACKDROP_BASE = "https://image.tmdb.org/t/p/w1280"


class TMDBService:
    """Async TMDB client with in-memory cache."""

    def __init__(self) -> None:
        self._api_key: str = os.getenv("TMDB_API_KEY", "")
        self._cache: dict[int, Optional[dict]] = {}

    @property
    def _available(self) -> bool:
        return bool(self._api_key)

    async def get_movie_details(self, tmdb_id: int) -> Optional[dict]:
        """
        Fetch movie details from TMDB for the given tmdb_id.

        Returns a dict with keys:
            overview, poster_url, tagline, runtime,
            tmdb_rating, tmdb_votes, release_date

        Returns None if the API key is not set or the request fails.
        """
        if not self._available:
            logger.warning("TMDB_API_KEY not set — poster/overview unavailable")
            return None

        if tmdb_id in self._cache:
            return self._cache[tmdb_id]

        url = f"{_TMDB_BASE}/movie/{tmdb_id}"
        params = {"api_key": self._api_key, "language": "en-US"}

        try:
            async with httpx.AsyncClient(timeout=5.0) as client:
                resp = await client.get(url, params=params)
                resp.raise_for_status()
                data = resp.json()

            poster_path = data.get("poster_path")
            backdrop_path = data.get("backdrop_path")
            result = {
                "overview": data.get("overview") or None,
                "poster_url": f"{_POSTER_BASE}{poster_path}" if poster_path else None,
                "backdrop_url": f"{_BACKDROP_BASE}{backdrop_path}" if backdrop_path else None,
                "tagline": data.get("tagline") or None,
                "runtime": data.get("runtime") or None,
                "tmdb_rating": data.get("vote_average") or None,
                "tmdb_votes": data.get("vote_count") or None,
                "release_date": data.get("release_date") or None,
            }
            self._cache[tmdb_id] = result
            return result

        except Exception as exc:
            logger.error(f"TMDB request failed for tmdb_id={tmdb_id}: {exc}")
            self._cache[tmdb_id] = None
            return None


# ── Module-level singleton ──────────────────────────────────────────────────
_tmdb_service: Optional[TMDBService] = None


def get_tmdb_service() -> TMDBService:
    """FastAPI dependency — returns the shared TMDBService instance."""
    global _tmdb_service
    if _tmdb_service is None:
        _tmdb_service = TMDBService()
    return _tmdb_service
