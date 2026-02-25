"""
PopularityService — loads pre-computed movie stats from the feature store
parquet and returns top-N popular movies.

No database required: the parquet file produced by the ML pipeline already
contains per-movie aggregates (avg_rating, num_ratings, popularity score).
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# PROJECT_ROOT env var is set in docker-compose to /app.
# Locally falls back to 3 levels up: backend/app/services/ -> project root
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])

# Pre-extracted movie catalogue (17K rows) — produced by `make extract-movies`.
# ~300× smaller than reading the full interaction parquet.
_MOVIES_PARQUET = _PROJECT_ROOT / "data" / "processed" / "movies.parquet"


class PopularityService:
    """
    Loads the top-N most popular movies from the feature store.

    Movies are sorted by:
        popularity = avg_rating * log(1 + num_ratings)
    This score was computed during the ML preprocessing pipeline.

    The result is cached in memory after the first call — the parquet
    is read only once per application lifetime.
    """

    def __init__(self, movies_path: Optional[Path] = None):
        self._path = movies_path or _MOVIES_PARQUET
        self._movies: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        """Read movies.parquet — already deduplicated, sorted and link-merged."""
        logger.info(f"Loading movie catalogue from {self._path}")
        if not self._path.exists():
            raise FileNotFoundError(
                f"movies.parquet not found at {self._path}. "
                "Run `make extract-movies` (or `make preprocess`) to generate it."
            )
        df = pd.read_parquet(self._path)
        logger.info(f"Loaded {len(df):,} movies ({self._path.stat().st_size // 1024} KB)")
        return df

    def _ensure_movies_loaded(self) -> None:
        """Public helper so other services can trigger loading without a full get_popular() call."""
        if self._movies is None:
            self._movies = self._load()

    def total_count(self) -> int:
        """Total number of movies in the feature store."""
        if self._movies is None:
            self._movies = self._load()
        return len(self._movies)

    def get_popular(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """
        Return the top-`limit` most popular movies.

        Args:
            limit:  Number of movies to return (max 100).
            offset: Pagination offset.

        Returns:
            List of movie dicts with keys:
                id, title, genres, year, avg_rating, num_ratings, popularity_score
        """
        if self._movies is None:
            self._movies = self._load()

        limit = min(limit, 300) # NUMBER OF FILMS
        rows = self._movies.iloc[offset : offset + limit]

        return [
            {
                "id": int(row.movieId),
                "title": row.title,
                "genres": row.genres if row.genres != "(no genres listed)" else None,
                "year": int(row.year) if pd.notna(row.year) else None,
                "avg_rating": round(float(row.movie_avg_rating), 2) if pd.notna(row.movie_avg_rating) else None,
                "num_ratings": int(row.movie_num_ratings) if pd.notna(row.movie_num_ratings) else None,
                "popularity_score": round(float(row.movie_popularity), 4) if pd.notna(row.movie_popularity) else None,
                "tmdb_id": int(row.tmdbId) if pd.notna(row.tmdbId) else None,
                "imdb_id": str(row.imdbId) if pd.notna(row.imdbId) else None,
            }
            for row in rows.itertuples(index=False)
        ]

    def get_tmdb_id(self, movie_id: int) -> Optional[int]:
        """Return the TMDB id for a given MovieLens movieId, or None."""
        if self._movies is None:
            self._movies = self._load()
        row = self._movies[self._movies["movieId"] == movie_id]
        if row.empty or pd.isna(row.iloc[0]["tmdbId"]):
            return None
        return int(row.iloc[0]["tmdbId"])

    def get_movie(self, movie_id: int) -> Optional[dict]:
        """Return a single movie dict by MovieLens movie_id, or None."""
        if self._movies is None:
            self._movies = self._load()
        rows = self._movies[self._movies["movieId"] == movie_id]
        if rows.empty:
            return None
        row = rows.iloc[0]
        return {
            "id": int(row.movieId),
            "title": row.title,
            "genres": row.genres if row.genres != "(no genres listed)" else None,
            "year": int(row.year) if pd.notna(row.year) else None,
            "avg_rating": round(float(row.movie_avg_rating), 2) if pd.notna(row.movie_avg_rating) else None,
            "num_ratings": int(row.movie_num_ratings) if pd.notna(row.movie_num_ratings) else None,
            "popularity_score": round(float(row.movie_popularity), 4) if pd.notna(row.movie_popularity) else None,
            "tmdb_id": int(row.tmdbId) if pd.notna(row.tmdbId) else None,
            "imdb_id": str(row.imdbId) if pd.notna(row.imdbId) else None,
        }


# ── Module-level singleton ──────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_popularity_service() -> PopularityService:
    """FastAPI dependency — returns the shared PopularityService instance."""
    return PopularityService()
