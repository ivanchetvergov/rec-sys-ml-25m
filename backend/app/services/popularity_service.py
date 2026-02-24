"""
PopularityService — loads pre-computed movie stats from the feature store
parquet and returns top-N popular movies.

No database required: the parquet file produced by the ML pipeline already
contains per-movie aggregates (avg_rating, num_ratings, popularity score).
"""
import logging
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

# Columns we actually need — avoids loading 56-column parquet into memory
_MOVIE_COLS = [
    "movieId",
    "title",
    "genres",
    "year",
    "movie_avg_rating",
    "movie_num_ratings",
    "movie_popularity",
]

# Relative path inside the container / local run
_FEATURE_STORE = Path(__file__).parents[3] / "data" / "processed" / "feature_store"
_DATASET_TAG = "ml_v_20260215_184134"
_LINKS_CSV = Path(__file__).parents[3] / "data" / "raw" / "ml-25m" / "links.csv"


class PopularityService:
    """
    Loads the top-N most popular movies from the feature store.

    Movies are sorted by:
        popularity = avg_rating * log(1 + num_ratings)
    This score was computed during the ML preprocessing pipeline.

    The result is cached in memory after the first call — the parquet
    is read only once per application lifetime.
    """

    def __init__(self, feature_store_path: Optional[Path] = None, dataset_tag: str = _DATASET_TAG):
        self._path = (feature_store_path or _FEATURE_STORE) / dataset_tag / "train.parquet"
        self._movies: Optional[pd.DataFrame] = None

    def _load(self) -> pd.DataFrame:
        """Read parquet, keep one row per movie, sort by popularity."""
        logger.info(f"Loading feature store from {self._path}")

        df = pd.read_parquet(self._path, columns=_MOVIE_COLS)

        # Each (userId, movieId) pair is a row — deduplicate to unique movies
        movies = (
            df.drop_duplicates("movieId")
            .sort_values("movie_popularity", ascending=False)
            .reset_index(drop=True)
        )

        # Merge TMDB / IMDB links
        if _LINKS_CSV.exists():
            links = pd.read_csv(_LINKS_CSV, dtype={"imdbId": str})
            movies = movies.merge(links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left")
            logger.info("Merged links.csv — tmdb_id / imdb_id available")
        else:
            movies["imdbId"] = None
            movies["tmdbId"] = None
            logger.warning(f"links.csv not found at {_LINKS_CSV}")

        logger.info(f"Loaded {len(movies):,} unique movies")
        return movies

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

        limit = min(limit, 100)
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
