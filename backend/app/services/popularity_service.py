"""
PopularityService — loads pre-computed movie stats from the feature store
parquet and returns top-N popular movies.

No database required: the parquet file produced by the ML pipeline already
contains per-movie aggregates (avg_rating, num_ratings, popularity score).
"""
import logging
import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd
from rapidfuzz import fuzz, process

logger = logging.getLogger(__name__)

# PROJECT_ROOT env var is set in docker-compose to /app.
# Locally falls back to 3 levels up: backend/app/services/ -> project root
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])

# Pre-extracted movie catalogue (17K rows) — produced by `make extract-movies`.
# ~300× smaller than reading the full interaction parquet.
_MOVIES_PARQUET = _PROJECT_ROOT / "data" / "processed" / "movies.parquet"

# Articles that MovieLens moves to the end of the title, e.g. "Lion King, The"
_TRAILING_ARTICLES = re.compile(
    r'^(.*),\s*(The|A|An|Les|Le|La|Das|Die|Der|Los|Las|El|Il)\s*$',
    re.IGNORECASE,
)
_YEAR_SUFFIX = re.compile(r'\s*\(\d{4}\)\s*$')


def _normalize_title(title: str) -> str:
    """Build a search-friendly version of a MovieLens title.

    'Lion King, The (1994)'  → 'the lion king'
    'Matrix, The (1999)'     → 'the matrix'
    'Amélie (2001)'          → 'amelie'   (accents stripped)
    """
    t = _YEAR_SUFFIX.sub('', str(title)).strip()
    m = _TRAILING_ARTICLES.match(t)
    if m:
        t = f"{m.group(2)} {m.group(1)}"
    t = t.lower()
    # Strip accents so "Amelie" matches "Amélie", "naive" matches "naïve", etc.
    t = unicodedata.normalize('NFD', t)
    t = ''.join(c for c in t if unicodedata.category(c) != 'Mn')
    return t


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
        self._search_titles: list[str] = []

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
            # Pre-build normalised search titles once — avoids repeated work per query.
            # MovieLens stores "Lion King, The (1994)"; we normalise to "the lion king".
            self._search_titles = [_normalize_title(t) for t in self._movies["title"]]

    def total_count(self) -> int:
        """Total number of movies in the feature store."""
        self._ensure_movies_loaded()
        return len(self._movies)

    def get_popular(self, limit: int = 20, offset: int = 0) -> list[dict]:
        """
        Return the top-`limit` most popular movies.

        Args:
            limit:  Number of movies to return (max 300).
            offset: Pagination offset.

        Returns:
            List of movie dicts with keys:
                id, title, genres, year, avg_rating, num_ratings, popularity_score
        """
        self._ensure_movies_loaded()

        limit = min(limit, 300)
        rows = self._movies.iloc[offset : offset + limit]

        return [self._row_to_dict(row) for row in rows.itertuples(index=False)]

    def search(self, query: str, limit: int = 15) -> list[dict]:
        """
        Fuzzy search movies by title.

        Scoring pipeline (highest wins):
        • Exact normalised match           → 200 pts  (pinned to top)
        • Prefix match                     → 160 pts  (e.g. "star w" → "star wars")
        • partial_ratio  ≥ 50              — substring / prefix queries
        • token_set_ratio ≥ 55            — word-order tolerance & typos

        Ties are broken by popularity (movies.parquet is pre-sorted by
        popularity_score desc, so lower DataFrame index = more popular).
        """
        if not query or not query.strip():
            return []
        self._ensure_movies_loaded()

        # Normalise query the same way titles are normalised (strip accents too)
        q = query.strip().lower()
        q = unicodedata.normalize('NFD', q)
        q = ''.join(c for c in q if unicodedata.category(c) != 'Mn')

        seen: dict[int, float] = {}

        # ── Pass 1: partial_ratio — great for substring / prefix queries ──────
        for _title, score, idx in process.extract(
            q,
            self._search_titles,
            scorer=fuzz.partial_ratio,
            limit=limit * 6,
            score_cutoff=50,
        ):
            seen[idx] = max(seen.get(idx, 0.0), float(score))

        # ── Pass 2: token_set_ratio — great for word reordering & typos ───────
        for _title, score, idx in process.extract(
            q,
            self._search_titles,
            scorer=fuzz.token_set_ratio,
            limit=limit * 6,
            score_cutoff=55,
        ):
            seen[idx] = max(seen.get(idx, 0.0), float(score))

        if not seen:
            return []

        # ── Boost exact and prefix matches so they always rank at the top ──────
        for idx in list(seen.keys()):
            norm = self._search_titles[idx]
            if norm == q:
                seen[idx] = 200.0
            elif norm.startswith(q):
                seen[idx] = max(seen[idx], 160.0)

        # Sort: primary = score desc, secondary = popularity (lower idx = more popular)
        top_indices = sorted(seen, key=lambda i: (-seen[i], i))[:limit]

        return [
            self._row_to_dict(self._movies.iloc[idx])
            for idx in top_indices
        ]

    def get_tmdb_id(self, movie_id: int) -> Optional[int]:
        """Return the TMDB id for a given MovieLens movieId, or None."""
        self._ensure_movies_loaded()
        row = self._movies[self._movies["movieId"] == movie_id]
        if row.empty or pd.isna(row.iloc[0]["tmdbId"]):
            return None
        return int(row.iloc[0]["tmdbId"])

    def get_movie(self, movie_id: int) -> Optional[dict]:
        """Return a single movie dict by MovieLens movie_id, or None."""
        self._ensure_movies_loaded()
        rows = self._movies[self._movies["movieId"] == movie_id]
        if rows.empty:
            return None
        return self._row_to_dict(rows.iloc[0])

    # ── Internal helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _row_to_dict(row) -> dict:
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
