"""
RecommenderService — loads the trained TwoStageRecommender and serves
personal recommendations via a thin FastAPI-friendly wrapper.

Cold-start (unknown user): falls back to PopularityService.
Model not found on disk: falls back to PopularityService with a warning.
"""

from __future__ import annotations

import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import List, Optional

import pandas as pd

# ── Make project root importable so we can load src.models ────────────────
_PROJECT_ROOT = Path(__file__).parents[3]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from src.models.two_stage_recommender import TwoStageRecommender  # noqa: E402
    _TWO_STAGE_AVAILABLE = True
except ImportError as _e:
    _TWO_STAGE_AVAILABLE = False
    logging.getLogger(__name__).warning(
        "TwoStageRecommender unavailable (%s). Personal recs will fall back to popularity.", _e
    )

from app.services.popularity_service import PopularityService  # noqa: E402

logger = logging.getLogger(__name__)

# ─── Paths ────────────────────────────────────────────────────────────────
_MODEL_DIR = _PROJECT_ROOT / "data" / "models" / "two_stage_ranker"

_FEATURE_STORE = _PROJECT_ROOT / "data" / "processed" / "feature_store"
_DATASET_TAG = "ml_v_20260215_184134"
_LINKS_CSV = _PROJECT_ROOT / "data" / "raw" / "ml-25m" / "links.csv"


# ---------------------------------------------------------------------------
class RecommenderService:
    """
    Loads TwoStageRecommender from disk and enriches results with movie
    metadata from the feature store.
    """

    def __init__(self) -> None:
        self._pipeline: Optional["TwoStageRecommender"] = None
        self._movies: Optional[pd.DataFrame] = None
        self._loaded = False

    # ── Private helpers ───────────────────────────────────────────────────

    def _load_pipeline(self) -> None:
        """Attempt to load the TwoStageRecommender from disk."""
        if not _TWO_STAGE_AVAILABLE:
            logger.warning("TwoStageRecommender package not available; skip load.")
            return
        if not _MODEL_DIR.exists():
            logger.warning(
                "Model dir '%s' not found. Run `make train-ranker` first.", _MODEL_DIR
            )
            return
        try:
            self._pipeline = TwoStageRecommender.load(_MODEL_DIR)
            logger.info("TwoStageRecommender loaded from '%s'.", _MODEL_DIR)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load TwoStageRecommender: %s", exc)

    def _load_movies(self) -> pd.DataFrame:
        """Load feature-store metadata for id → movie info lookups."""
        split_dir = _FEATURE_STORE / _DATASET_TAG

        # Load one split — we only need the metadata columns
        df = pd.read_parquet(split_dir / "train.parquet")
        meta_cols = [
            "movieId", "title", "genres", "year",
            "movie_avg_rating", "movie_num_ratings", "movie_popularity",
        ]
        df = df[meta_cols].drop_duplicates("movieId").reset_index(drop=True)

        # Merge TMDB/IMDB ids if the links file exists
        links_path = _LINKS_CSV
        if not links_path.exists():
            # Try alternate location
            links_path = _FEATURE_STORE / _DATASET_TAG / "links.csv"
        if links_path.exists():
            links = pd.read_csv(links_path)[["movieId", "tmdbId", "imdbId"]]
            df = df.merge(links, on="movieId", how="left")
        else:
            df["tmdbId"] = None
            df["imdbId"] = None

        return df.set_index("movieId")

    def _ensure_loaded(self) -> None:
        if self._loaded:
            return
        self._load_pipeline()
        self._movies = self._load_movies()
        self._loaded = True

    def _enrich(self, item_id: int, score: float) -> dict:
        """Return a fully populated movie dict for the given item_id."""
        row = self._movies.loc[item_id] if item_id in self._movies.index else None
        if row is None:
            return {
                "id": item_id,
                "score": round(float(score), 6),
                "title": None,
                "genres": None,
                "year": None,
                "avg_rating": None,
                "num_ratings": None,
                "popularity_score": None,
                "tmdb_id": None,
            }

        genres_val = row.get("genres", None)
        if genres_val == "(no genres listed)":
            genres_val = None

        return {
            "id": item_id,
            "score": round(float(score), 6),
            "title": row["title"],
            "genres": genres_val,
            "year": int(row["year"]) if pd.notna(row.get("year")) else None,
            "avg_rating": round(float(row["movie_avg_rating"]), 2)
            if pd.notna(row.get("movie_avg_rating"))
            else None,
            "num_ratings": int(row["movie_num_ratings"])
            if pd.notna(row.get("movie_num_ratings"))
            else None,
            "popularity_score": round(float(row["movie_popularity"]), 4)
            if pd.notna(row.get("movie_popularity"))
            else None,
            "tmdb_id": int(row["tmdbId"]) if pd.notna(row.get("tmdbId")) else None,
        }

    # ── Public interface ──────────────────────────────────────────────────

    @property
    def model_available(self) -> bool:
        """True when the TwoStageRecommender model is loaded and ready."""
        self._ensure_loaded()
        return self._pipeline is not None

    def get_personal_recs(
        self,
        user_id: int,
        n: int = 20,
        exclude_seen: bool = True,
        pop_fallback: Optional[PopularityService] = None,
    ) -> tuple[list[dict], str]:
        """
        Return (movie_list, model_name) for the given user.

        Args:
            user_id:        MovieLens user id.
            n:              Number of recommendations.
            exclude_seen:   Whether to exclude already-rated items.
            pop_fallback:   PopularityService to use when falling back.

        Returns:
            (movies, model_name) where model_name is "two_stage" or
            "popularity_fallback".
        """
        self._ensure_loaded()

        # ── Try the two-stage model ───────────────────────────────────────
        if self._pipeline is not None:
            try:
                known_users = set(self._pipeline.candidate_model.user_id_map_.keys())
                if user_id in known_users:
                    raw = self._pipeline.recommend(
                        user_id=user_id,
                        n=n,
                        exclude_items=None,
                        explain=False,
                    )
                    movies = [self._enrich(r["item_id"], r["score"]) for r in raw]
                    return movies, "two_stage"
                else:
                    logger.info("User %d not in training set → popularity fallback.", user_id)
            except Exception as exc:  # noqa: BLE001
                logger.error("two_stage.recommend failed for user %d: %s", user_id, exc)

        # ── Popularity fallback ───────────────────────────────────────────
        if pop_fallback is not None:
            pop_movies = pop_fallback.get_popular(limit=n, offset=0)
            # Normalise score field (popularity_score → score)
            for m in pop_movies:
                m["score"] = m.get("popularity_score", 0.0)
            return pop_movies, "popularity_fallback"

        return [], "popularity_fallback"


# ── Module-level singleton ─────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    """FastAPI dependency — returns the shared RecommenderService instance."""
    return RecommenderService()
