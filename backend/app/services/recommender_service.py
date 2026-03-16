"""
RecommenderService — loads the trained TwoStageRecommender and serves
personal recommendations via a thin FastAPI-friendly wrapper.

Cold-start (unknown user): falls back to PopularityService.
Model not found on disk: falls back to PopularityService with a warning.
"""

from __future__ import annotations

import logging
import os
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set

import pandas as pd

# PROJECT_ROOT env var is set in docker-compose to /app.
# Locally falls back to 3 levels up: backend/app/services/ -> project root
_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
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
from app.services.similarity_service import get_similarity_service  # noqa: E402
from app.core.db import get_connection  # noqa: E402

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
        """Load movie metadata for id → display info lookups.

        Reuses the PopularityService singleton (already loaded) to avoid
        re-reading the 17.4M-row train.parquet a second time.
        """
        from app.services.popularity_service import get_popularity_service
        pop = get_popularity_service()
        # Trigger load if not yet done, then grab the internal DataFrame
        pop._ensure_movies_loaded()
        df = pop._movies.copy()  # already: movieId, title, genres, year, …, tmdbId, imdbId
        df = df.rename(columns={"tmdbId": "tmdbId", "imdbId": "imdbId"})  # keep names as-is
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

    def _extract_movie_genres(self, movie_id: int) -> List[str]:
        if self._movies is None or movie_id not in self._movies.index:
            return []
        genres_val = self._movies.loc[movie_id].get("genres")
        if not isinstance(genres_val, str):
            return []
        return [g for g in genres_val.split("|") if g and g != "(no genres listed)"]

    def _load_user_live_signals(self, user_id: int) -> Dict:
        """Read recent watched/review events and derive online preference signals."""
        seen_ids: Set[int] = set()
        seed_ids: List[int] = []
        genre_pref: Dict[str, float] = defaultdict(float)

        with get_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT movie_id, watched_at
                    FROM watched
                    WHERE user_id = %s
                    ORDER BY watched_at DESC
                    LIMIT 40
                    """,
                    (user_id,),
                )
                watched_rows = cur.fetchall()

                cur.execute(
                    """
                    SELECT movie_id, rating, created_at
                    FROM reviews
                    WHERE user_id = %s
                    ORDER BY created_at DESC
                    LIMIT 40
                    """,
                    (user_id,),
                )
                review_rows = cur.fetchall()

        # Seen set = watched ∪ reviewed
        for mid, _ in watched_rows:
            seen_ids.add(int(mid))
        for mid, _rating, _ in review_rows:
            seen_ids.add(int(mid))

        # Seed movies for similar-item boosts: recent highly-rated first, then recent watched.
        for idx, (mid, rating, _ts) in enumerate(review_rows):
            if rating is None or float(rating) < 4.0:
                continue
            movie_id = int(mid)
            if movie_id not in seed_ids:
                seed_ids.append(movie_id)

            # Genre affinity from high ratings with light recency decay.
            w = (float(rating) - 3.0) / (1.0 + 0.12 * idx)
            for genre in self._extract_movie_genres(movie_id):
                genre_pref[genre] += w

        for idx, (mid, _ts) in enumerate(watched_rows):
            movie_id = int(mid)
            if movie_id not in seed_ids:
                seed_ids.append(movie_id)
            # Weak positive signal from recent watched actions.
            w = 0.25 / (1.0 + 0.10 * idx)
            for genre in self._extract_movie_genres(movie_id):
                genre_pref[genre] += w

        return {
            "seen_ids": seen_ids,
            "seed_ids": seed_ids[:12],
            "genre_pref": dict(genre_pref),
        }

    def _build_similar_boost(self, seed_ids: List[int]) -> Dict[int, float]:
        if not seed_ids:
            return {}

        sim = get_similarity_service()
        boost: Dict[int, float] = defaultdict(float)

        for seed_rank, seed_id in enumerate(seed_ids):
            similar_ids = sim.get_similar_ids(seed_id, n=30)
            if not similar_ids:
                continue
            seed_weight = 1.0 / (1.0 + 0.25 * seed_rank)
            for rank, cand_id in enumerate(similar_ids, start=1):
                boost[int(cand_id)] += seed_weight * (1.0 / rank)

        if not boost:
            return {}

        mx = max(boost.values())
        if mx <= 0:
            return {}
        return {k: v / mx for k, v in boost.items()}

    def _apply_online_post_ranking(
        self,
        user_id: int,
        movies: List[dict],
        n: int,
        exclude_seen: bool,
    ) -> List[dict]:
        """Cheap quasi-live rerank over static model outputs.

        Strategy:
          1) Exclude already watched/reviewed items from app DB.
          2) Boost candidates similar to recent positive interactions.
          3) Boost candidates matching genres from recent high ratings.
        """
        if not movies:
            return movies

        try:
            sig = self._load_user_live_signals(user_id)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Live signals unavailable for user %s: %s", user_id, exc)
            return movies[:n]

        seen_ids: Set[int] = sig["seen_ids"]
        seed_ids: List[int] = sig["seed_ids"]
        genre_pref: Dict[str, float] = sig["genre_pref"]

        filtered = [m for m in movies if int(m.get("id", -1)) not in seen_ids] if exclude_seen else list(movies)
        if not filtered:
            return []

        similar_boost = self._build_similar_boost(seed_ids)

        # Genre boost per candidate (normalized later)
        raw_genre_boost: Dict[int, float] = {}
        for m in filtered:
            mid = int(m["id"])
            genres = self._extract_movie_genres(mid)
            if not genres:
                raw_genre_boost[mid] = 0.0
                continue
            raw_genre_boost[mid] = sum(genre_pref.get(g, 0.0) for g in genres)

        max_g = max((v for v in raw_genre_boost.values() if v > 0), default=0.0)

        # Normalize base score by min-max to keep boosts impactful regardless of score scale.
        base_scores = [float(m.get("score", 0.0)) for m in filtered]
        b_min, b_max = min(base_scores), max(base_scores)
        b_rng = b_max - b_min

        reranked: List[tuple[float, dict]] = []
        for m in filtered:
            mid = int(m["id"])
            b = float(m.get("score", 0.0))
            b_norm = (b - b_min) / b_rng if b_rng > 1e-12 else 0.0
            s_boost = similar_boost.get(mid, 0.0)
            g_boost = (raw_genre_boost.get(mid, 0.0) / max_g) if max_g > 0 else 0.0

            # Weighted blend tuned for low-risk behaviour shift.
            online_rank_score = 0.65 * b_norm + 0.25 * s_boost + 0.10 * g_boost
            reranked.append((online_rank_score, m))

        reranked.sort(key=lambda x: x[0], reverse=True)
        return [m for _, m in reranked[:n]]

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
                    movies = self._apply_online_post_ranking(
                        user_id=user_id,
                        movies=movies,
                        n=n,
                        exclude_seen=exclude_seen,
                    )
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
            pop_movies = self._apply_online_post_ranking(
                user_id=user_id,
                movies=pop_movies,
                n=n,
                exclude_seen=exclude_seen,
            )
            return pop_movies, "popularity_fallback"

        return [], "popularity_fallback"


# ── Module-level singleton ─────────────────────────────────────────────────
@lru_cache(maxsize=1)
def get_recommender_service() -> RecommenderService:
    """FastAPI dependency — returns the shared RecommenderService instance."""
    return RecommenderService()
