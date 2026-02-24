"""Two-Stage Recommender: ALS Candidate Generation + CatBoost Re-ranking.

Pipeline:
    1. ALSRecommender  →  top-N candidates per user  (~1-2ms)
    2. CatBoostRanker  →  re-rank candidates to top-K  (~5-10ms)

For cold-start users (unseen during ALS training) the pipeline falls back
to PopularityRecommender.

Usage:
    rec = TwoStageRecommender(
        candidate_model=als_model,
        ranker=catboost_ranker,
        user_features=user_feat_df,   # one row per user
        item_features=item_feat_df,   # one row per item
    )
    results = rec.recommend(user_id=123, n=10, n_candidates=200)
    # [{'item_id': 42, 'score': 0.83, 'explanation': {'movie_popularity': 0.41, ...}}, ...]
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.models.als_recommender import ALSRecommender
from src.models.catboost_ranker import CatBoostRanker, RANKER_FEATURE_COLS

logger = logging.getLogger(__name__)


class TwoStageRecommender:
    """Full inference pipeline: ALS retrieval → CatBoost ranking.

    Args:
        candidate_model: Fitted ALSRecommender (stage 1).
        ranker: Fitted CatBoostRanker (stage 2).
        user_features: DataFrame indexed by userId, columns = USER_FEATURE_COLS.
        item_features: DataFrame indexed by movieId, columns = ITEM_FEATURE_COLS.
        fallback_items: Ordered list of popular item IDs for cold-start fallback.
    """

    def __init__(
        self,
        candidate_model: ALSRecommender,
        ranker: CatBoostRanker,
        user_features: pd.DataFrame,
        item_features: pd.DataFrame,
        fallback_items: Optional[List[int]] = None,
    ) -> None:
        self.candidate_model = candidate_model
        self.ranker = ranker
        self.user_features = user_features
        self.item_features = item_features
        self.fallback_items = fallback_items or []

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        n_candidates: int = 200,
        exclude_items: Optional[set] = None,
        explain: bool = False,
    ) -> List[Dict]:
        """Recommend top-N items for a user.

        Args:
            user_id: Target user.
            n: Final number of recommendations.
            n_candidates: Stage-1 candidate list size.
            exclude_items: Item IDs to suppress (e.g. already seen).
            explain: If True, attach SHAP explanation to each result.

        Returns:
            List of dicts ordered by score descending:
                {'item_id': int, 'score': float,
                 'explanation': dict}   ← only when explain=True
        """
        # --- Stage 1: retrieve candidates ---
        candidate_ids, als_scores = self.candidate_model.recommend_with_scores(
            user_id=user_id,
            n=n_candidates,
            exclude_items=exclude_items,
            filter_already_liked=True,
        )

        # Cold-start fallback
        if not candidate_ids:
            logger.debug(f"Cold-start user {user_id}: using fallback list")
            fallback = [i for i in self.fallback_items if not (exclude_items and i in exclude_items)]
            return [{"item_id": i, "score": 0.0} for i in fallback[:n]]

        # --- Build feature matrix ---
        X, valid_ids, valid_scores = self._build_feature_matrix(
            user_id, candidate_ids, als_scores
        )
        if X is None or len(valid_ids) == 0:
            return [{"item_id": i, "score": s} for i, s in zip(candidate_ids[:n], als_scores[:n])]

        # --- Stage 2: re-rank ---
        scores = self.ranker.predict(X)
        order = np.argsort(scores)[::-1]

        results = []
        for rank_idx in order[:n]:
            entry: Dict = {
                "item_id": valid_ids[rank_idx],
                "score": round(float(scores[rank_idx]), 6),
            }
            if explain:
                shap_row = self.ranker.explain(X.iloc[[rank_idx]])
                entry["explanation"] = shap_row[0]
            results.append(entry)

        return results

    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = 10,
        n_candidates: int = 200,
        exclude_items_per_user: Optional[Dict[int, set]] = None,
    ) -> Dict[int, List[Dict]]:
        """Batch inference for multiple users.

        Returns:
            Dict[user_id -> list of result dicts]
        """
        results: Dict[int, List[Dict]] = {}
        for user_id in user_ids:
            exclude = None
            if exclude_items_per_user:
                exclude = exclude_items_per_user.get(int(user_id))
            results[int(user_id)] = self.recommend(
                user_id=user_id,
                n=n,
                n_candidates=n_candidates,
                exclude_items=exclude,
            )
        return results

    # ------------------------------------------------------------------
    # Feature matrix construction
    # ------------------------------------------------------------------

    def _build_feature_matrix(
        self,
        user_id: int,
        candidate_ids: List[int],
        als_scores: List[float],
    ) -> Tuple[Optional[pd.DataFrame], List[int], List[float]]:
        """Build ranker input features for (user, candidate_ids) pairs.

        Returns (X_df, valid_item_ids, valid_als_scores).
        Returns (None, [], []) if user features are missing.
        """
        uid = int(user_id)

        # User features
        if uid not in self.user_features.index:
            return None, [], []
        uf = self.user_features.loc[uid]

        # Item features — filter to items in item_features index
        valid_ids: List[int] = []
        valid_scores: List[float] = []
        for item_id, score in zip(candidate_ids, als_scores):
            if item_id in self.item_features.index:
                valid_ids.append(item_id)
                valid_scores.append(score)

        if not valid_ids:
            return None, [], []

        itf = self.item_features.loc[valid_ids]

        # Broadcast user features across all candidate rows
        user_df = pd.DataFrame([uf.values] * len(valid_ids), columns=uf.index, index=itf.index)

        X = pd.concat([user_df, itf], axis=1)
        X["als_score"] = valid_scores

        # Ensure column order matches training
        feature_cols = [c for c in RANKER_FEATURE_COLS if c in X.columns]
        return X[feature_cols].reset_index(drop=True), valid_ids, valid_scores

    # ------------------------------------------------------------------
    # Helpers — build feature lookup tables from feature store splits
    # ------------------------------------------------------------------

    @staticmethod
    def build_feature_tables(
        train_df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Extract per-user and per-item feature tables from train split.

        These are passed to the TwoStageRecommender constructor.

        Args:
            train_df: Training split from the feature store.

        Returns:
            (user_features_df, item_features_df)
            user_features_df — indexed by userId
            item_features_df — indexed by movieId
        """
        from src.models.catboost_ranker import USER_FEATURE_COLS, ITEM_FEATURE_COLS

        user_cols = [c for c in USER_FEATURE_COLS if c in train_df.columns]
        item_cols = [c for c in ITEM_FEATURE_COLS if c in train_df.columns]

        user_features = (
            train_df[["userId"] + user_cols]
            .drop_duplicates("userId")
            .set_index("userId")
        )
        item_features = (
            train_df[["movieId"] + item_cols]
            .drop_duplicates("movieId")
            .set_index("movieId")
        )
        return user_features, item_features

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save both sub-models and feature tables to directory.

        Layout:
            <path>/
                als/          ← ALSRecommender artifacts
                ranker/       ← CatBoostRanker artifacts
                user_features.parquet
                item_features.parquet
                pipeline_config.json
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.candidate_model.save(path / "als")
        self.ranker.save(path / "ranker")
        self.user_features.to_parquet(path / "user_features.parquet")
        self.item_features.to_parquet(path / "item_features.parquet")

        config = {
            "model_type": "TwoStageRecommender",
            "n_users": self.candidate_model.n_users_,
            "n_items": self.candidate_model.n_items_,
            "n_user_features": len(self.user_features.columns),
            "n_item_features": len(self.item_features.columns),
        }
        with open(path / "pipeline_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"TwoStageRecommender saved to {path}")

    @classmethod
    def load(cls, path: Path, fallback_items: Optional[List[int]] = None) -> "TwoStageRecommender":
        """Load TwoStageRecommender from directory saved by :meth:`save`."""
        path = Path(path)

        candidate_model = ALSRecommender.load(path / "als")
        ranker = CatBoostRanker.load(path / "ranker")
        user_features = pd.read_parquet(path / "user_features.parquet")
        item_features = pd.read_parquet(path / "item_features.parquet")

        logger.info(f"TwoStageRecommender loaded from {path}")
        return cls(
            candidate_model=candidate_model,
            ranker=ranker,
            user_features=user_features,
            item_features=item_features,
            fallback_items=fallback_items,
        )

    def __repr__(self) -> str:
        return (
            f"TwoStageRecommender("
            f"stage1={self.candidate_model!r}, "
            f"stage2={self.ranker!r})"
        )
