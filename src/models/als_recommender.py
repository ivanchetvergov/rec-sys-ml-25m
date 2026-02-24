"""Implicit ALS (iALS) Recommender — Stage 1 Candidate Generator.

Implements the Hu, Koren & Volinsky (2008) "Collaborative Filtering for
Implicit Feedback Datasets" algorithm via the `implicit` library.

The model learns user and item latent factors by treating ratings as
confidence-weighted binary preferences:
    - Preference p_ui = 1 (user u has interacted with item i)
    - Confidence c_ui = 1 + α · f(r_ui)  where f is the confidence mode

Supported confidence modes:
    'linear' : c_ui = 1 + alpha * r_ui          (standard iALS)
    'log'    : c_ui = 1 + alpha * log1p(r_ui)   (log-confidence variant)
    'binary' : c_ui = alpha for all rated items  (ignores rating scale)

Usage:
    model = ALSRecommender(factors=128, iterations=20, alpha=15.0)
    model.fit(train_df)
    candidates, scores = model.recommend_with_scores(user_id=123, n=500)
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

logger = logging.getLogger(__name__)

ConfidenceMode = Literal["linear", "log", "binary"]


class ALSRecommender:
    """Implicit ALS (iALS) candidate generator.

    Args:
        factors: Number of latent factors.
        iterations: Number of ALS iterations.
        regularization: L2 regularization coefficient.
        alpha: Confidence scaling factor.
        confidence_mode: How to compute confidence from ratings.
            'linear'  → c = 1 + alpha * rating
            'log'     → c = 1 + alpha * log1p(rating)
            'binary'  → c = alpha for all rated items
        random_state: Random seed (sets implicit's random_state).
        num_threads: CPU threads for implicit (0 = all cores).
    """

    def __init__(
        self,
        factors: int = 128,
        iterations: int = 20,
        regularization: float = 0.01,
        alpha: float = 15.0,
        confidence_mode: ConfidenceMode = "linear",
        random_state: int = 42,
        num_threads: int = 0,
    ) -> None:
        self.factors = factors
        self.iterations = iterations
        self.regularization = regularization
        self.alpha = alpha
        self.confidence_mode = confidence_mode
        self.random_state = random_state
        self.num_threads = num_threads
        self.is_fitted_: bool = False

        self._model = None
        self._user_item_matrix: Optional[csr_matrix] = None
        self.user_id_map_: Optional[Dict[int, int]] = None
        self.item_id_map_: Optional[Dict[int, int]] = None
        self.idx_to_user_: Optional[Dict[int, int]] = None
        self.idx_to_item_: Optional[Dict[int, int]] = None
        self.n_users_: int = 0
        self.n_items_: int = 0

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_df: pd.DataFrame,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating",
    ) -> "ALSRecommender":
        """Fit iALS on a user-item interactions DataFrame.

        Args:
            train_df: Interactions with at minimum user_col, item_col, rating_col.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.
            rating_col: Column name for ratings.

        Returns:
            self
        """
        try:
            from implicit.als import AlternatingLeastSquares
        except ImportError:
            raise ImportError(
                "implicit is required: pip install implicit"
            )

        logger.info("Building confidence matrix for iALS...")

        users = train_df[user_col].unique()
        items = train_df[item_col].unique()
        n_users, n_items = len(users), len(items)

        logger.info(f"Matrix shape: {n_users:,} users × {n_items:,} items")

        self.user_id_map_ = {int(u): i for i, u in enumerate(users)}
        self.item_id_map_ = {int(v): i for i, v in enumerate(items)}
        self.idx_to_user_ = {i: int(u) for u, i in self.user_id_map_.items()}
        self.idx_to_item_ = {i: int(v) for v, i in self.item_id_map_.items()}

        row = train_df[user_col].map(self.user_id_map_).values
        col = train_df[item_col].map(self.item_id_map_).values
        ratings = train_df[rating_col].values.astype(np.float32)

        confidence = self._compute_confidence(ratings)

        self._user_item_matrix = csr_matrix(
            (confidence, (row, col)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(
            f"Confidence matrix: density="
            f"{100.0 * self._user_item_matrix.nnz / (n_users * n_items):.4f}%"
        )

        self._model = AlternatingLeastSquares(
            factors=self.factors,
            iterations=self.iterations,
            regularization=self.regularization,
            random_state=self.random_state,
            num_threads=self.num_threads,
            use_gpu=False,
        )

        logger.info(
            f"Training iALS: factors={self.factors}, "
            f"iterations={self.iterations}, alpha={self.alpha}, "
            f"confidence_mode={self.confidence_mode}"
        )
        self._model.fit(self._user_item_matrix)

        self.n_users_ = n_users
        self.n_items_ = n_items
        self.is_fitted_ = True

        logger.info(
            f"iALS complete — user_factors: {self._model.user_factors.shape}, "
            f"item_factors: {self._model.item_factors.shape}"
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend_with_scores(
        self,
        user_id: int,
        n: int = 500,
        exclude_items: Optional[Set[int]] = None,
        filter_already_liked: bool = False,
    ) -> Tuple[List[int], List[float]]:
        """Return top-N candidates with their ALS scores.

        Args:
            user_id: Original user ID.
            n: Number of candidates.
            exclude_items: Item IDs to exclude from results.
            filter_already_liked: If True, suppress items seen in training.

        Returns:
            (item_ids, scores) — both ordered best-first.
        """
        self._check_fitted()

        user_idx = self.user_id_map_.get(int(user_id))
        if user_idx is None:
            return [], []

        user_row = self._user_item_matrix[user_idx]
        item_ids_arr, scores_arr = self._model.recommend(
            user_idx,
            user_row,
            N=n + (len(exclude_items) if exclude_items else 0) + 50,
            filter_already_liked_items=filter_already_liked,
        )

        result_ids: List[int] = []
        result_scores: List[float] = []
        for idx, score in zip(item_ids_arr, scores_arr):
            item_id = self.idx_to_item_[int(idx)]
            if exclude_items and item_id in exclude_items:
                continue
            result_ids.append(item_id)
            result_scores.append(float(score))
            if len(result_ids) >= n:
                break

        return result_ids, result_scores

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """Return top-N recommended item IDs (scores discarded).

        Cold-start users not seen during training return [].
        """
        ids, _ = self.recommend_with_scores(
            user_id, n=n, exclude_items=exclude_items, filter_already_liked=True
        )
        return ids

    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = 10,
        exclude_items_per_user: Optional[Dict[int, Set[int]]] = None,
    ) -> Dict[int, List[int]]:
        """Batch recommendations; cold-start users return []."""
        self._check_fitted()
        result: Dict[int, List[int]] = {}
        for user_id in user_ids:
            exclude = None
            if exclude_items_per_user:
                exclude = exclude_items_per_user.get(int(user_id))
            result[int(user_id)] = self.recommend(user_id, n=n, exclude_items=exclude)
        return result

    def similar_items(self, item_id: int, n: int = 20) -> Tuple[List[int], List[float]]:
        """Return items most similar to item_id by latent factor cosine similarity.

        Useful for item-based explanations: "because you watched X".
        """
        self._check_fitted()
        item_idx = self.item_id_map_.get(int(item_id))
        if item_idx is None:
            return [], []
        ids_arr, scores_arr = self._model.similar_items(item_idx, N=n + 1)
        result_ids = [self.idx_to_item_[int(i)] for i in ids_arr if int(i) != item_idx][:n]
        result_scores = [float(s) for s in scores_arr[: len(result_ids)]]
        return result_ids, result_scores

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model to directory.

        Artifacts:
        - ``implicit_model.pkl``: serialized implicit model
        - ``user_item_matrix.pkl``: confidence matrix (needed for recommend)
        - ``id_maps.pkl``: user/item index mappings
        - ``model_config.json``: hyper-parameters and sizes
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / "implicit_model.pkl", "wb") as f:
            pickle.dump(self._model, f)

        with open(path / "user_item_matrix.pkl", "wb") as f:
            pickle.dump(self._user_item_matrix, f)

        id_maps = {
            "user_id_map": self.user_id_map_,
            "item_id_map": self.item_id_map_,
            "idx_to_user": self.idx_to_user_,
            "idx_to_item": self.idx_to_item_,
        }
        with open(path / "id_maps.pkl", "wb") as f:
            pickle.dump(id_maps, f)

        config = {
            "model_type": "ALSRecommender",
            "factors": self.factors,
            "iterations": self.iterations,
            "regularization": self.regularization,
            "alpha": self.alpha,
            "confidence_mode": self.confidence_mode,
            "random_state": self.random_state,
            "n_users": self.n_users_,
            "n_items": self.n_items_,
        }
        with open(path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"ALSRecommender saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "ALSRecommender":
        """Load ALSRecommender from directory written by :meth:`save`."""
        path = Path(path)
        with open(path / "model_config.json") as f:
            config = json.load(f)

        model = cls(
            factors=config["factors"],
            iterations=config["iterations"],
            regularization=config["regularization"],
            alpha=config["alpha"],
            confidence_mode=config["confidence_mode"],
            random_state=config["random_state"],
        )

        with open(path / "implicit_model.pkl", "rb") as f:
            model._model = pickle.load(f)

        with open(path / "user_item_matrix.pkl", "rb") as f:
            model._user_item_matrix = pickle.load(f)

        with open(path / "id_maps.pkl", "rb") as f:
            id_maps = pickle.load(f)

        model.user_id_map_ = id_maps["user_id_map"]
        model.item_id_map_ = id_maps["item_id_map"]
        model.idx_to_user_ = id_maps["idx_to_user"]
        model.idx_to_item_ = id_maps["idx_to_item"]
        model.n_users_ = config["n_users"]
        model.n_items_ = config["n_items"]
        model.is_fitted_ = True

        logger.info(
            f"ALSRecommender loaded from {path} "
            f"({model.n_users_:,} users, {model.n_items_:,} items, "
            f"factors={model.factors})"
        )
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _compute_confidence(self, ratings: np.ndarray) -> np.ndarray:
        if self.confidence_mode == "linear":
            return (1.0 + self.alpha * ratings).astype(np.float32)
        elif self.confidence_mode == "log":
            return (1.0 + self.alpha * np.log1p(ratings)).astype(np.float32)
        elif self.confidence_mode == "binary":
            return np.full_like(ratings, fill_value=self.alpha, dtype=np.float32)
        else:
            raise ValueError(
                f"Unknown confidence_mode '{self.confidence_mode}'. "
                "Use 'linear', 'log', or 'binary'."
            )

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        details = ""
        if self.is_fitted_:
            details = (
                f", {self.n_users_:,} users, {self.n_items_:,} items, "
                f"factors={self.factors}, mode={self.confidence_mode}"
            )
        return f"ALSRecommender({status}{details})"
