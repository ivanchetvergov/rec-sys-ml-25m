"""Collaborative Filtering recommender using Truncated SVD (Matrix Factorization).

Uses scipy.sparse.linalg.svds on the user-item ratings matrix to learn
latent user and item embeddings. Scores are computed on-the-fly per user
to keep memory usage predictable at scale.

Usage:
    model = SVDRecommender(factors=100, random_state=42)
    model.fit(train_df)
    recs = model.recommend(user_id=123, n=10, exclude_items={1, 2, 3})
"""
import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import svds

logger = logging.getLogger(__name__)


class SVDRecommender:
    """Matrix Factorization recommender via Truncated SVD.

    Decomposes the centered user-item ratings matrix R_centered ≈ U Σ Vᵀ,
    then scores an unseen (user, item) pair as:
        score(u, i) = μ_u + (U[u] * Σ) · V[i]

    where μ_u is the user's mean rating (used for centering).

    Args:
        factors: Number of latent factors (rank of decomposition).
        random_state: Random seed passed to svds via v0.
    """

    def __init__(self, factors: int = 100, random_state: int = 42) -> None:
        self.factors = factors
        self.random_state = random_state
        self.is_fitted_: bool = False

        # Learned attributes (set by fit)
        self.user_factors_: Optional[np.ndarray] = None   # (n_users, k)
        self.item_factors_: Optional[np.ndarray] = None   # (n_items, k)
        self.user_mean_: Optional[pd.Series] = None       # index=userId
        self.user_id_map_: Optional[Dict[int, int]] = None
        self.item_id_map_: Optional[Dict[int, int]] = None
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
    ) -> "SVDRecommender":
        """Fit SVD on the user-item ratings matrix.

        Args:
            train_df: DataFrame with user-item-rating interactions.
            user_col: Column name for user IDs.
            item_col: Column name for item IDs.
            rating_col: Column name for ratings.

        Returns:
            self (for chaining)
        """
        logger.info("Building user-item matrix for SVD...")

        users = train_df[user_col].unique()
        items = train_df[item_col].unique()
        n_users, n_items = len(users), len(items)

        logger.info(f"Matrix shape: {n_users:,} users × {n_items:,} items")

        # Build index mappings
        self.user_id_map_ = {int(uid): idx for idx, uid in enumerate(users)}
        self.item_id_map_ = {int(iid): idx for idx, iid in enumerate(items)}
        self.idx_to_item_ = {idx: int(iid) for iid, idx in self.item_id_map_.items()}

        # Center ratings by user mean to remove user rating bias
        user_means = train_df.groupby(user_col)[rating_col].mean()
        self.user_mean_ = user_means.rename(index=int)

        user_mean_vec = train_df[user_col].map(user_means).values
        centered_ratings = (train_df[rating_col].values - user_mean_vec).astype(np.float32)

        # Build sparse CSR matrix
        row = train_df[user_col].map(self.user_id_map_).values
        col = train_df[item_col].map(self.item_id_map_).values
        matrix = csr_matrix(
            (centered_ratings, (row, col)),
            shape=(n_users, n_items),
            dtype=np.float32,
        )

        logger.info(
            f"Sparse matrix density: "
            f"{100.0 * matrix.nnz / (n_users * n_items):.4f}% "
            f"({matrix.nnz:,} non-zeros)"
        )

        # Truncated SVD — k must be < min(rows, cols)
        k = min(self.factors, min(n_users, n_items) - 1)
        if k < self.factors:
            logger.warning(
                f"Requested {self.factors} factors but matrix only supports "
                f"{k}. Using k={k}."
            )

        logger.info(f"Running svds with k={k}...")
        rng = np.random.default_rng(self.random_state)
        v0 = rng.standard_normal(min(n_users, n_items))

        U, sigma, Vt = svds(matrix, k=k, v0=v0)

        # svds returns singular values in ascending order — reverse to descending
        order = np.argsort(sigma)[::-1]
        sigma = sigma[order]
        U = U[:, order]
        Vt = Vt[order, :]

        # Store scaled user factors (U * Σ) and item factors (Vᵀ)ᵀ
        # score(u, i) = user_factors_[u] · item_factors_[i]
        self.user_factors_ = (U * sigma[np.newaxis, :]).astype(np.float32)  # (n_users, k)
        self.item_factors_ = Vt.T.astype(np.float32)                        # (n_items, k)
        self.n_users_ = n_users
        self.n_items_ = n_items
        self.is_fitted_ = True

        logger.info(
            f"SVD complete. user_factors shape: {self.user_factors_.shape}, "
            f"item_factors shape: {self.item_factors_.shape}"
        )
        logger.info(
            f"Top singular values: {sigma[:5].round(2).tolist()}"
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def recommend(
        self,
        user_id: int,
        n: int = 10,
        exclude_items: Optional[Set[int]] = None,
    ) -> List[int]:
        """Generate top-N personalized recommendations for a user.

        For cold-start users (not seen during training) an empty list is
        returned — the caller should fall back to a popularity baseline.

        Args:
            user_id: User ID.
            n: Number of recommendations.
            exclude_items: Set of item IDs to exclude (e.g. already seen).

        Returns:
            Ordered list of recommended item IDs (best first).
        """
        self._check_fitted()

        user_idx = self.user_id_map_.get(int(user_id))
        if user_idx is None:
            logger.debug(f"Cold-start user {user_id}: returning empty list.")
            return []

        # Compute scores for all items: (k,) · (n_items, k)ᵀ = (n_items,)
        scores = self.user_factors_[user_idx] @ self.item_factors_.T

        # Sort descending
        sorted_indices = np.argsort(scores)[::-1]

        recs: List[int] = []
        for item_idx in sorted_indices:
            item_id = self.idx_to_item_[int(item_idx)]
            if exclude_items and item_id in exclude_items:
                continue
            recs.append(item_id)
            if len(recs) >= n:
                break

        return recs

    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = 10,
        exclude_items_per_user: Optional[Dict[int, Set[int]]] = None,
    ) -> Dict[int, List[int]]:
        """Generate recommendations for multiple users.

        Args:
            user_ids: List of user IDs.
            n: Number of recommendations per user.
            exclude_items_per_user: Dict[user_id -> set of items to exclude].

        Returns:
            Dict[user_id -> list of recommended item IDs].
        """
        self._check_fitted()

        recommendations: Dict[int, List[int]] = {}
        for user_id in user_ids:
            exclude = None
            if exclude_items_per_user:
                exclude = exclude_items_per_user.get(int(user_id))
            recommendations[int(user_id)] = self.recommend(user_id, n=n, exclude_items=exclude)

        return recommendations

    def score(self, user_id: int, item_id: int) -> float:
        """Predict the rating score for a (user, item) pair.

        Adds back the user's mean rating so the output is on the original
        rating scale ≈ [0.5, 5.0].

        Args:
            user_id: User ID.
            item_id: Item ID.

        Returns:
            Predicted score. Returns the global mean for unknown users/items.
        """
        self._check_fitted()

        user_idx = self.user_id_map_.get(int(user_id))
        item_idx = self.item_id_map_.get(int(item_id))

        if user_idx is None or item_idx is None:
            return float(self.user_mean_.mean())

        latent_score = float(
            self.user_factors_[user_idx] @ self.item_factors_[item_idx]
        )
        user_bias = float(self.user_mean_.get(int(user_id), self.user_mean_.mean()))
        return latent_score + user_bias

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model artifacts to directory.

        Saves:
        - ``user_factors.npy``: scaled user latent matrix (n_users, k)
        - ``item_factors.npy``: item latent matrix (n_items, k)
        - ``id_maps.pkl``: user/item index mappings and user means
        - ``model_config.json``: hyper-parameters and sizes

        Args:
            path: Directory path. Created if it does not exist.
        """
        self._check_fitted()

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        np.save(path / "user_factors.npy", self.user_factors_)
        np.save(path / "item_factors.npy", self.item_factors_)

        id_maps = {
            "user_id_map": self.user_id_map_,
            "item_id_map": self.item_id_map_,
            "idx_to_item": self.idx_to_item_,
            "user_mean": self.user_mean_.to_dict(),
        }
        with open(path / "id_maps.pkl", "wb") as f:
            pickle.dump(id_maps, f)

        config = {
            "model_type": "SVDRecommender",
            "factors": self.factors,
            "random_state": self.random_state,
            "n_users": self.n_users_,
            "n_items": self.n_items_,
            "actual_factors": int(self.user_factors_.shape[1]),
        }
        with open(path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"SVDRecommender saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "SVDRecommender":
        """Load a saved SVDRecommender from directory.

        Args:
            path: Directory path written by :meth:`save`.

        Returns:
            Loaded SVDRecommender instance.
        """
        path = Path(path)

        with open(path / "model_config.json", "r") as f:
            config = json.load(f)

        model = cls(
            factors=config["factors"],
            random_state=config["random_state"],
        )

        model.user_factors_ = np.load(path / "user_factors.npy")
        model.item_factors_ = np.load(path / "item_factors.npy")

        with open(path / "id_maps.pkl", "rb") as f:
            id_maps = pickle.load(f)

        model.user_id_map_ = id_maps["user_id_map"]
        model.item_id_map_ = id_maps["item_id_map"]
        model.idx_to_item_ = id_maps["idx_to_item"]
        model.user_mean_ = pd.Series(id_maps["user_mean"])
        model.n_users_ = config["n_users"]
        model.n_items_ = config["n_items"]
        model.is_fitted_ = True

        logger.info(
            f"SVDRecommender loaded from {path} "
            f"({model.n_users_:,} users, {model.n_items_:,} items, "
            f"k={model.user_factors_.shape[1]})"
        )
        return model

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        details = ""
        if self.is_fitted_:
            k = self.user_factors_.shape[1]
            details = f", {self.n_users_:,} users, {self.n_items_:,} items, k={k}"
        return f"SVDRecommender({status}{details})"
