"""Popularity-based recommender model (baseline)."""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PopularityRecommender:
    """
    Popularity-based recommender (non-personalized baseline).

    Recommends items sorted by popularity score:
        popularity = avg_rating * log(1 + num_ratings)

    This is a simple but strong baseline for recommender systems.
    """

    def __init__(
        self,
        min_ratings: int = 10,
        rating_weight: float = 1.0,
        count_weight: float = 1.0
    ):
        """
        Initialize PopularityRecommender.

        Args:
            min_ratings: Minimum number of ratings for an item to be recommended
            rating_weight: Weight for average rating in popularity score
            count_weight: Weight for log(count) in popularity score
        """
        self.min_ratings = min_ratings
        self.rating_weight = rating_weight
        self.count_weight = count_weight

        # Will be filled during fit()
        self.item_scores_: Optional[pd.Series] = None
        self.item_stats_: Optional[pd.DataFrame] = None
        self.is_fitted_ = False

        logger.info(
            f"PopularityRecommender initialized: "
            f"min_ratings={min_ratings}, "
            f"rating_weight={rating_weight}, "
            f"count_weight={count_weight}"
        )

    def fit(
        self,
        train_df: pd.DataFrame,
        user_col: str = "userId",
        item_col: str = "movieId",
        rating_col: str = "rating"
    ) -> "PopularityRecommender":
        """
        Fit the model by calculating popularity scores for all items.

        Args:
            train_df: Training dataframe with user-item interactions
            user_col: Column name for user ID
            item_col: Column name for item ID
            rating_col: Column name for rating

        Returns:
            self
        """
        logger.info(f"Fitting PopularityRecommender on {len(train_df):,} interactions...")

        # Calculate item statistics
        item_stats = train_df.groupby(item_col).agg({
            rating_col: ["mean", "std", "count"],
            user_col: "nunique"
        }).reset_index()

        item_stats.columns = [item_col, "avg_rating", "std_rating", "num_ratings", "num_users"]

        # Filter by minimum ratings
        item_stats = item_stats[item_stats["num_ratings"] >= self.min_ratings].copy()

        # Calculate popularity score
        # popularity = avg_rating * log(1 + num_ratings)
        item_stats["popularity_score"] = (
            self.rating_weight * item_stats["avg_rating"] +
            self.count_weight * np.log1p(item_stats["num_ratings"])
        )

        # Sort by popularity
        item_stats = item_stats.sort_values("popularity_score", ascending=False)

        # Store results
        self.item_stats_ = item_stats
        self.item_scores_ = pd.Series(
            item_stats["popularity_score"].values,
            index=item_stats[item_col].values
        )
        self.is_fitted_ = True

        logger.info(
            f"Fitted on {len(item_stats):,} items "
            f"(filtered from {train_df[item_col].nunique():,})"
        )
        logger.info(
            f"Top 5 items: {item_stats[item_col].head().tolist()}"
        )

        return self

    def recommend(
        self,
        user_id: Optional[int] = None,
        n: int = 10,
        exclude_items: Optional[Set[int]] = None
    ) -> List[int]:
        """
        Generate top-N recommendations.

        Note: This is non-personalized, user_id is ignored.

        Args:
            user_id: User ID (ignored for popularity-based)
            n: Number of recommendations to generate
            exclude_items: Set of item IDs to exclude from recommendations

        Returns:
            List of recommended item IDs (sorted by popularity)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        # Get all items sorted by popularity
        items = self.item_stats_["movieId"].values

        # Filter excluded items
        if exclude_items:
            items = [item for item in items if item not in exclude_items]

        # Return top N
        return items[:n]

    def recommend_batch(
        self,
        user_ids: List[int],
        n: int = 10,
        exclude_items_per_user: Optional[Dict[int, Set[int]]] = None
    ) -> Dict[int, List[int]]:
        """
        Generate recommendations for multiple users.

        Args:
            user_ids: List of user IDs
            n: Number of recommendations per user
            exclude_items_per_user: Dict[user_id -> set of items to exclude]

        Returns:
            Dict[user_id -> list of recommended item IDs]
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        recommendations = {}

        for user_id in user_ids:
            exclude = None
            if exclude_items_per_user and user_id in exclude_items_per_user:
                exclude = exclude_items_per_user[user_id]

            recommendations[user_id] = self.recommend(
                user_id=user_id,
                n=n,
                exclude_items=exclude
            )

        return recommendations

    def get_item_score(self, item_id: int) -> float:
        """
        Get popularity score for a specific item.

        Args:
            item_id: Item ID

        Returns:
            Popularity score (or 0.0 if item not in catalog)
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.item_scores_.get(item_id, 0.0)

    def get_top_items(self, n: int = 100) -> pd.DataFrame:
        """
        Get top N items with their statistics.

        Args:
            n: Number of top items to return

        Returns:
            DataFrame with top items and their stats
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

        return self.item_stats_.head(n).copy()

    def save(self, path: Path) -> None:
        """
        Save model to disk.

        Args:
            path: Directory path to save model
        """
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Cannot save unfitted model.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save item statistics
        self.item_stats_.to_csv(path / "item_stats.csv", index=False)

        # Save model config
        config = {
            "model_type": "PopularityRecommender",
            "min_ratings": self.min_ratings,
            "rating_weight": self.rating_weight,
            "count_weight": self.count_weight,
            "n_items": len(self.item_stats_)
        }

        with open(path / "model_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"Model saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "PopularityRecommender":
        """
        Load model from disk.

        Args:
            path: Directory path to load model from

        Returns:
            Loaded PopularityRecommender instance
        """
        path = Path(path)

        # Load config
        with open(path / "model_config.json", "r") as f:
            config = json.load(f)

        # Create instance
        model = cls(
            min_ratings=config["min_ratings"],
            rating_weight=config["rating_weight"],
            count_weight=config["count_weight"]
        )

        # Load item statistics
        model.item_stats_ = pd.read_csv(path / "item_stats.csv")
        model.item_scores_ = pd.Series(
            model.item_stats_["popularity_score"].values,
            index=model.item_stats_["movieId"].values
        )
        model.is_fitted_ = True

        logger.info(f"Model loaded from {path}")
        return model

    def __repr__(self) -> str:
        """String representation."""
        fitted_str = "fitted" if self.is_fitted_ else "not fitted"
        items_str = f", {len(self.item_stats_)} items" if self.is_fitted_ else ""
        return f"PopularityRecommender({fitted_str}{items_str})"
