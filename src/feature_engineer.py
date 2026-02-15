"""Feature engineering module for MovieLens."""
import logging
from typing import List

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features from preprocessed MovieLens data."""

    def __init__(self, top_n_genres: int = 20):
        """Initialize FeatureEngineer.

        Args:
            top_n_genres: Number of top genres to create binary features for.
        """
        self.top_n_genres = top_n_genres
        logger.info(f"FeatureEngineer initialized: top_n_genres={top_n_genres}")

    def create_movie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create movie-level features.

        Args:
            df: Input DataFrame with movie metadata.

        Returns:
            DataFrame with movie features added.
        """
        logger.info("Creating movie features...")

        # Parse genres into list
        df["genre_list"] = df["genres"].str.split("|")
        df["num_genres"] = df["genre_list"].apply(len)

        # Create binary genre features for top genres
        all_genres = df["genre_list"].explode().value_counts()
        top_genres = all_genres.head(self.top_n_genres).index.tolist()

        for genre in top_genres:
            df[f"genre_{genre.lower().replace('-', '_')}"] = df["genre_list"].apply(
                lambda x: 1 if genre in x else 0
            )

        logger.info(f"Created {len(top_genres)} genre features")

        # Year-based features
        current_year = 2026
        df["movie_age"] = current_year - df["year"]
        df["decade"] = (df["year"] // 10) * 10

        # Fill missing years with median
        median_year = df["year"].median()
        df["year"] = df["year"].fillna(median_year)
        df["movie_age"] = df["movie_age"].fillna(current_year - median_year)

        # Title length
        df["title_length"] = df["title_clean"].str.len()

        logger.info("Created movie features")
        return df

    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-level aggregate features.

        Args:
            df: Input DataFrame with user-item interactions.

        Returns:
            DataFrame with user features added.
        """
        logger.info("Creating user features...")

        # User rating statistics
        user_stats = df.groupby("userId").agg({
            "rating": ["mean", "std", "count", "min", "max"],
            "timestamp": ["min", "max"]
        }).reset_index()

        user_stats.columns = [
            "userId",
            "user_avg_rating",
            "user_rating_std",
            "user_num_ratings",
            "user_min_rating",
            "user_max_rating",
            "user_first_interaction",
            "user_last_interaction",
        ]

        # User activity span (days)
        user_stats["user_activity_days"] = (
            (user_stats["user_last_interaction"] - user_stats["user_first_interaction"])
            .dt.total_seconds() / (24 * 3600)
        )

        # User rating velocity (ratings per day)
        user_stats["user_rating_velocity"] = (
            user_stats["user_num_ratings"] / user_stats["user_activity_days"].replace(0, 1)
        )

        # Fill missing std with 0 (single-rating users)
        user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0)

        # Merge back to main dataframe
        df = df.merge(user_stats, on="userId", how="left")

        logger.info("Created user features")
        return df

    def create_movie_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create movie popularity features.

        Args:
            df: Input DataFrame with user-item interactions.

        Returns:
            DataFrame with popularity features added.
        """
        logger.info("Creating movie popularity features...")

        # Movie rating statistics
        movie_stats = df.groupby("movieId").agg({
            "rating": ["mean", "std", "count"],
            "userId": "nunique"
        }).reset_index()

        movie_stats.columns = [
            "movieId",
            "movie_avg_rating",
            "movie_rating_std",
            "movie_num_ratings",
            "movie_num_users",
        ]

        # Popularity score
        movie_stats["movie_popularity"] = (
            movie_stats["movie_avg_rating"] * np.log1p(movie_stats["movie_num_ratings"])
        )

        # Fill missing std
        movie_stats["movie_rating_std"] = movie_stats["movie_rating_std"].fillna(0)

        # Merge back
        df = df.merge(movie_stats, on="movieId", how="left")

        logger.info("Created movie popularity features")
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction features.

        Args:
            df: Input DataFrame with user-item interactions.

        Returns:
            DataFrame with interaction features added.
        """
        logger.info("Creating interaction features...")

        # Rating deviation from user's mean
        df["rating_deviation_user"] = df["rating"] - df["user_avg_rating"]

        # Rating deviation from movie's mean
        df["rating_deviation_movie"] = df["rating"] - df["movie_avg_rating"]

        # Is this above/below average for user?
        df["above_user_avg"] = (df["rating"] > df["user_avg_rating"]).astype(int)

        # Is this above/below average for movie?
        df["above_movie_avg"] = (df["rating"] > df["movie_avg_rating"]).astype(int)

        # Timestamp features
        df["interaction_year"] = df["timestamp"].dt.year
        df["interaction_month"] = df["timestamp"].dt.month
        df["interaction_day_of_week"] = df["timestamp"].dt.dayofweek
        df["interaction_hour"] = df["timestamp"].dt.hour

        # Time since user's first interaction
        df["days_since_first_interaction"] = (
            (df["timestamp"] - df["user_first_interaction"]).dt.total_seconds() / (24 * 3600)
        )

        logger.info("Created interaction features")
        return df

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature engineering pipeline.

        Args:
            df: Preprocessed DataFrame.

        Returns:
            DataFrame with all engineered features.
        """
        logger.info("Starting feature engineering...")

        df = self.create_movie_features(df)
        df = self.create_user_features(df)
        df = self.create_movie_popularity_features(df)
        df = self.create_interaction_features(df)

        logger.info(f"Feature engineering completed: {df.shape[1]} total columns")
        return df
