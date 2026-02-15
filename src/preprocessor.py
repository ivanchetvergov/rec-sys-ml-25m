"""Data preprocessing module for MovieLens."""
import logging
from typing import Dict

import pandas as pd
import numpy as np

from src.config import (
    MIN_USER_RATINGS,
    MIN_MOVIE_RATINGS,
    MIN_RATING,
    MAX_RATING,
)

logger = logging.getLogger(__name__)


class Preprocessor:
    """Preprocesses MovieLens user-item interaction data."""

    def __init__(
        self,
        min_user_ratings: int = MIN_USER_RATINGS,
        min_movie_ratings: int = MIN_MOVIE_RATINGS,
        min_rating: float = MIN_RATING,
        max_rating: float = MAX_RATING,
    ):
        """Initialize Preprocessor.

        Args:
            min_user_ratings: Minimum ratings per user (filter inactive users).
            min_movie_ratings: Minimum ratings per movie (filter unpopular movies).
            min_rating: Minimum valid rating value.
            max_rating: Maximum valid rating value.
        """
        self.min_user_ratings = min_user_ratings
        self.min_movie_ratings = min_movie_ratings
        self.min_rating = min_rating
        self.max_rating = max_rating

        logger.info(
            f"Preprocessor initialized: min_user_ratings={min_user_ratings}, "
            f"min_movie_ratings={min_movie_ratings}, rating_range=[{min_rating}, {max_rating}]"
        )

    def clean_ratings(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and filter ratings data.

        Args:
            df: Raw ratings DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning ratings...")
        initial_rows = len(df)
        initial_users = df["userId"].nunique()
        initial_movies = df["movieId"].nunique()

        # Filter valid rating values
        df = df[
            (df["rating"] >= self.min_rating) &
            (df["rating"] <= self.max_rating)
        ].copy()
        logger.info(f"  After rating value filter: {len(df):,} rows")

        # Remove duplicates (same user-movie-timestamp)
        df = df.drop_duplicates(subset=["userId", "movieId", "timestamp"], keep="last")
        logger.info(f"  After removing duplicates: {len(df):,} rows")

        # Iterative filtering: users and movies
        for iteration in range(3):  # Usually converges in 2-3 iterations
            # Count ratings per user and movie
            user_counts = df["userId"].value_counts()
            movie_counts = df["movieId"].value_counts()

            # Filter users with too few ratings
            active_users = user_counts[user_counts >= self.min_user_ratings].index
            df = df[df["userId"].isin(active_users)].copy()

            # Filter movies with too few ratings
            popular_movies = movie_counts[movie_counts >= self.min_movie_ratings].index
            df = df[df["movieId"].isin(popular_movies)].copy()

            logger.info(
                f"  Iteration {iteration + 1}: {len(df):,} rows, "
                f"{df['userId'].nunique():,} users, {df['movieId'].nunique():,} movies"
            )

        final_users = df["userId"].nunique()
        final_movies = df["movieId"].nunique()

        logger.info(
            f"ratings: {initial_rows:,} → {len(df):,} rows "
            f"({100*len(df)/initial_rows:.1f}% kept)"
        )
        logger.info(
            f"  Users: {initial_users:,} → {final_users:,} "
            f"({100*final_users/initial_users:.1f}% kept)"
        )
        logger.info(
            f"  Movies: {initial_movies:,} → {final_movies:,} "
            f"({100*final_movies/initial_movies:.1f}% kept)"
        )

        return df

    def clean_movies(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and parse movies data.

        Args:
            df: Raw movies DataFrame.

        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning movies...")
        initial_rows = len(df)

        # Remove duplicates
        df = df.drop_duplicates(subset=["movieId"], keep="last").copy()

        # Parse title to extract year
        df["year"] = df["title"].str.extract(r"\((\d{4})\)$", expand=False)
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        # Clean title (remove year)
        df["title_clean"] = df["title"].str.replace(r"\s*\(\d{4}\)$", "", regex=True)

        # Handle missing genres
        df["genres"] = df["genres"].fillna("(no genres listed)")

        logger.info(f"movies: {initial_rows:,} → {len(df):,} rows")
        return df

    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge ratings with movies metadata.

        Args:
            datasets: Dictionary with 'ratings' and 'movies' DataFrames.

        Returns:
            Merged DataFrame.
        """
        logger.info("Merging datasets...")

        # Start with ratings (core interaction data)
        df = datasets["ratings"].copy()
        logger.info(f"Starting with ratings: {len(df):,} rows")

        # Merge with movies metadata
        df = df.merge(datasets["movies"], on="movieId", how="left")
        logger.info(f"After merging movies: {len(df):,} rows")

        # Drop rows without movie metadata (shouldn't happen, but just in case)
        missing_movies = df["title"].isna().sum()
        if missing_movies > 0:
            logger.warning(f"Dropping {missing_movies} rows with missing movie metadata")
            df = df.dropna(subset=["title"]).copy()

        logger.info(f"Final merged dataset: {df.shape}")
        return df

    def process(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Run full preprocessing pipeline.

        Args:
            datasets: Dictionary with 'ratings' and 'movies' DataFrames.

        Returns:
            Preprocessed and merged DataFrame.
        """
        # Clean individual datasets
        cleaned = {}
        cleaned["ratings"] = self.clean_ratings(datasets["ratings"])
        cleaned["movies"] = self.clean_movies(datasets["movies"])

        # Merge datasets
        merged_df = self.merge_datasets(cleaned)

        logger.info("Preprocessing completed")
        return merged_df
