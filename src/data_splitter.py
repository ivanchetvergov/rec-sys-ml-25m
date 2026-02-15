"""Data splitting module for MovieLens with temporal splits."""
import logging
from typing import Tuple, Dict

import pandas as pd
import numpy as np

from src.config import TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT

logger = logging.getLogger(__name__)


class DataSplitter:
    """Splits data into train/validation/test sets for recommender systems."""

    def __init__(
        self,
        train_split: float = TRAIN_SPLIT,
        val_split: float = VAL_SPLIT,
        test_split: float = TEST_SPLIT,
        seed: int = 42,
    ):
        """Initialize DataSplitter.

        Args:
            train_split: Fraction of data for training (oldest interactions).
            val_split: Fraction of data for validation.
            test_split: Fraction of data for testing (newest interactions).
            seed: Random seed for reproducibility.
        """
        assert abs(train_split + val_split + test_split - 1.0) < 1e-6, "Splits must sum to 1.0"

        self.train_split = train_split
        self.val_split = val_split
        self.test_split = test_split
        self.seed = seed

        logger.info(
            f"DataSplitter initialized: train={train_split}, val={val_split}, "
            f"test={test_split}, seed={seed}"
        )

    def temporal_split(
        self, df: pd.DataFrame, time_column: str = "timestamp"
    ) -> Dict[str, pd.DataFrame]:
        """Split data by timestamp (critical for recommender systems!).

        Args:
            df: Input DataFrame with temporal column.
            time_column: Column containing timestamps.

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames.
        """
        logger.info(f"Performing temporal split on column: {time_column}")

        # Sort by timestamp
        df = df.sort_values(time_column).reset_index(drop=True)

        # Calculate split indices
        n_total = len(df)
        n_train = int(n_total * self.train_split)
        n_val = int(n_total * self.val_split)

        # Split
        train_df = df.iloc[:n_train].copy()
        val_df = df.iloc[n_train:n_train + n_val].copy()
        test_df = df.iloc[n_train + n_val:].copy()

        # Log split stats
        logger.info(f"Split statistics:")
        logger.info(f"  Total: {n_total:,} rows")
        logger.info(f"  Train: {len(train_df):,} rows ({100*len(train_df)/n_total:.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} rows ({100*len(val_df)/n_total:.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} rows ({100*len(test_df)/n_total:.1f}%)")

        # Timestamp ranges
        logger.info(f"Timestamp ranges:")
        logger.info(f"  Train: {train_df[time_column].min()} to {train_df[time_column].max()}")
        logger.info(f"  Val:   {val_df[time_column].min()} to {val_df[time_column].max()}")
        logger.info(f"  Test:  {test_df[time_column].min()} to {test_df[time_column].max()}")

        # User/movie coverage
        logger.info(f"User coverage:")
        logger.info(f"  Train: {train_df['userId'].nunique():,} users")
        logger.info(f"  Val:   {val_df['userId'].nunique():,} users")
        logger.info(f"  Test:  {test_df['userId'].nunique():,} users")

        logger.info(f"Movie coverage:")
        logger.info(f"  Train: {train_df['movieId'].nunique():,} movies")
        logger.info(f"  Val:   {val_df['movieId'].nunique():,} movies")
        logger.info(f"  Test:  {test_df['movieId'].nunique():,} movies")

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

    def user_based_split(
        self, df: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """Split by users (alternative approach to avoid cold-start in val/test).

        Each user's interactions are split temporally:
        - User's oldest 70% → train
        - User's middle 15% → val
        - User's newest 15% → test

        Args:
            df: Input DataFrame with userId and timestamp.

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames.
        """
        logger.info("Performing user-based temporal split...")

        np.random.seed(self.seed)

        train_list = []
        val_list = []
        test_list = []

        # Split each user's history
        for user_id, user_df in df.groupby("userId"):
            user_df = user_df.sort_values("timestamp").reset_index(drop=True)
            n = len(user_df)

            n_train = int(n * self.train_split)
            n_val = int(n * self.val_split)

            train_list.append(user_df.iloc[:n_train])
            val_list.append(user_df.iloc[n_train:n_train + n_val])
            test_list.append(user_df.iloc[n_train + n_val:])

        train_df = pd.concat(train_list, ignore_index=True)
        val_df = pd.concat(val_list, ignore_index=True)
        test_df = pd.concat(test_list, ignore_index=True)

        n_total = len(df)
        logger.info(f"User-based split statistics:")
        logger.info(f"  Total: {n_total:,} rows")
        logger.info(f"  Train: {len(train_df):,} rows ({100*len(train_df)/n_total:.1f}%)")
        logger.info(f"  Val:   {len(val_df):,} rows ({100*len(val_df)/n_total:.1f}%)")
        logger.info(f"  Test:  {len(test_df):,} rows ({100*len(test_df)/n_total:.1f}%)")

        return {
            "train": train_df,
            "val": val_df,
            "test": test_df,
        }

    def get_split_metadata(self, splits: Dict[str, pd.DataFrame]) -> Dict:
        """Generate metadata about the splits.

        Args:
            splits: Dictionary of split DataFrames.

        Returns:
            Metadata dictionary.
        """
        metadata = {
            "train_split": self.train_split,
            "val_split": self.val_split,
            "test_split": self.test_split,
            "seed": self.seed,
            "splits": {},
        }

        for split_name, split_df in splits.items():
            metadata["splits"][split_name] = {
                "n_rows": len(split_df),
                "n_users": split_df["userId"].nunique() if "userId" in split_df.columns else 0,
                "n_movies": split_df["movieId"].nunique() if "movieId" in split_df.columns else 0,
                "n_columns": len(split_df.columns),
                "memory_mb": round(split_df.memory_usage(deep=True).sum() / 1024**2, 2),
            }

        return metadata
