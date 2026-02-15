"""Data loading module for MovieLens datasets."""
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from src.config import DATA_RAW, DATASETS

logger = logging.getLogger(__name__)


class DataLoader:
    """Loads MovieLens datasets from CSV files."""

    def __init__(self, data_path: Path = DATA_RAW):
        """Initialize DataLoader.

        Args:
            data_path: Path to directory containing MovieLens CSV files.
        """
        self.data_path = data_path
        logger.info(f"DataLoader initialized with path: {data_path}")

    def load_ratings(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load ratings dataset (main interaction data).

        Args:
            nrows: Number of rows to load (None = all rows).

        Returns:
            DataFrame with userId, movieId, rating, timestamp.
        """
        file_path = self.data_path / DATASETS["ratings"]
        if not file_path.exists():
            raise FileNotFoundError(f"Ratings file not found: {file_path}")

        logger.info(f"Loading ratings from {file_path}")
        df = pd.read_csv(file_path, nrows=nrows)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        logger.info(f"Loaded ratings: {df.shape[0]:,} rows, {df.shape[1]} columns")
        logger.info(f"  Users: {df['userId'].nunique():,}")
        logger.info(f"  Movies: {df['movieId'].nunique():,}")
        logger.info(f"  Timestamp range: {df['timestamp'].min()} to {df['timestamp'].max()}")

        return df

    def load_movies(self) -> pd.DataFrame:
        """Load movies dataset with titles and genres.

        Returns:
            DataFrame with movieId, title, genres.
        """
        file_path = self.data_path / DATASETS["movies"]
        if not file_path.exists():
            raise FileNotFoundError(f"Movies file not found: {file_path}")

        logger.info(f"Loading movies from {file_path}")
        df = pd.read_csv(file_path)

        logger.info(f"Loaded movies: {df.shape[0]:,} rows, {df.shape[1]} columns")
        return df

    def load_tags(self, nrows: Optional[int] = None) -> pd.DataFrame:
        """Load user-generated tags (optional).

        Args:
            nrows: Number of rows to load (None = all rows).

        Returns:
            DataFrame with userId, movieId, tag, timestamp.
        """
        file_path = self.data_path / DATASETS["tags"]
        if not file_path.exists():
            logger.warning(f"Tags file not found: {file_path}, skipping")
            return pd.DataFrame()

        logger.info(f"Loading tags from {file_path}")
        df = pd.read_csv(file_path, nrows=nrows)

        # Convert timestamp to datetime
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")

        logger.info(f"Loaded tags: {df.shape[0]:,} rows")
        return df

    def load_links(self) -> pd.DataFrame:
        """Load links to IMDb and TMDB (optional).

        Returns:
            DataFrame with movieId, imdbId, tmdbId.
        """
        file_path = self.data_path / DATASETS["links"]
        if not file_path.exists():
            logger.warning(f"Links file not found: {file_path}, skipping")
            return pd.DataFrame()

        logger.info(f"Loading links from {file_path}")
        df = pd.read_csv(file_path)

        logger.info(f"Loaded links: {df.shape[0]:,} rows")
        return df

    def load_all(self, load_tags: bool = False, load_links: bool = False) -> Dict[str, pd.DataFrame]:
        """Load core MovieLens datasets.

        Args:
            load_tags: Whether to load tags dataset.
            load_links: Whether to load links dataset.

        Returns:
            Dictionary of DataFrames.
        """
        datasets = {}

        # Core datasets (required)
        datasets["ratings"] = self.load_ratings()
        datasets["movies"] = self.load_movies()

        # Optional datasets
        if load_tags:
            datasets["tags"] = self.load_tags()

        if load_links:
            datasets["links"] = self.load_links()

        logger.info(f"Loaded {len(datasets)} datasets")
        return datasets
