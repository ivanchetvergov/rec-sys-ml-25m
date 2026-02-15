"""Feature store module for saving/loading processed features."""
import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime

import pandas as pd

from src.config import FEATURE_STORE_PATH

logger = logging.getLogger(__name__)


class FeatureStore:
    """Manages feature storage and retrieval."""

    def __init__(self, store_path: Path = FEATURE_STORE_PATH):
        """Initialize FeatureStore.

        Args:
            store_path: Path to feature store directory.
        """
        self.store_path = store_path
        self.store_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"FeatureStore initialized at: {store_path}")

    def save_features(
        self,
        df: pd.DataFrame,
        dataset_tag: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save features to feature store.

        Args:
            df: DataFrame with features to save.
            dataset_tag: Tag to identify this dataset version.
            metadata: Optional metadata to store alongside features.
        """
        logger.info(f"Saving features with tag: {dataset_tag}")

        # Create versioned directory
        version_dir = self.store_path / dataset_tag
        version_dir.mkdir(exist_ok=True)

        # Save main features as parquet (efficient format)
        features_path = version_dir / "features.parquet"
        df.to_parquet(features_path, index=False, engine="pyarrow")
        logger.info(f"Saved features to {features_path}")

        # Save metadata
        full_metadata = {
            "dataset_tag": dataset_tag,
            "created_at": datetime.now().isoformat(),
            "n_rows": len(df),
            "n_columns": len(df.columns),
            "columns": df.columns.tolist(),
            "dtypes": df.dtypes.astype(str).to_dict(),
            "memory_usage_mb": round(df.memory_usage(deep=True).sum() / 1024**2, 2),
        }

        if metadata:
            full_metadata.update(metadata)

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        # Save summary statistics
        stats_path = version_dir / "statistics.csv"
        df.describe(include="all").to_csv(stats_path)
        logger.info(f"Saved statistics to {stats_path}")

    def save_splits(
        self,
        splits: Dict[str, pd.DataFrame],
        dataset_tag: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Save train/val/test splits to feature store.

        Args:
            splits: Dictionary with 'train', 'val', 'test' DataFrames.
            dataset_tag: Tag to identify this dataset version.
            metadata: Optional metadata to store alongside features.
        """
        logger.info(f"Saving splits with tag: {dataset_tag}")

        # Create versioned directory
        version_dir = self.store_path / dataset_tag
        version_dir.mkdir(exist_ok=True)

        # Save each split as separate parquet file
        for split_name, split_df in splits.items():
            split_path = version_dir / f"{split_name}.parquet"
            split_df.to_parquet(split_path, index=False, engine="pyarrow")
            logger.info(f"  Saved {split_name}: {len(split_df):,} rows → {split_path}")

        # Save combined metadata
        full_metadata = {
            "dataset_tag": dataset_tag,
            "created_at": datetime.now().isoformat(),
            "has_splits": True,
            "splits": {},
        }

        for split_name, split_df in splits.items():
            full_metadata["splits"][split_name] = {
                "n_rows": len(split_df),
                "n_columns": len(split_df.columns),
                "memory_mb": round(split_df.memory_usage(deep=True).sum() / 1024**2, 2),
            }

        # Get columns from first split
        first_split = list(splits.values())[0]
        full_metadata["columns"] = first_split.columns.tolist()
        full_metadata["dtypes"] = first_split.dtypes.astype(str).to_dict()

        if metadata:
            full_metadata.update(metadata)

        metadata_path = version_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(full_metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")

        # Save statistics for train split only
        stats_path = version_dir / "train_statistics.csv"
        splits["train"].describe(include="all").to_csv(stats_path)
        logger.info(f"Saved train statistics to {stats_path}")

    def load_features(self, dataset_tag: str, split: Optional[str] = None) -> pd.DataFrame:
        """Load features from feature store.

        Args:
            dataset_tag: Tag identifying the dataset version.
            split: Optional split name ('train', 'val', 'test'). If None, loads 'features.parquet'.

        Returns:
            DataFrame with loaded features.
        """
        logger.info(f"Loading features with tag: {dataset_tag}, split: {split}")

        if split:
            features_path = self.store_path / dataset_tag / f"{split}.parquet"
        else:
            features_path = self.store_path / dataset_tag / "features.parquet"

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        df = pd.read_parquet(features_path, engine="pyarrow")
        logger.info(f"Loaded features: {df.shape}")

        return df

    def load_all_splits(self, dataset_tag: str) -> Dict[str, pd.DataFrame]:
        """Load all splits (train/val/test) for a dataset.

        Args:
            dataset_tag: Tag identifying the dataset version.

        Returns:
            Dictionary with 'train', 'val', 'test' DataFrames.
        """
        logger.info(f"Loading all splits for tag: {dataset_tag}")

        splits = {}
        for split_name in ["train", "val", "test"]:
            split_path = self.store_path / dataset_tag / f"{split_name}.parquet"
            if split_path.exists():
                splits[split_name] = pd.read_parquet(split_path, engine="pyarrow")
                logger.info(f"  Loaded {split_name}: {splits[split_name].shape}")

        return splits

    def load_metadata(self, dataset_tag: str) -> Dict[str, Any]:
        """Load metadata for a dataset version.

        Args:
            dataset_tag: Tag identifying the dataset version.

        Returns:
            Metadata dictionary.
        """
        metadata_path = self.store_path / dataset_tag / "metadata.json"

        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            metadata = json.load(f)

        return metadata

    def list_versions(self) -> list:
        """List all available dataset versions.

        Returns:
            List of dataset tags.
        """
        versions = [d.name for d in self.store_path.iterdir() if d.is_dir()]
        logger.info(f"Found {len(versions)} versions: {versions}")
        return sorted(versions)

    def get_latest_version(self) -> Optional[str]:
        """Get the most recent dataset version.

        Returns:
            Latest dataset tag or None if no versions exist.
        """
        versions = self.list_versions()
        return versions[-1] if versions else None
