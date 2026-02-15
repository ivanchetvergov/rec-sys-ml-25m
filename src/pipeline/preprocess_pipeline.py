"""Main preprocessing pipeline script for MovieLens."""
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.data_loader import DataLoader
from src.preprocessor import Preprocessor
from src.feature_engineer import FeatureEngineer
from src.data_splitter import DataSplitter
from src.feature_store import FeatureStore

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("preprocessing.log"),
    ],
)
logger = logging.getLogger(__name__)


def run_preprocessing_pipeline(dataset_tag: str = None) -> None:
    """Execute the full preprocessing pipeline for MovieLens.

    Args:
        dataset_tag: Tag for this dataset version. Defaults to timestamp.
    """
    if dataset_tag is None:
        dataset_tag = datetime.now().strftime("ml_v_%Y%m%d_%H%M%S")

    logger.info("=" * 80)
    logger.info(f"Starting MovieLens preprocessing pipeline: {dataset_tag}")
    logger.info("=" * 80)

    try:
        # Step 1: Load data
        logger.info("\n[STEP 1/5] Loading MovieLens data...")
        loader = DataLoader()
        datasets = loader.load_all(load_tags=False, load_links=False)

        # Step 2: Preprocess and merge
        logger.info("\n[STEP 2/5] Preprocessing and merging...")
        preprocessor = Preprocessor()
        merged_df = preprocessor.process(datasets)

        # Step 3: Feature engineering
        logger.info("\n[STEP 3/5] Engineering features...")
        feature_engineer = FeatureEngineer()
        features_df = feature_engineer.engineer_features(merged_df)

        # Step 4: Split into train/val/test
        logger.info("\n[STEP 4/5] Splitting into train/val/test...")
        splitter = DataSplitter()
        splits = splitter.temporal_split(features_df, time_column="timestamp")

        # Step 5: Save to feature store
        logger.info("\n[STEP 5/5] Saving to feature store...")
        feature_store = FeatureStore()

        metadata = {
            "dataset": "MovieLens-25M",
            "min_user_ratings": preprocessor.min_user_ratings,
            "min_movie_ratings": preprocessor.min_movie_ratings,
            "rating_range": [preprocessor.min_rating, preprocessor.max_rating],
            "split_config": splitter.get_split_metadata(splits),
        }

        feature_store.save_splits(splits, dataset_tag, metadata)

        # Summary
        logger.info("\n" + "=" * 80)
        logger.info("Pipeline completed successfully!")
        logger.info(f"Dataset tag: {dataset_tag}")
        logger.info(f"Total records: {len(features_df):,}")
        logger.info(f"  Train: {len(splits['train']):,}")
        logger.info(f"  Val:   {len(splits['val']):,}")
        logger.info(f"  Test:  {len(splits['test']):,}")
        logger.info(f"Total features: {len(features_df.columns)}")
        logger.info(f"Users: {features_df['userId'].nunique():,}")
        logger.info(f"Movies: {features_df['movieId'].nunique():,}")
        logger.info(f"Memory usage: {features_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
        logger.info("=" * 80)

    except Exception as e:
        logger.error(f"Pipeline failed: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    # Accept optional dataset tag from command line
    dataset_tag = sys.argv[1] if len(sys.argv) > 1 else None
    run_preprocessing_pipeline(dataset_tag)
