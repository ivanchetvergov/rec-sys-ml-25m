"""Training script for Popularity-based recommender with MLflow tracking."""
import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlflow
import mlflow.pyfunc
import numpy as np
import pandas as pd

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.config import FEATURE_STORE_PATH
from src.evaluation.metrics import create_ground_truth, evaluate_recommendations
from src.models.popularity_based import PopularityRecommender

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)


def load_data(
    dataset_tag: str,
    sample_frac: Optional[float] = None,
    sample_seed: int = 42
) -> Dict[str, pd.DataFrame]:
    """
    Load train/val/test data from feature store.

    Args:
        dataset_tag: Dataset version tag (e.g., "ml_v_20260215_184134")
        sample_frac: Fraction of data to sample (0.0 to 1.0). None = full data
        sample_seed: Random seed for sampling

    Returns:
        Dict with "train", "val", "test" dataframes
    """
    dataset_path = FEATURE_STORE_PATH / dataset_tag

    if not dataset_path.exists():
        raise ValueError(f"Dataset not found: {dataset_path}")

    logger.info("=" * 80)
    logger.info(f"Loading dataset: {dataset_tag}")
    logger.info(f"Path: {dataset_path}")

    # Load metadata
    with open(dataset_path / "metadata.json", "r") as f:
        metadata = json.load(f)

    logger.info(f"Dataset created: {metadata['created_at']}")
    logger.info(f"Total splits: {metadata['splits']}")

    # Load splits
    data = {}
    for split in ["train", "val", "test"]:
        split_path = dataset_path / f"{split}.parquet"
        logger.info(f"Loading {split} from {split_path}...")

        df = pd.read_parquet(split_path)

        # Sample if requested
        if sample_frac is not None and 0 < sample_frac < 1.0:
            n_original = len(df)
            df = df.sample(frac=sample_frac, random_state=sample_seed)
            logger.info(f"  Sampled {len(df):,} / {n_original:,} rows ({sample_frac*100:.1f}%)")
        else:
            logger.info(f"  Loaded {len(df):,} rows")

        data[split] = df

    logger.info("=" * 80)

    return data


def train_and_evaluate(
    dataset_tag: str,
    min_ratings: int = 10,
    rating_weight: float = 1.0,
    count_weight: float = 1.0,
    sample_frac: Optional[float] = None,
    relevance_threshold: float = 4.0,
    k_values: list = [5, 10, 20],
    mlflow_experiment: str = "popularity_baseline",
    run_name: Optional[str] = None,
    seed: int = 42
) -> Dict:
    """
    Train and evaluate Popularity-based recommender with MLflow tracking.

    Args:
        dataset_tag: Dataset version tag
        min_ratings: Minimum ratings per item
        rating_weight: Weight for avg rating in popularity score
        count_weight: Weight for log(count) in popularity score
        sample_frac: Fraction of data to sample (None = full data)
        relevance_threshold: Minimum rating to consider item relevant
        k_values: List of K values to evaluate at
        mlflow_experiment: MLflow experiment name
        run_name: MLflow run name (auto-generated if None)
        seed: Random seed

    Returns:
        Dict with evaluation metrics
    """
    # Set random seed
    np.random.seed(seed)

    # Generate run name
    if run_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_str = f"_sample{int(sample_frac*100)}" if sample_frac else "_full"
        run_name = f"popularity{sample_str}_{timestamp}"

    # Set MLflow experiment
    mlflow.set_experiment(mlflow_experiment)

    # Start MLflow run
    with mlflow.start_run(run_name=run_name):
        logger.info("=" * 80)
        logger.info(f"MLflow Run: {run_name}")
        logger.info(f"Experiment: {mlflow_experiment}")
        logger.info("=" * 80)

        # Log parameters
        mlflow.log_param("dataset_tag", dataset_tag)
        mlflow.log_param("model_type", "PopularityRecommender")
        mlflow.log_param("min_ratings", min_ratings)
        mlflow.log_param("rating_weight", rating_weight)
        mlflow.log_param("count_weight", count_weight)
        mlflow.log_param("sample_frac", sample_frac or 1.0)
        mlflow.log_param("relevance_threshold", relevance_threshold)
        mlflow.log_param("seed", seed)

        # Load data
        start_time = time.time()
        data = load_data(dataset_tag, sample_frac=sample_frac, sample_seed=seed)
        load_time = time.time() - start_time

        train_df = data["train"]
        val_df = data["val"]
        test_df = data["test"]

        # Log data statistics
        mlflow.log_metric("n_train", len(train_df))
        mlflow.log_metric("n_val", len(val_df))
        mlflow.log_metric("n_test", len(test_df))
        mlflow.log_metric("n_users_train", train_df["userId"].nunique())
        mlflow.log_metric("n_movies_train", train_df["movieId"].nunique())
        mlflow.log_metric("data_load_time_sec", load_time)

        # Train model
        logger.info("\n" + "=" * 80)
        logger.info("Training model...")
        logger.info("=" * 80)

        model = PopularityRecommender(
            min_ratings=min_ratings,
            rating_weight=rating_weight,
            count_weight=count_weight
        )

        train_start = time.time()
        model.fit(train_df)
        train_time = time.time() - train_start

        mlflow.log_metric("train_time_sec", train_time)
        mlflow.log_metric("n_items_catalog", len(model.item_stats_))

        # Log top 10 popular items
        top_items = model.get_top_items(10)
        top_items_str = "\n".join([
            f"  {i+1}. Movie {row['movieId']}: "
            f"score={row['popularity_score']:.2f}, "
            f"avg_rating={row['avg_rating']:.2f}, "
            f"n_ratings={row['num_ratings']}"
            for i, (_, row) in enumerate(top_items.iterrows())
        ])
        logger.info(f"\nTop 10 popular items:\n{top_items_str}")

        # Save top items as artifact
        top_items_path = Path("top_items.csv")
        top_items.to_csv(top_items_path, index=False)
        mlflow.log_artifact(top_items_path)
        top_items_path.unlink()  # Clean up

        # Evaluate on validation set
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating on VALIDATION set...")
        logger.info("=" * 80)

        val_users = val_df["userId"].unique()[:1000]  # Sample 1000 users for speed
        logger.info(f"Evaluating on {len(val_users)} users...")

        # Get items each user has already interacted with (to exclude)
        val_user_items = train_df.groupby("userId")["movieId"].apply(set).to_dict()

        # Generate recommendations
        eval_start = time.time()
        recommendations = model.recommend_batch(
            user_ids=val_users.tolist(),
            n=max(k_values),
            exclude_items_per_user=val_user_items
        )
        eval_time = time.time() - eval_start

        # Create ground truth
        ground_truth = create_ground_truth(
            val_df[val_df["userId"].isin(val_users)],
            relevance_threshold=relevance_threshold
        )

        # Evaluate
        catalog = set(train_df["movieId"].unique())
        metrics = evaluate_recommendations(
            recommendations=recommendations,
            ground_truth=ground_truth,
            catalog=catalog,
            k_values=k_values
        )

        # Log metrics
        for metric_name, value in metrics.items():
            mlflow.log_metric(f"val_{metric_name}", value)

        mlflow.log_metric("val_eval_time_sec", eval_time)
        mlflow.log_metric("val_eval_users", len(val_users))

        # Evaluate on test set
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating on TEST set...")
        logger.info("=" * 80)

        test_users = test_df["userId"].unique()[:1000]  # Sample 1000 users
        logger.info(f"Evaluating on {len(test_users)} users...")

        # Get items each user has already interacted with
        train_val_df = pd.concat([train_df, val_df])
        test_user_items = train_val_df.groupby("userId")["movieId"].apply(set).to_dict()

        # Generate recommendations
        test_eval_start = time.time()
        test_recommendations = model.recommend_batch(
            user_ids=test_users.tolist(),
            n=max(k_values),
            exclude_items_per_user=test_user_items
        )
        test_eval_time = time.time() - test_eval_start

        # Create ground truth
        test_ground_truth = create_ground_truth(
            test_df[test_df["userId"].isin(test_users)],
            relevance_threshold=relevance_threshold
        )

        # Evaluate
        test_metrics = evaluate_recommendations(
            recommendations=test_recommendations,
            ground_truth=test_ground_truth,
            catalog=catalog,
            k_values=k_values
        )

        # Log test metrics
        for metric_name, value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", value)

        mlflow.log_metric("test_eval_time_sec", test_eval_time)
        mlflow.log_metric("test_eval_users", len(test_users))

        # Save model
        logger.info("\n" + "=" * 80)
        logger.info("Saving model...")
        logger.info("=" * 80)

        model_dir = Path("model_artifacts")
        model_dir.mkdir(exist_ok=True)
        model.save(model_dir)

        # Log model artifacts
        mlflow.log_artifacts(model_dir)

        # Clean up
        import shutil
        shutil.rmtree(model_dir)

        # Log summary
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model: PopularityRecommender")
        logger.info(f"Dataset: {dataset_tag}")
        logger.info(f"Sample fraction: {sample_frac or 1.0}")
        logger.info(f"Train time: {train_time:.2f}s")
        logger.info(f"Catalog size: {len(model.item_stats_)} items")
        logger.info("")
        logger.info("Validation metrics:")
        for k in k_values:
            logger.info(
                f"  K={k}: "
                f"P={metrics[f'precision_at_{k}']:.4f}, "
                f"R={metrics[f'recall_at_{k}']:.4f}, "
                f"NDCG={metrics[f'ndcg_at_{k}']:.4f}, "
                f"Coverage={metrics[f'coverage_at_{k}']:.4f}"
            )
        logger.info("")
        logger.info("Test metrics:")
        for k in k_values:
            logger.info(
                f"  K={k}: "
                f"P={test_metrics[f'precision_at_{k}']:.4f}, "
                f"R={test_metrics[f'recall_at_{k}']:.4f}, "
                f"NDCG={test_metrics[f'ndcg_at_{k}']:.4f}, "
                f"Coverage={test_metrics[f'coverage_at_{k}']:.4f}"
            )
        logger.info("=" * 80)

        return {**metrics, **{f"test_{k}": v for k, v in test_metrics.items()}}


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Train Popularity-based recommender with MLflow tracking"
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        required=True,
        help="Dataset version tag (e.g., ml_v_20260215_184134)"
    )
    parser.add_argument(
        "--min-ratings",
        type=int,
        default=10,
        help="Minimum ratings per item (default: 10)"
    )
    parser.add_argument(
        "--rating-weight",
        type=float,
        default=1.0,
        help="Weight for avg rating in popularity score (default: 1.0)"
    )
    parser.add_argument(
        "--count-weight",
        type=float,
        default=1.0,
        help="Weight for log(count) in popularity score (default: 1.0)"
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fraction of data to sample for fast testing (e.g., 0.1 = 10%%). None = full data"
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to consider item relevant (default: 4.0)"
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="K values for evaluation metrics (default: 5 10 20)"
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="popularity_baseline",
        help="MLflow experiment name (default: popularity_baseline)"
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not provided)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )

    args = parser.parse_args()

    # Train and evaluate
    train_and_evaluate(
        dataset_tag=args.dataset_tag,
        min_ratings=args.min_ratings,
        rating_weight=args.rating_weight,
        count_weight=args.count_weight,
        sample_frac=args.sample_frac,
        relevance_threshold=args.relevance_threshold,
        k_values=args.k_values,
        mlflow_experiment=args.experiment,
        run_name=args.run_name,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
