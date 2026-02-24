"""Train Collaborative Filtering (SVD) recommender with MLflow tracking.

Loads pre-processed splits from the feature store, fits an SVDRecommender,
evaluates on validation and test sets, and logs everything to MLflow.

Example usage:
    # Full dataset — slow, use on a server
    python -m src.training.train_collaborative --dataset-tag ml_v_20260215_184134

    # Quick dev run on 10% of data
    python -m src.training.train_collaborative \\
        --dataset-tag ml_v_20260215_184134 \\
        --sample-frac 0.1 \\
        --factors 50 \\
        --run-name svd_dev_run
"""
import argparse
import json
import logging
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import mlflow
import numpy as np
import pandas as pd

from src.config import FEATURE_STORE_PATH
from src.evaluation.metrics import create_ground_truth, evaluate_recommendations
from src.models.collaborative_filtering import SVDRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Data loading (same contract as train_popularity.py)
# ---------------------------------------------------------------------------


def load_data(
    dataset_tag: str,
    sample_frac: Optional[float] = None,
    sample_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    """Load train/val/test splits from the feature store.

    Args:
        dataset_tag: Version tag written by the preprocessing pipeline
                     (e.g. ``ml_v_20260215_184134``).
        sample_frac: If provided, randomly sample this fraction of the
                     *training set* rows for fast experimentation.
        sample_seed: Random seed used for sampling.

    Returns:
        Dict with keys ``"train"``, ``"val"``, ``"test"``, ``"metadata"``.
    """
    version_dir = FEATURE_STORE_PATH / dataset_tag
    if not version_dir.exists():
        raise FileNotFoundError(
            f"Feature store version not found: {version_dir}\n"
            f"Run the preprocessing pipeline first to generate '{dataset_tag}'."
        )

    logger.info("=" * 80)
    logger.info(f"Loading feature store: {dataset_tag}")
    logger.info("=" * 80)

    train_df = pd.read_parquet(version_dir / "train.parquet")
    val_df = pd.read_parquet(version_dir / "val.parquet")
    test_df = pd.read_parquet(version_dir / "test.parquet")

    metadata_path = version_dir / "metadata.json"
    metadata: Dict = {}
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)

    if sample_frac is not None and 0.0 < sample_frac < 1.0:
        original = len(train_df)
        train_df = train_df.sample(frac=sample_frac, random_state=sample_seed)
        logger.info(f"Sampled train: {original:,} → {len(train_df):,} rows ({sample_frac:.0%})")

    logger.info(f"  train : {len(train_df):,} rows")
    logger.info(f"  val   : {len(val_df):,} rows")
    logger.info(f"  test  : {len(test_df):,} rows")
    logger.info(
        f"  train users/items: "
        f"{train_df['userId'].nunique():,} / {train_df['movieId'].nunique():,}"
    )
    logger.info("=" * 80)

    return {"train": train_df, "val": val_df, "test": test_df, "metadata": metadata}


# ---------------------------------------------------------------------------
# Evaluation helper
# ---------------------------------------------------------------------------


def _build_recommendations(
    model: SVDRecommender,
    eval_df: pd.DataFrame,
    seen_df: pd.DataFrame,
    max_users: int,
    k: int,
) -> Dict[int, List[int]]:
    """Generate recommendation dicts for a sample of users.

    Args:
        model: Fitted SVDRecommender.
        eval_df: DataFrame containing the users to evaluate (val or test).
        seen_df: DataFrame of interactions to exclude (train [+ val]).
        max_users: Cap on number of users to evaluate (for speed).
        k: Maximum list length per user.

    Returns:
        Dict[user_id -> list of recommended item IDs].
    """
    # Only evaluate users known to the model (in-matrix users)
    known_users = set(model.user_id_map_.keys())
    eval_users = [u for u in eval_df["userId"].unique() if u in known_users]
    eval_users = eval_users[:max_users]

    logger.info(f"Generating recs for {len(eval_users):,} users (k={k})...")

    exclude_per_user = (
        seen_df.groupby("userId")["movieId"].apply(set).to_dict()
    )

    start = time.time()
    recs = model.recommend_batch(
        user_ids=eval_users,
        n=k,
        exclude_items_per_user=exclude_per_user,
    )
    elapsed = time.time() - start
    logger.info(f"Recs generated in {elapsed:.2f}s ({1000*elapsed/max(len(eval_users),1):.1f} ms/user)")
    return recs


# ---------------------------------------------------------------------------
# Main training & evaluation function
# ---------------------------------------------------------------------------


def train_and_evaluate(
    dataset_tag: str,
    factors: int = 100,
    sample_frac: Optional[float] = None,
    relevance_threshold: float = 4.0,
    k_values: List[int] = [5, 10, 20],
    max_eval_users: int = 1000,
    mlflow_experiment: str = "collaborative_filtering",
    run_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, float]:
    """Train SVDRecommender and log everything to MLflow.

    Args:
        dataset_tag: Feature store version tag.
        factors: Number of SVD latent factors.
        sample_frac: Fraction of training rows to use (None = full).
        relevance_threshold: Minimum rating to count as relevant.
        k_values: List of cut-off values for ranking metrics.
        max_eval_users: Max users to evaluate on val/test (for speed).
        mlflow_experiment: MLflow experiment name.
        run_name: MLflow run name (auto-generated if None).
        seed: Random seed for reproducibility.

    Returns:
        Dict of evaluation metrics from the test set.
    """
    np.random.seed(seed)

    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_str = f"_s{int(sample_frac * 100)}" if sample_frac else "_full"
        run_name = f"svd_k{factors}{sample_str}_{ts}"

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        logger.info("=" * 80)
        logger.info(f"MLflow run : {run_name}")
        logger.info(f"Experiment : {mlflow_experiment}")
        logger.info("=" * 80)

        # ---- log hyper-parameters ----------------------------------------
        mlflow.log_param("model_type", "SVDRecommender")
        mlflow.log_param("dataset_tag", dataset_tag)
        mlflow.log_param("factors", factors)
        mlflow.log_param("sample_frac", sample_frac or 1.0)
        mlflow.log_param("relevance_threshold", relevance_threshold)
        mlflow.log_param("max_eval_users", max_eval_users)
        mlflow.log_param("seed", seed)

        # ---- load data -------------------------------------------------------
        t0 = time.time()
        data = load_data(dataset_tag, sample_frac=sample_frac, sample_seed=seed)
        mlflow.log_metric("data_load_time_sec", round(time.time() - t0, 2))

        train_df = data["train"]
        val_df = data["val"]
        test_df = data["test"]

        mlflow.log_metric("n_train", len(train_df))
        mlflow.log_metric("n_val", len(val_df))
        mlflow.log_metric("n_test", len(test_df))
        mlflow.log_metric("n_users_train", int(train_df["userId"].nunique()))
        mlflow.log_metric("n_items_train", int(train_df["movieId"].nunique()))

        # ---- train model -----------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("Training SVDRecommender...")
        logger.info("=" * 80)

        model = SVDRecommender(factors=factors, random_state=seed)

        t_train = time.time()
        model.fit(train_df)
        train_time = time.time() - t_train

        mlflow.log_metric("train_time_sec", round(train_time, 2))
        mlflow.log_metric("actual_factors", int(model.user_factors_.shape[1]))
        logger.info(f"Training complete in {train_time:.2f}s | {model}")

        # ---- evaluate on validation set -------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating on VALIDATION set...")
        logger.info("=" * 80)

        val_recs = _build_recommendations(
            model, val_df, seen_df=train_df,
            max_users=max_eval_users, k=max(k_values),
        )
        val_ground_truth = create_ground_truth(
            val_df[val_df["userId"].isin(val_recs.keys())],
            relevance_threshold=relevance_threshold,
        )
        catalog = set(int(x) for x in train_df["movieId"].unique())
        val_metrics = evaluate_recommendations(
            recommendations=val_recs,
            ground_truth=val_ground_truth,
            catalog=catalog,
            k_values=k_values,
        )

        for name, value in val_metrics.items():
            mlflow.log_metric(f"val_{name}", round(value, 6))

        _log_metrics_table("Validation", val_metrics, k_values)

        # ---- evaluate on test set -------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("Evaluating on TEST set...")
        logger.info("=" * 80)

        train_val_df = pd.concat([train_df, val_df], ignore_index=True)
        test_recs = _build_recommendations(
            model, test_df, seen_df=train_val_df,
            max_users=max_eval_users, k=max(k_values),
        )
        test_ground_truth = create_ground_truth(
            test_df[test_df["userId"].isin(test_recs.keys())],
            relevance_threshold=relevance_threshold,
        )
        test_metrics = evaluate_recommendations(
            recommendations=test_recs,
            ground_truth=test_ground_truth,
            catalog=catalog,
            k_values=k_values,
        )

        for name, value in test_metrics.items():
            mlflow.log_metric(f"test_{name}", round(value, 6))

        _log_metrics_table("Test", test_metrics, k_values)

        # ---- save & log model artifacts -------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("Saving model artifacts...")
        logger.info("=" * 80)

        artifact_dir = Path("svd_model_artifacts")
        artifact_dir.mkdir(exist_ok=True)

        model.save(artifact_dir)

        # Summary JSON alongside model artifacts
        summary = {
            "model_type": "SVDRecommender",
            "dataset_tag": dataset_tag,
            "factors": int(model.user_factors_.shape[1]),
            "n_users": model.n_users_,
            "n_items": model.n_items_,
            "train_time_sec": round(train_time, 2),
            "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
        }
        with open(artifact_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifacts(str(artifact_dir))
        shutil.rmtree(artifact_dir)

        logger.info("Artifacts logged to MLflow.")

        # ---- final summary --------------------------------------------------
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model      : SVDRecommender (k={model.user_factors_.shape[1]})")
        logger.info(f"Dataset    : {dataset_tag}")
        logger.info(f"Sample frac: {sample_frac or 1.0}")
        logger.info(f"Train time : {train_time:.2f}s")
        logger.info(f"Users      : {model.n_users_:,}")
        logger.info(f"Items      : {model.n_items_:,}")
        logger.info("=" * 80)

        return test_metrics


# ---------------------------------------------------------------------------
# Logging helper
# ---------------------------------------------------------------------------


def _log_metrics_table(split: str, metrics: Dict[str, float], k_values: List[int]) -> None:
    logger.info(f"\n{split} metrics:")
    header = f"  {'K':>4}  {'Precision':>10}  {'Recall':>10}  {'NDCG':>10}  {'MAP':>10}  {'Coverage':>10}"
    logger.info(header)
    logger.info("  " + "-" * (len(header) - 2))
    for k in k_values:
        logger.info(
            f"  {k:>4}  "
            f"{metrics.get(f'precision_at_{k}', 0):>10.4f}  "
            f"{metrics.get(f'recall_at_{k}', 0):>10.4f}  "
            f"{metrics.get(f'ndcg_at_{k}', 0):>10.4f}  "
            f"{metrics.get(f'map_at_{k}', 0):>10.4f}  "
            f"{metrics.get(f'coverage_at_{k}', 0):>10.4f}"
        )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Entry point for ``python -m src.training.train_collaborative``."""
    parser = argparse.ArgumentParser(
        description="Train Collaborative Filtering (SVD) recommender",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--dataset-tag",
        type=str,
        required=True,
        help="Feature store version tag, e.g. ml_v_20260215_184134",
    )
    parser.add_argument(
        "--factors",
        type=int,
        default=100,
        help="Number of SVD latent factors",
    )
    parser.add_argument(
        "--sample-frac",
        type=float,
        default=None,
        help="Fraction of training rows to use (None = full dataset)",
    )
    parser.add_argument(
        "--relevance-threshold",
        type=float,
        default=4.0,
        help="Minimum rating to count as relevant for metrics",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[5, 10, 20],
        help="Cut-off values for ranking metrics",
    )
    parser.add_argument(
        "--max-eval-users",
        type=int,
        default=1000,
        help="Max users to evaluate on val/test per split",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default="collaborative_filtering",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help="MLflow run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()

    train_and_evaluate(
        dataset_tag=args.dataset_tag,
        factors=args.factors,
        sample_frac=args.sample_frac,
        relevance_threshold=args.relevance_threshold,
        k_values=args.k_values,
        max_eval_users=args.max_eval_users,
        mlflow_experiment=args.experiment,
        run_name=args.run_name,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
