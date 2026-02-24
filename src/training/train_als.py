"""Train iALS (implicit ALS) Candidate Generator with MLflow tracking.

iALS is stage 1 of the two-stage recommendation pipeline:
    iALS  →  top-N candidates  →  CatBoost Ranker  →  top-K

This script trains and evaluates the iALS model standalone.
To train the full two-stage pipeline, use train_ranker.py instead.

Example:
    # Quick dev run (10% data, linear confidence)
    python -m src.training.train_als \\
        --dataset-tag ml_v_20260215_184134 \\
        --sample-frac 0.1 \\
        --factors 64 --iterations 15

    # Full production run (log-confidence variant)
    python -m src.training.train_als \\
        --dataset-tag ml_v_20260215_184134 \\
        --factors 128 --iterations 20 \\
        --confidence-mode log --alpha 10.0
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
from src.models.als_recommender import ALSRecommender

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(
    dataset_tag: str,
    sample_frac: Optional[float] = None,
    sample_seed: int = 42,
) -> Dict[str, pd.DataFrame]:
    version_dir = FEATURE_STORE_PATH / dataset_tag
    if not version_dir.exists():
        raise FileNotFoundError(f"Feature store not found: {version_dir}")

    logger.info("=" * 80)
    logger.info(f"Loading feature store: {dataset_tag}")
    logger.info("=" * 80)

    train_df = pd.read_parquet(version_dir / "train.parquet")
    val_df = pd.read_parquet(version_dir / "val.parquet")
    test_df = pd.read_parquet(version_dir / "test.parquet")

    metadata: Dict = {}
    meta_path = version_dir / "metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            metadata = json.load(f)

    if sample_frac and 0.0 < sample_frac < 1.0:
        orig = len(train_df)
        train_df = train_df.sample(frac=sample_frac, random_state=sample_seed)
        logger.info(f"Sampled train: {orig:,} → {len(train_df):,} rows ({sample_frac:.0%})")

    logger.info(f"  train : {len(train_df):,} rows  ─  "
                f"{train_df['userId'].nunique():,} users / {train_df['movieId'].nunique():,} items")
    logger.info(f"  val   : {len(val_df):,} rows")
    logger.info(f"  test  : {len(test_df):,} rows")
    logger.info("=" * 80)

    return {"train": train_df, "val": val_df, "test": test_df, "metadata": metadata}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_recommendations(
    model: ALSRecommender,
    eval_df: pd.DataFrame,
    seen_df: pd.DataFrame,
    max_users: int,
    k: int,
) -> Dict[int, List[int]]:
    known = set(model.user_id_map_.keys())
    eval_users = [u for u in eval_df["userId"].unique() if u in known][:max_users]
    logger.info(f"Generating recs for {len(eval_users):,} users (k={k})...")

    exclude = seen_df.groupby("userId")["movieId"].apply(set).to_dict()
    t0 = time.time()
    recs = model.recommend_batch(user_ids=eval_users, n=k,
                                  exclude_items_per_user=exclude)
    logger.info(f"Done in {time.time()-t0:.2f}s")
    return recs


def _log_table(split: str, metrics: Dict, k_values: List[int]) -> None:
    logger.info(f"\n{split} metrics:")
    logger.info(f"  {'K':>4}  {'Precision':>10}  {'Recall':>10}  {'NDCG':>10}  "
                f"{'MAP':>10}  {'Coverage':>10}")
    logger.info("  " + "-" * 62)
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
# Main
# ---------------------------------------------------------------------------

def train_and_evaluate(
    dataset_tag: str,
    factors: int = 128,
    iterations: int = 20,
    regularization: float = 0.01,
    alpha: float = 15.0,
    confidence_mode: str = "linear",
    sample_frac: Optional[float] = None,
    relevance_threshold: float = 4.0,
    k_values: List[int] = [5, 10, 20],
    max_eval_users: int = 1000,
    mlflow_experiment: str = "als_candidate_generator",
    run_name: Optional[str] = None,
    seed: int = 42,
) -> Dict[str, float]:
    np.random.seed(seed)

    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_str = f"_s{int(sample_frac * 100)}" if sample_frac else "_full"
        run_name = f"als_k{factors}_{confidence_mode}{sample_str}_{ts}"

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        logger.info(f"MLflow run : {run_name}")
        logger.info(f"Experiment : {mlflow_experiment}")

        mlflow.log_param("model_type", "ALSRecommender")
        mlflow.log_param("dataset_tag", dataset_tag)
        mlflow.log_param("factors", factors)
        mlflow.log_param("iterations", iterations)
        mlflow.log_param("regularization", regularization)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("confidence_mode", confidence_mode)
        mlflow.log_param("sample_frac", sample_frac or 1.0)
        mlflow.log_param("relevance_threshold", relevance_threshold)
        mlflow.log_param("seed", seed)

        # Load data
        t0 = time.time()
        data = load_data(dataset_tag, sample_frac=sample_frac, sample_seed=seed)
        mlflow.log_metric("data_load_time_sec", round(time.time() - t0, 2))
        train_df, val_df, test_df = data["train"], data["val"], data["test"]

        mlflow.log_metric("n_train", len(train_df))
        mlflow.log_metric("n_users_train", int(train_df["userId"].nunique()))
        mlflow.log_metric("n_items_train", int(train_df["movieId"].nunique()))

        # Train
        logger.info("\n" + "=" * 80)
        logger.info("Training ALSRecommender (iALS)...")
        logger.info("=" * 80)

        model = ALSRecommender(
            factors=factors,
            iterations=iterations,
            regularization=regularization,
            alpha=alpha,
            confidence_mode=confidence_mode,
            random_state=seed,
        )

        t_train = time.time()
        model.fit(train_df)
        train_time = round(time.time() - t_train, 2)
        mlflow.log_metric("train_time_sec", train_time)
        logger.info(f"Train time: {train_time}s | {model}")

        catalog = set(int(x) for x in train_df["movieId"].unique())

        # Validation
        logger.info("\nEvaluating on VALIDATION set...")
        val_recs = _build_recommendations(model, val_df, train_df, max_eval_users, max(k_values))
        val_gt = create_ground_truth(
            val_df[val_df["userId"].isin(val_recs.keys())],
            relevance_threshold=relevance_threshold,
        )
        val_metrics = evaluate_recommendations(val_recs, val_gt, catalog, k_values)
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", round(v, 6))
        _log_table("Validation", val_metrics, k_values)

        # Test
        logger.info("\nEvaluating on TEST set...")
        train_val = pd.concat([train_df, val_df], ignore_index=True)
        test_recs = _build_recommendations(model, test_df, train_val, max_eval_users, max(k_values))
        test_gt = create_ground_truth(
            test_df[test_df["userId"].isin(test_recs.keys())],
            relevance_threshold=relevance_threshold,
        )
        test_metrics = evaluate_recommendations(test_recs, test_gt, catalog, k_values)
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", round(v, 6))
        _log_table("Test", test_metrics, k_values)

        # Save artifacts
        artifact_dir = Path("als_model_artifacts")
        artifact_dir.mkdir(exist_ok=True)
        model.save(artifact_dir)

        summary = {
            "model_type": "ALSRecommender",
            "dataset_tag": dataset_tag,
            "factors": factors,
            "confidence_mode": confidence_mode,
            "alpha": alpha,
            "n_users": model.n_users_,
            "n_items": model.n_items_,
            "train_time_sec": train_time,
            "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
        }
        with open(artifact_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        mlflow.log_artifacts(str(artifact_dir))
        shutil.rmtree(artifact_dir)

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Model      : ALSRecommender (iALS)")
        logger.info(f"Factors    : {factors}, confidence={confidence_mode}, alpha={alpha}")
        logger.info(f"Train time : {train_time}s")
        logger.info(f"Users/Items: {model.n_users_:,} / {model.n_items_:,}")
        logger.info("=" * 80)

        return test_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train iALS candidate generator",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-tag", type=str, required=True)
    parser.add_argument("--factors", type=int, default=128)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--regularization", type=float, default=0.01)
    parser.add_argument("--alpha", type=float, default=15.0,
                        help="Confidence scaling factor")
    parser.add_argument("--confidence-mode", type=str, default="linear",
                        choices=["linear", "log", "binary"],
                        help="How to compute confidence from ratings")
    parser.add_argument("--sample-frac", type=float, default=None)
    parser.add_argument("--relevance-threshold", type=float, default=4.0)
    parser.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    parser.add_argument("--max-eval-users", type=int, default=1000)
    parser.add_argument("--experiment", type=str, default="als_candidate_generator")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    train_and_evaluate(
        dataset_tag=args.dataset_tag,
        factors=args.factors,
        iterations=args.iterations,
        regularization=args.regularization,
        alpha=args.alpha,
        confidence_mode=args.confidence_mode,
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
