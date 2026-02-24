"""Train Two-Stage Recommender: iALS → CatBoost Ranker.

Full pipeline:
    1. Train iALS on train split  (candidate generator)
    2. Build ranker training data via ALS-driven negative sampling
    3. Train CatBoost YetiRank on candidate-level features
    4. Evaluate end-to-end two-stage pipeline on val + test
    5. Log all params, metrics, and artifacts to MLflow

Ranker training data construction
    For each sampled user:
      - positives : items rated >= relevance_threshold in train
      - candidates: ALS top-N (hard negatives + some positives mixed in)
      - label      : 1 if in positives, else 0
      - features   : user_features ⊕ item_features ⊕ [als_score]
      - group      : userId  (required by YetiRank)

MLflow experiments:
    als_candidate_generator  ← stage-1 run (nested)
    two_stage_ranker         ← parent run with end-to-end metrics

Example:
    # Dev run (10% data, ~5-10 min total)
    python -m src.training.train_ranker \\
        --dataset-tag ml_v_20260215_184134 \\
        --sample-frac 0.1 \\
        --als-factors 64 --ranker-iterations 300

    # Production run
    python -m src.training.train_ranker \\
        --dataset-tag ml_v_20260215_184134 \\
        --als-factors 128 --ranker-iterations 600 \\
        --n-candidates 300 --max-ranker-users 10000
"""
import argparse
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd

from src.config import FEATURE_STORE_PATH
from src.evaluation.metrics import create_ground_truth, evaluate_recommendations
from src.models.als_recommender import ALSRecommender
from src.models.catboost_ranker import (
    CatBoostRanker,
    RANKER_FEATURE_COLS,
    USER_FEATURE_COLS,
    ITEM_FEATURE_COLS,
)
from src.models.two_stage_recommender import TwoStageRecommender

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
        logger.info(f"Sampled train: {orig:,} → {len(train_df):,} ({sample_frac:.0%})")

    logger.info(f"  train : {len(train_df):,}  ─  "
                f"{train_df['userId'].nunique():,} users / {train_df['movieId'].nunique():,} items")
    logger.info(f"  val   : {len(val_df):,}")
    logger.info(f"  test  : {len(test_df):,}")
    logger.info("=" * 80)

    return {"train": train_df, "val": val_df, "test": test_df, "metadata": metadata}


# ---------------------------------------------------------------------------
# Feature table helpers
# ---------------------------------------------------------------------------

def build_feature_lookup_tables(
    train_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Build per-user and per-item feature lookup tables from train split.

    Returns:
        user_features: DataFrame indexed by userId
        item_features: DataFrame indexed by movieId
    """
    user_cols = [c for c in USER_FEATURE_COLS if c in train_df.columns]
    item_cols = [c for c in ITEM_FEATURE_COLS if c in train_df.columns]

    user_features = (
        train_df[["userId"] + user_cols]
        .drop_duplicates("userId")
        .set_index("userId")
        .astype(np.float32)
    )
    item_features = (
        train_df[["movieId"] + item_cols]
        .drop_duplicates("movieId")
        .set_index("movieId")
        .astype(np.float32)
    )

    logger.info(f"User feature table: {user_features.shape}  ({len(user_cols)} features)")
    logger.info(f"Item feature table: {item_features.shape}  ({len(item_cols)} features)")
    return user_features, item_features


# ---------------------------------------------------------------------------
# Ranker dataset construction
# ---------------------------------------------------------------------------

def build_ranker_dataset(
    als_model: ALSRecommender,
    train_df: pd.DataFrame,
    user_features: pd.DataFrame,
    item_features: pd.DataFrame,
    n_candidates: int = 300,
    max_users: int = 8000,
    relevance_threshold: float = 4.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Build (user, candidate_item, features, label) dataset for CatBoost.

    For every sampled user:
      - ALS retrieves top-N candidates (no exclusion — hard negatives included)
      - Label = 1 if the candidate appears in the user's positives, else 0
      - Feature row = user_features ⊕ item_features ⊕ [als_score]

    Groups are sorted by userId (required by CatBoost YetiRank: groups must
    appear as contiguous blocks, sorted ascending).

    Args:
        als_model: Fitted ALSRecommender.
        train_df: Training split (to determine positives).
        user_features: Per-user feature DataFrame (indexed by userId).
        item_features: Per-item feature DataFrame (indexed by movieId).
        n_candidates: ALS candidates per user (hard negatives + positives).
        max_users: Cap on number of training users (memory/speed tradeoff).
        relevance_threshold: Min rating to count as relevant.
        seed: RNG seed for user sampling.

    Returns:
        DataFrame with columns: userId, label, *user_feat, *item_feat, als_score
        Sorted by userId ascending (YetiRank requirement).
    """
    rng = np.random.default_rng(seed)

    # Sample users — prefer users with enough positives
    positive_counts = (
        train_df[train_df["rating"] >= relevance_threshold]
        .groupby("userId")["movieId"]
        .count()
    )
    eligible_users = positive_counts[positive_counts >= 5].index.tolist()
    eligible_users = [u for u in eligible_users if u in als_model.user_id_map_]

    if len(eligible_users) > max_users:
        eligible_users = rng.choice(eligible_users, max_users, replace=False).tolist()

    logger.info(
        f"Building ranker dataset: {len(eligible_users):,} users × "
        f"{n_candidates} candidates"
    )

    # Pre-build positives dict
    positives_dict: Dict[int, set] = (
        train_df[train_df["rating"] >= relevance_threshold]
        .groupby("userId")["movieId"]
        .apply(set)
        .to_dict()
    )

    user_feat_cols = [c for c in USER_FEATURE_COLS if c in user_features.columns]
    item_feat_cols = [c for c in ITEM_FEATURE_COLS if c in item_features.columns]

    rows = []
    n_pos_total = 0
    n_neg_total = 0

    for user_id in eligible_users:
        uid = int(user_id)

        if uid not in user_features.index:
            continue

        # Stage 1: get candidates WITH scores, no seen-item exclusion
        candidate_ids, als_scores = als_model.recommend_with_scores(
            user_id=uid,
            n=n_candidates,
            filter_already_liked=False,  # keep positives in candidate pool
        )
        if not candidate_ids:
            continue

        uf_vals = user_features.loc[uid].values  # (n_user_feats,)
        pos_set = positives_dict.get(uid, set())

        for item_id, score in zip(candidate_ids, als_scores):
            iid = int(item_id)
            if iid not in item_features.index:
                continue
            if_vals = item_features.loc[iid].values  # (n_item_feats,)
            label = 1 if iid in pos_set else 0

            row = [uid, label] + uf_vals.tolist() + if_vals.tolist() + [float(score)]
            rows.append(row)

            if label == 1:
                n_pos_total += 1
            else:
                n_neg_total += 1

    if not rows:
        raise RuntimeError("Ranker dataset is empty — check ALS model and feature tables.")

    col_names = ["userId", "label"] + user_feat_cols + item_feat_cols + ["als_score"]
    df = pd.DataFrame(rows, columns=col_names)

    # Sort by userId — CatBoost YetiRank requires contiguous groups sorted asc
    df = df.sort_values("userId").reset_index(drop=True)

    pos_rate = 100.0 * n_pos_total / (n_pos_total + n_neg_total)
    logger.info(
        f"Ranker dataset: {len(df):,} rows, "
        f"{n_pos_total:,} positives ({pos_rate:.1f}%), "
        f"{n_neg_total:,} negatives"
    )
    return df


# ---------------------------------------------------------------------------
# End-to-end evaluation helper
# ---------------------------------------------------------------------------

def evaluate_two_stage(
    two_stage: TwoStageRecommender,
    eval_df: pd.DataFrame,
    seen_df: pd.DataFrame,
    catalog: set,
    max_users: int,
    n_candidates: int,
    k_values: List[int],
    relevance_threshold: float,
) -> Dict[str, float]:
    known_users = set(two_stage.candidate_model.user_id_map_.keys())
    eval_users = [u for u in eval_df["userId"].unique() if u in known_users][:max_users]
    logger.info(f"End-to-end evaluation: {len(eval_users):,} users...")

    exclude_per_user = seen_df.groupby("userId")["movieId"].apply(set).to_dict()

    t0 = time.time()
    batch_results = two_stage.recommend_batch(
        user_ids=eval_users,
        n=max(k_values),
        n_candidates=n_candidates,
        exclude_items_per_user=exclude_per_user,
    )
    elapsed = time.time() - t0
    logger.info(f"Recs generated in {elapsed:.2f}s ({1000*elapsed/max(len(eval_users),1):.1f}ms/user)")

    # Convert to {user_id: [item_id, ...]} for evaluate_recommendations
    recommendations = {
        uid: [r["item_id"] for r in recs]
        for uid, recs in batch_results.items()
    }
    ground_truth = create_ground_truth(
        eval_df[eval_df["userId"].isin(recommendations.keys())],
        relevance_threshold=relevance_threshold,
    )
    return evaluate_recommendations(recommendations, ground_truth, catalog, k_values)


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
# Main training function
# ---------------------------------------------------------------------------

def train_and_evaluate(
    dataset_tag: str,
    # ALS params
    als_factors: int = 128,
    als_iterations: int = 20,
    als_regularization: float = 0.01,
    als_alpha: float = 15.0,
    als_confidence_mode: str = "linear",
    # Ranker params
    ranker_iterations: int = 500,
    ranker_learning_rate: float = 0.05,
    ranker_depth: int = 6,
    ranker_loss: str = "YetiRank",
    # Dataset params
    n_candidates: int = 300,
    max_ranker_users: int = 8000,
    relevance_threshold: float = 4.0,
    k_values: List[int] = [5, 10, 20],
    max_eval_users: int = 1000,
    # Run params
    sample_frac: Optional[float] = None,
    mlflow_experiment: str = "two_stage_ranker",
    run_name: Optional[str] = None,
    seed: int = 42,
    model_save_path: str = "data/models/two_stage_ranker",
) -> Dict[str, float]:
    np.random.seed(seed)

    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        sample_str = f"_s{int(sample_frac * 100)}" if sample_frac else "_full"
        run_name = f"als{als_factors}_{ranker_loss.lower()}_c{n_candidates}{sample_str}_{ts}"

    mlflow.set_experiment(mlflow_experiment)

    with mlflow.start_run(run_name=run_name):
        logger.info("=" * 80)
        logger.info(f"MLflow run : {run_name}")
        logger.info(f"Experiment : {mlflow_experiment}")
        logger.info("=" * 80)

        # ── log all hyper-parameters ────────────────────────────────────
        mlflow.log_params({
            "dataset_tag": dataset_tag,
            "als_factors": als_factors,
            "als_iterations": als_iterations,
            "als_regularization": als_regularization,
            "als_alpha": als_alpha,
            "als_confidence_mode": als_confidence_mode,
            "ranker_iterations": ranker_iterations,
            "ranker_learning_rate": ranker_learning_rate,
            "ranker_depth": ranker_depth,
            "ranker_loss": ranker_loss,
            "n_candidates": n_candidates,
            "max_ranker_users": max_ranker_users,
            "relevance_threshold": relevance_threshold,
            "sample_frac": sample_frac or 1.0,
            "seed": seed,
        })

        # ── load data ────────────────────────────────────────────────────
        t0 = time.time()
        data = load_data(dataset_tag, sample_frac=sample_frac, sample_seed=seed)
        mlflow.log_metric("data_load_time_sec", round(time.time() - t0, 2))
        train_df, val_df, test_df = data["train"], data["val"], data["test"]

        mlflow.log_metric("n_train", len(train_df))
        mlflow.log_metric("n_users_train", int(train_df["userId"].nunique()))
        mlflow.log_metric("n_items_train", int(train_df["movieId"].nunique()))

        # ── build feature lookup tables ──────────────────────────────────
        logger.info("\nBuilding feature lookup tables...")
        user_features, item_features = build_feature_lookup_tables(train_df)

        # ── stage 1: train ALS ───────────────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("Stage 1 — Training iALS Candidate Generator")
        logger.info("=" * 80)

        als_model = ALSRecommender(
            factors=als_factors,
            iterations=als_iterations,
            regularization=als_regularization,
            alpha=als_alpha,
            confidence_mode=als_confidence_mode,
            random_state=seed,
        )
        t_als = time.time()
        als_model.fit(train_df)
        als_time = round(time.time() - t_als, 2)
        mlflow.log_metric("als_train_time_sec", als_time)
        logger.info(f"ALS done in {als_time}s | {als_model}")

        # ── build ranker training dataset ────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("Building ranker training dataset (ALS-driven negatives)...")
        logger.info("=" * 80)

        t_build = time.time()
        ranker_df = build_ranker_dataset(
            als_model=als_model,
            train_df=train_df,
            user_features=user_features,
            item_features=item_features,
            n_candidates=n_candidates,
            max_users=max_ranker_users,
            relevance_threshold=relevance_threshold,
            seed=seed,
        )
        build_time = round(time.time() - t_build, 2)
        mlflow.log_metric("ranker_dataset_build_time_sec", build_time)
        mlflow.log_metric("ranker_n_rows", len(ranker_df))
        mlflow.log_metric("ranker_n_users", int(ranker_df["userId"].nunique()))
        mlflow.log_metric(
            "ranker_pos_rate",
            round(ranker_df["label"].mean(), 4)
        )
        logger.info(f"Dataset built in {build_time}s")

        # Train/val split on ranker dataset (last 15% of users by ID → val)
        unique_users_sorted = np.sort(ranker_df["userId"].unique())
        val_cutoff = unique_users_sorted[int(len(unique_users_sorted) * 0.85)]
        ranker_train_mask = ranker_df["userId"] < val_cutoff
        X_rt = ranker_df[ranker_train_mask]
        X_rv = ranker_df[~ranker_train_mask]

        feature_cols = [c for c in RANKER_FEATURE_COLS if c in ranker_df.columns]

        logger.info(
            f"Ranker split: train={len(X_rt):,} rows "
            f"({X_rt['userId'].nunique():,} users), "
            f"val={len(X_rv):,} rows ({X_rv['userId'].nunique():,} users)"
        )

        # ── stage 2: train CatBoost Ranker ───────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("Stage 2 — Training CatBoost Ranker")
        logger.info("=" * 80)

        ranker = CatBoostRanker(
            iterations=ranker_iterations,
            learning_rate=ranker_learning_rate,
            depth=ranker_depth,
            loss_function=ranker_loss,
            random_seed=seed,
            verbose=100,
        )
        t_ranker = time.time()
        ranker.fit(
            X_train=X_rt[feature_cols],
            y_train=X_rt["label"],
            group_ids_train=X_rt["userId"],
            eval_set=(X_rv[feature_cols], X_rv["label"], X_rv["userId"]),
        )
        ranker_time = round(time.time() - t_ranker, 2)
        mlflow.log_metric("ranker_train_time_sec", ranker_time)
        mlflow.log_metric("ranker_best_iteration", ranker.best_iteration_ or ranker_iterations)
        logger.info(f"Ranker done in {ranker_time}s | {ranker}")

        # Log feature importances
        fi_df = ranker.feature_importance()
        logger.info("\nTop-10 feature importances:")
        for _, row in fi_df.head(10).iterrows():
            logger.info(f"  {row['feature']:35s} {row['importance']:.2f}")

        fi_path = Path("feature_importances.csv")
        fi_df.to_csv(fi_path, index=False)
        mlflow.log_artifact(str(fi_path))
        fi_path.unlink()

        # ── assemble two-stage model ─────────────────────────────────────
        two_stage = TwoStageRecommender(
            candidate_model=als_model,
            ranker=ranker,
            user_features=user_features,
            item_features=item_features,
        )

        catalog = set(int(x) for x in train_df["movieId"].unique())

        # ── end-to-end validation evaluation ────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("End-to-end evaluation on VALIDATION set...")
        logger.info("=" * 80)

        val_metrics = evaluate_two_stage(
            two_stage, val_df, train_df,
            catalog, max_eval_users, n_candidates,
            k_values, relevance_threshold,
        )
        for k, v in val_metrics.items():
            mlflow.log_metric(f"val_{k}", round(v, 6))
        _log_table("Validation (two-stage)", val_metrics, k_values)

        # ── end-to-end test evaluation ───────────────────────────────────
        logger.info("\n" + "=" * 80)
        logger.info("End-to-end evaluation on TEST set...")
        logger.info("=" * 80)

        train_val = pd.concat([train_df, val_df], ignore_index=True)
        test_metrics = evaluate_two_stage(
            two_stage, test_df, train_val,
            catalog, max_eval_users, n_candidates,
            k_values, relevance_threshold,
        )
        for k, v in test_metrics.items():
            mlflow.log_metric(f"test_{k}", round(v, 6))
        _log_table("Test (two-stage)", test_metrics, k_values)

        # ── save all artifacts ────────────────────────────────────────────
        logger.info("\nSaving model artifacts...")

        # Permanent on-disk location for the backend to load from
        permanent_dir = Path(model_save_path)
        permanent_dir.mkdir(parents=True, exist_ok=True)
        two_stage.save(permanent_dir)

        # Write training summary
        summary = {
            "model_type": "TwoStageRecommender",
            "dataset_tag": dataset_tag,
            "als_factors": als_factors,
            "als_confidence_mode": als_confidence_mode,
            "ranker_loss": ranker_loss,
            "ranker_best_iteration": ranker.best_iteration_,
            "n_candidates": n_candidates,
            "n_users": als_model.n_users_,
            "n_items": als_model.n_items_,
            "als_train_time_sec": als_time,
            "ranker_train_time_sec": ranker_time,
            "val_metrics": {k: round(v, 6) for k, v in val_metrics.items()},
            "test_metrics": {k: round(v, 6) for k, v in test_metrics.items()},
        }
        with open(permanent_dir / "training_summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Also log to MLflow
        mlflow.log_artifacts(str(permanent_dir))
        logger.info(f"Artifacts saved to '{permanent_dir}' and logged to MLflow.")

        logger.info("\n" + "=" * 80)
        logger.info("TRAINING SUMMARY")
        logger.info("=" * 80)
        logger.info(f"Stage 1 : ALSRecommender  factors={als_factors}, "
                    f"mode={als_confidence_mode}, time={als_time}s")
        logger.info(f"Stage 2 : CatBoostRanker  loss={ranker_loss}, "
                    f"best_iter={ranker.best_iteration_}, time={ranker_time}s")
        logger.info(f"Candidates per user : {n_candidates}")
        logger.info(f"Users / Items : {als_model.n_users_:,} / {als_model.n_items_:,}")
        logger.info("=" * 80)

        return test_metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train two-stage recommender: iALS + CatBoost Ranker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--dataset-tag", type=str, required=True)

    # ALS
    als = parser.add_argument_group("iALS (Stage 1)")
    als.add_argument("--als-factors", type=int, default=128)
    als.add_argument("--als-iterations", type=int, default=20)
    als.add_argument("--als-regularization", type=float, default=0.01)
    als.add_argument("--als-alpha", type=float, default=15.0)
    als.add_argument("--als-confidence-mode", type=str, default="linear",
                     choices=["linear", "log", "binary"])

    # Ranker
    rk = parser.add_argument_group("CatBoost Ranker (Stage 2)")
    rk.add_argument("--ranker-iterations", type=int, default=500)
    rk.add_argument("--ranker-learning-rate", type=float, default=0.05)
    rk.add_argument("--ranker-depth", type=int, default=6)
    rk.add_argument("--ranker-loss", type=str, default="YetiRank",
                    choices=["YetiRank", "PairLogit"])

    # Dataset
    ds = parser.add_argument_group("Dataset")
    ds.add_argument("--n-candidates", type=int, default=300,
                    help="ALS candidates per user for ranker training")
    ds.add_argument("--max-ranker-users", type=int, default=8000,
                    help="Max users to use for ranker training dataset")
    ds.add_argument("--relevance-threshold", type=float, default=4.0)
    ds.add_argument("--k-values", type=int, nargs="+", default=[5, 10, 20])
    ds.add_argument("--max-eval-users", type=int, default=1000)
    ds.add_argument("--sample-frac", type=float, default=None)

    # Run
    parser.add_argument("--experiment", type=str, default="two_stage_ranker")
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--model-save-path", type=str, default="data/models/two_stage_ranker",
                        help="Directory to permanently save the trained model (backend loads from here)")

    args = parser.parse_args()
    train_and_evaluate(
        dataset_tag=args.dataset_tag,
        als_factors=args.als_factors,
        als_iterations=args.als_iterations,
        als_regularization=args.als_regularization,
        als_alpha=args.als_alpha,
        als_confidence_mode=args.als_confidence_mode,
        ranker_iterations=args.ranker_iterations,
        ranker_learning_rate=args.ranker_learning_rate,
        ranker_depth=args.ranker_depth,
        ranker_loss=args.ranker_loss,
        n_candidates=args.n_candidates,
        max_ranker_users=args.max_ranker_users,
        relevance_threshold=args.relevance_threshold,
        k_values=args.k_values,
        max_eval_users=args.max_eval_users,
        sample_frac=args.sample_frac,
        mlflow_experiment=args.experiment,
        run_name=args.run_name,
        seed=args.seed,
        model_save_path=args.model_save_path,
    )


if __name__ == "__main__":
    main()
