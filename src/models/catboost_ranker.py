"""CatBoost Ranker — Stage 2 Re-ranker.

Learns to re-rank a short list of pre-filtered candidates (from ALS/SVD)
into a final top-K recommendation list.

Input to fit():
    X         — feature matrix  (n_rows, n_features)
                Each row = user_features ⊕ item_features ⊕ [als_score]
    y         — relevance labels (0 / 1 or graded)
    group_ids — user IDs, used by CatBoost to form ranking groups

Loss:
    YetiRank  — CatBoost's pairwise ranking objective, strong on NDCG
    Alternative: 'PairLogit' for smoother gradients

Usage:
    ranker = CatBoostRanker(iterations=500, learning_rate=0.05)
    ranker.fit(X_train, y_train, group_ids_train, eval_set=(X_val, y_val, gids_val))
    scores = ranker.predict(X_candidates)
    explanation = ranker.explain(X_row)   # SHAP top-3 features
"""
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Feature column definitions — must match feature store schema
USER_FEATURE_COLS: List[str] = [
    "user_avg_rating",
    "user_rating_std",
    "user_num_ratings",
    "user_min_rating",
    "user_max_rating",
    "user_activity_days",
    "user_rating_velocity",
]

ITEM_FEATURE_COLS: List[str] = [
    "movie_avg_rating",
    "movie_rating_std",
    "movie_num_ratings",
    "movie_num_users",
    "movie_popularity",
    "year",
    "movie_age",
    "decade",
    "title_length",
    "num_genres",
    "genre_drama",
    "genre_comedy",
    "genre_action",
    "genre_thriller",
    "genre_adventure",
    "genre_romance",
    "genre_sci_fi",
    "genre_crime",
    "genre_fantasy",
    "genre_children",
    "genre_mystery",
    "genre_horror",
    "genre_animation",
    "genre_war",
    "genre_imax",
    "genre_musical",
    "genre_western",
    "genre_documentary",
    "genre_film_noir",
    "genre_(no genres listed)",
]

# All features fed into the ranker (order matters — must be consistent)
RANKER_FEATURE_COLS: List[str] = USER_FEATURE_COLS + ITEM_FEATURE_COLS + ["als_score"]


class CatBoostRanker:
    """CatBoost-based learning-to-rank model for candidate re-ranking.

    Args:
        iterations: Number of boosting iterations.
        learning_rate: Gradient boosting learning rate.
        depth: Tree depth.
        loss_function: CatBoost ranking loss ('YetiRank' or 'PairLogit').
        early_stopping_rounds: Stop if val metric doesn't improve.
        random_seed: Reproducibility seed.
        verbose: Logging interval (0 = silent).
    """

    def __init__(
        self,
        iterations: int = 500,
        learning_rate: float = 0.05,
        depth: int = 6,
        loss_function: str = "YetiRank",
        early_stopping_rounds: int = 50,
        random_seed: int = 42,
        verbose: int = 100,
    ) -> None:
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.loss_function = loss_function
        self.early_stopping_rounds = early_stopping_rounds
        self.random_seed = random_seed
        self.verbose = verbose
        self.is_fitted_: bool = False
        self._model = None
        self.feature_names_: List[str] = RANKER_FEATURE_COLS.copy()
        self.best_iteration_: Optional[int] = None

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        group_ids_train: pd.Series,
        eval_set: Optional[Tuple[pd.DataFrame, pd.Series, pd.Series]] = None,
    ) -> "CatBoostRanker":
        """Train the ranker.

        Args:
            X_train: Feature matrix, columns must include RANKER_FEATURE_COLS.
            y_train: Binary or graded relevance labels (same index as X_train).
            group_ids_train: User IDs defining ranking groups.
            eval_set: Optional (X_val, y_val, group_ids_val) for early stopping.

        Returns:
            self
        """
        try:
            from catboost import CatBoost, Pool
        except ImportError:
            raise ImportError("catboost is required: pip install catboost")

        feature_cols = [c for c in RANKER_FEATURE_COLS if c in X_train.columns]
        self.feature_names_ = feature_cols

        logger.info(
            f"Building ranker training pool: {len(X_train):,} rows, "
            f"{len(feature_cols)} features, "
            f"{group_ids_train.nunique():,} users"
        )

        train_pool = Pool(
            data=X_train[feature_cols].values,
            label=y_train.values,
            group_id=group_ids_train.values,
            feature_names=feature_cols,
        )

        eval_pool = None
        if eval_set is not None:
            X_val, y_val, gids_val = eval_set
            eval_pool = Pool(
                data=X_val[feature_cols].values,
                label=y_val.values,
                group_id=gids_val.values,
                feature_names=feature_cols,
            )

        params = {
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "loss_function": self.loss_function,
            "eval_metric": "NDCG",
            "random_seed": self.random_seed,
            "early_stopping_rounds": self.early_stopping_rounds,
            "verbose": self.verbose,
            "use_best_model": eval_pool is not None,
        }

        logger.info(
            f"Training CatBoost Ranker: loss={self.loss_function}, "
            f"iterations={self.iterations}, depth={self.depth}, "
            f"lr={self.learning_rate}"
        )

        self._model = CatBoost(params)
        self._model.fit(
            train_pool,
            eval_set=eval_pool,
        )

        self.best_iteration_ = self._model.get_best_iteration()
        self.is_fitted_ = True

        logger.info(
            f"CatBoost Ranker trained — best_iteration={self.best_iteration_}"
        )
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores for candidates.

        Args:
            X: Feature matrix (same columns as training).

        Returns:
            1-D array of scores (higher = more relevant).
        """
        self._check_fitted()
        feature_cols = [c for c in self.feature_names_ if c in X.columns]
        return self._model.predict(X[feature_cols].values)

    def explain(
        self, X: pd.DataFrame, top_n: int = 3
    ) -> List[Dict[str, float]]:
        """SHAP-based explanation for each row.

        Args:
            X: Feature matrix (n rows to explain).
            top_n: Number of top features to return per row.

        Returns:
            List of dicts  {feature_name: shap_value}  one per row.
        """
        self._check_fitted()
        feature_cols = [c for c in self.feature_names_ if c in X.columns]
        shap_values = self._model.get_feature_importance(
            data=X[feature_cols].values,
            type="ShapValues",
        )
        # shap_values shape: (n_rows, n_features + 1) — last col is bias
        shap_values = shap_values[:, :-1]

        explanations = []
        for row_shap in shap_values:
            top_indices = np.argsort(np.abs(row_shap))[::-1][:top_n]
            explanations.append(
                {feature_cols[i]: round(float(row_shap[i]), 4) for i in top_indices}
            )
        return explanations

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending."""
        self._check_fitted()
        importances = self._model.get_feature_importance()
        return (
            pd.DataFrame(
                {"feature": self.feature_names_, "importance": importances}
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: Path) -> None:
        """Save model to directory.

        Artifacts:
        - ``catboost_model.cbm``: native CatBoost format
        - ``ranker_config.json``: hyper-parameters + feature names
        """
        self._check_fitted()
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._model.save_model(str(path / "catboost_model.cbm"))

        config = {
            "model_type": "CatBoostRanker",
            "iterations": self.iterations,
            "learning_rate": self.learning_rate,
            "depth": self.depth,
            "loss_function": self.loss_function,
            "early_stopping_rounds": self.early_stopping_rounds,
            "random_seed": self.random_seed,
            "best_iteration": self.best_iteration_,
            "feature_names": self.feature_names_,
        }
        with open(path / "ranker_config.json", "w") as f:
            json.dump(config, f, indent=2)

        logger.info(f"CatBoostRanker saved to {path}")

    @classmethod
    def load(cls, path: Path) -> "CatBoostRanker":
        """Load CatBoostRanker from directory written by :meth:`save`."""
        try:
            from catboost import CatBoost
        except ImportError:
            raise ImportError("catboost is required: pip install catboost")

        path = Path(path)
        with open(path / "ranker_config.json") as f:
            config = json.load(f)

        ranker = cls(
            iterations=config["iterations"],
            learning_rate=config["learning_rate"],
            depth=config["depth"],
            loss_function=config["loss_function"],
            early_stopping_rounds=config["early_stopping_rounds"],
            random_seed=config["random_seed"],
        )
        ranker._model = CatBoost()
        ranker._model.load_model(str(path / "catboost_model.cbm"))
        ranker.feature_names_ = config["feature_names"]
        ranker.best_iteration_ = config.get("best_iteration")
        ranker.is_fitted_ = True

        logger.info(f"CatBoostRanker loaded from {path}")
        return ranker

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _check_fitted(self) -> None:
        if not self.is_fitted_:
            raise ValueError("Model not fitted. Call fit() first.")

    def __repr__(self) -> str:
        status = "fitted" if self.is_fitted_ else "not fitted"
        details = ""
        if self.is_fitted_ and self.best_iteration_ is not None:
            details = f", best_iter={self.best_iteration_}, loss={self.loss_function}"
        return f"CatBoostRanker({status}{details})"
