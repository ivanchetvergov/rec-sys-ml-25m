"""Feature engineering module for MovieLens."""
import logging
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Creates features from preprocessed MovieLens data.

    Attributes:
        top_n_genres:  Number of top genres to create binary features for.
        top_genres_:   List of top genre names — set during ``create_movie_features``
                       and reused by ``apply_movie_features`` so val/test get the
                       *same* genre columns as train.
    """

    def __init__(self, top_n_genres: int = 20):
        """Initialize FeatureEngineer.

        Args:
            top_n_genres: Number of top genres to create binary features for.
        """
        self.top_n_genres = top_n_genres
        self.top_genres_: Optional[List[str]] = None   # set after first call to create_movie_features
        logger.info(f"FeatureEngineer initialized: top_n_genres={top_n_genres}")

    # ------------------------------------------------------------------
    # Movie metadata features (static — no rating data, no leakage)
    # ------------------------------------------------------------------

    def create_movie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create movie-level features.

        Stores ``self.top_genres_`` so subsequent splits can call
        :meth:`apply_movie_features` with the same genre vocabulary.

        Args:
            df: Input DataFrame with movie metadata (genres, year, title_clean).

        Returns:
            DataFrame with movie features added.
        """
        logger.info("Creating movie features...")

        # Parse genres into list
        df["genre_list"] = df["genres"].str.split("|")
        df["num_genres"] = df["genre_list"].apply(len)

        # Create binary genre features for top genres
        all_genres = df["genre_list"].explode().value_counts()
        top_genres = all_genres.head(self.top_n_genres).index.tolist()
        self.top_genres_ = top_genres  # save for reuse on val/test

        for genre in top_genres:
            col = f"genre_{genre.lower().replace('-', '_')}"
            df[col] = df["genre_list"].apply(lambda x: 1 if genre in x else 0)

        logger.info(f"Created {len(top_genres)} genre features")

        # Year-based features
        current_year = 2026
        df["movie_age"] = current_year - df["year"]
        df["decade"] = (df["year"] // 10) * 10

        # Fill missing years with median
        median_year = df["year"].median()
        df["year"] = df["year"].fillna(median_year)
        df["movie_age"] = df["movie_age"].fillna(current_year - median_year)

        # Title length
        df["title_length"] = df["title_clean"].str.len()

        logger.info("Created movie features")
        return df

    def apply_movie_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply the same movie metadata features to a new split.

        Must be called *after* :meth:`create_movie_features` has already been
        run on the training set (so ``self.top_genres_`` is populated).

        Args:
            df: Split DataFrame (e.g. val or test) with genre/year/title columns.

        Returns:
            DataFrame with movie features added using the training genre vocabulary.
        """
        if self.top_genres_ is None:
            raise RuntimeError(
                "call create_movie_features() on the training split first "
                "so that self.top_genres_ is populated."
            )

        df["genre_list"] = df["genres"].str.split("|")
        df["num_genres"] = df["genre_list"].apply(len)

        for genre in self.top_genres_:
            col = f"genre_{genre.lower().replace('-', '_')}"
            df[col] = df["genre_list"].apply(lambda x: 1 if genre in x else 0)

        current_year = 2026
        df["movie_age"] = current_year - df["year"]
        df["decade"] = (df["year"] // 10) * 10

        median_year = df["year"].median()
        df["year"] = df["year"].fillna(median_year)
        df["movie_age"] = df["movie_age"].fillna(current_year - median_year)

        df["title_length"] = df["title_clean"].str.len()
        return df

    # ------------------------------------------------------------------
    # User features — MUST be computed from train only
    # ------------------------------------------------------------------

    def compute_user_stats(
        self,
        train_df: pd.DataFrame,
        ref_timestamp: Optional[pd.Timestamp] = None,
    ) -> pd.DataFrame:
        """Compute per-user feature lookup table from the TRAINING split only.

        Keeping feature computation on train prevents data leakage: val/test
        ratings are never visible when deriving user statistics.

        Args:
            train_df: Training interactions.
            ref_timestamp: Reference point for ``user_days_since_last_rating``.
                Defaults to ``train_df["timestamp"].max()``.  Pass
                ``pd.Timestamp.now()`` for online / production inference.

        Returns:
            DataFrame indexed by ``userId`` with all user feature columns.
        """
        logger.info("Computing user stats from training split...")

        if ref_timestamp is None:
            ref_timestamp = train_df["timestamp"].max()

        user_stats = train_df.groupby("userId").agg({
            "rating": ["mean", "std", "count", "min", "max"],
            "timestamp": ["min", "max"]
        }).reset_index()

        user_stats.columns = [
            "userId",
            "user_avg_rating",
            "user_rating_std",
            "user_num_ratings",
            "user_min_rating",
            "user_max_rating",
            "user_first_interaction",
            "user_last_interaction",
        ]

        user_stats["user_activity_days"] = (
            (user_stats["user_last_interaction"] - user_stats["user_first_interaction"])
            .dt.total_seconds() / (24 * 3600)
        )
        user_stats["user_rating_velocity"] = (
            user_stats["user_num_ratings"]
            / user_stats["user_activity_days"].replace(0, 1)
        )
        user_stats["user_rating_std"] = user_stats["user_rating_std"].fillna(0)

        # Recency — days since each user's last rating relative to ref_timestamp
        user_stats["user_days_since_last_rating"] = (
            (ref_timestamp - user_stats["user_last_interaction"])
            .dt.total_seconds()
            .div(86400.0)
            .clip(lower=0.0)
        )

        # Genre affinity — mean rating per user per genre (train only)
        # Must run AFTER movie features so genre_* columns exist in train_df.
        genre_cols = sorted([c for c in train_df.columns if c.startswith("genre_")])
        if not genre_cols:
            raise RuntimeError(
                "No genre_* columns found in train_df. "
                "Call create_movie_features() before compute_user_stats()."
            )

        affinity_frames = [user_stats.set_index("userId")]
        for genre_col in genre_cols:
            affinity_col = genre_col.replace("genre_", "user_affinity_")
            genre_mask = train_df[genre_col] == 1
            if genre_mask.any():
                aff = (
                    train_df[genre_mask]
                    .groupby("userId")["rating"]
                    .mean()
                    .rename(affinity_col)
                )
                affinity_frames.append(aff)
            else:
                # Genre exists in the vocab but no training user rated it
                affinity_frames.append(
                    pd.Series(dtype=np.float32, name=affinity_col)
                )

        user_lookup = pd.concat(affinity_frames, axis=1)
        user_lookup.index.name = "userId"  # pd.concat can drop the index name

        # Fill genre affinity NaN with user's global average rating
        for genre_col in genre_cols:
            affinity_col = genre_col.replace("genre_", "user_affinity_")
            if affinity_col in user_lookup.columns:
                user_lookup[affinity_col] = user_lookup[affinity_col].fillna(
                    user_lookup["user_avg_rating"]
                )

        # Drop raw timestamp helpers — they are datetime dtype and cannot be
        # averaged for cold-start fallback in apply_user_stats.  Derived
        # numeric features (user_activity_days, user_days_since_last_rating)
        # are already computed and kept.
        user_lookup = user_lookup.drop(
            columns=["user_first_interaction", "user_last_interaction"], errors="ignore"
        )

        logger.info(f"User stats computed: {user_lookup.shape}")
        return user_lookup  # indexed by userId

    def apply_user_stats(
        self,
        df: pd.DataFrame,
        user_lookup: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge pre-computed user stats into ``df``.

        Users that appear in ``df`` but not in ``user_lookup`` (cold-start users
        in val/test) receive the global train mean for every stat column.

        Args:
            df: Split DataFrame.
            user_lookup: Output of :meth:`compute_user_stats` (indexed by userId).

        Returns:
            ``df`` augmented with user stat columns.
        """
        global_fallback = user_lookup.mean(numeric_only=True)
        df = df.merge(
            user_lookup.rename_axis("userId").reset_index(),
            on="userId",
            how="left",
        )
        # Fill cold-start users with training global means
        for col in user_lookup.columns:
            if col in df.columns:
                df[col] = df[col].fillna(global_fallback.get(col, 0.0))
        return df

    # ------------------------------------------------------------------
    # Movie popularity features — MUST be computed from train only
    # ------------------------------------------------------------------

    def compute_movie_popularity_stats(self, train_df: pd.DataFrame) -> pd.DataFrame:
        """Compute per-movie popularity stats from the TRAINING split only.

        Args:
            train_df: Training interactions.

        Returns:
            DataFrame indexed by ``movieId`` with movie popularity columns.
        """
        logger.info("Computing movie popularity stats from training split...")

        movie_stats = train_df.groupby("movieId").agg({
            "rating": ["mean", "std", "count"],
            "userId": "nunique"
        }).reset_index()

        movie_stats.columns = [
            "movieId",
            "movie_avg_rating",
            "movie_rating_std",
            "movie_num_ratings",
            "movie_num_users",
        ]
        movie_stats["movie_popularity"] = (
            movie_stats["movie_avg_rating"]
            * np.log1p(movie_stats["movie_num_ratings"])
        )
        movie_stats["movie_rating_std"] = movie_stats["movie_rating_std"].fillna(0)

        logger.info(f"Movie popularity stats computed: {movie_stats.shape}")
        return movie_stats.set_index("movieId")

    def apply_movie_popularity_stats(
        self,
        df: pd.DataFrame,
        movie_lookup: pd.DataFrame,
    ) -> pd.DataFrame:
        """Merge pre-computed movie popularity stats into ``df``.

        Movies not in ``movie_lookup`` (long-tail items only in val/test) receive
        the global train mean for every stat column.

        Args:
            df: Split DataFrame.
            movie_lookup: Output of :meth:`compute_movie_popularity_stats`.

        Returns:
            ``df`` augmented with movie popularity columns.
        """
        global_fallback = movie_lookup.mean(numeric_only=True)
        df = df.merge(
            movie_lookup.rename_axis("movieId").reset_index(),
            on="movieId",
            how="left",
        )
        for col in movie_lookup.columns:
            if col in df.columns:
                df[col] = df[col].fillna(global_fallback.get(col, 0.0))
        return df

    # ------------------------------------------------------------------
    # Interaction features (derived — safe once stats are correct)
    # ------------------------------------------------------------------

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create user-item interaction features.

        Args:
            df: Input DataFrame with user-item interactions.

        Returns:
            DataFrame with interaction features added.
        """
        logger.info("Creating interaction features...")

        # Rating deviation from user's mean
        df["rating_deviation_user"] = df["rating"] - df["user_avg_rating"]

        # Rating deviation from movie's mean
        df["rating_deviation_movie"] = df["rating"] - df["movie_avg_rating"]

        # Is this above/below average for user?
        df["above_user_avg"] = (df["rating"] > df["user_avg_rating"]).astype(int)

        # Is this above/below average for movie?
        df["above_movie_avg"] = (df["rating"] > df["movie_avg_rating"]).astype(int)

        # Timestamp features
        df["interaction_year"] = df["timestamp"].dt.year
        df["interaction_month"] = df["timestamp"].dt.month
        df["interaction_day_of_week"] = df["timestamp"].dt.dayofweek
        df["interaction_hour"] = df["timestamp"].dt.hour

        # Time since user's first interaction — only available when
        # user_first_interaction is present (legacy engineer_features path).
        if "user_first_interaction" in df.columns:
            df["days_since_first_interaction"] = (
                (df["timestamp"] - df["user_first_interaction"]).dt.total_seconds() / (24 * 3600)
            )

        # ── Cross-features ─────────────────────────────────────────────────
        # User × item interactions that capture alignment between user taste
        # and item quality / popularity in a single multiplicative signal.
        if "user_avg_rating" in df.columns and "movie_avg_rating" in df.columns:
            df["user_avg_x_movie_avg"] = (
                df["user_avg_rating"] * df["movie_avg_rating"]
            ).astype(np.float32)

        if "user_num_ratings" in df.columns and "movie_popularity" in df.columns:
            df["user_activity_x_popularity"] = (
                df["user_num_ratings"] * df["movie_popularity"]
            ).astype(np.float32)

        logger.info("Created interaction features")
        return df

    # ------------------------------------------------------------------
    # Top-level pipelines
    # ------------------------------------------------------------------

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run full feature engineering pipeline on a single DataFrame.

        .. warning::
            When ``df`` contains multiple temporal splits (train + val + test
            combined), user/movie statistics are computed on ALL rows, leaking
            future interactions into training features.  Use
            :meth:`engineer_features_no_leakage` instead when you have separate
            split DataFrames.

        Args:
            df: Preprocessed DataFrame (all splits combined).

        Returns:
            DataFrame with all engineered features.
        """
        logger.info("Starting feature engineering (single-dataframe mode)...")
        df = self.create_movie_features(df)
        # User stats computed on all rows (legacy behaviour)
        df = self._create_user_features_inplace(df)
        df = self._create_movie_popularity_inplace(df)
        df = self.create_interaction_features(df)
        logger.info(f"Feature engineering completed: {df.shape[1]} total columns")
        return df

    def engineer_features_no_leakage(
        self,
        train_df: pd.DataFrame,
        val_df: pd.DataFrame,
        test_df: pd.DataFrame,
        ref_timestamp: Optional[pd.Timestamp] = None,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Run feature engineering with strict train-only stat computation.

        Correct temporal split order:
          1. Movie metadata features (static, no leakage) → applied to all splits.
          2. User statistics  computed from ``train_df`` only → joined to all splits.
          3. Movie popularity computed from ``train_df`` only → joined to all splits.
          4. Interaction features derived from the above → computed per-split.

        Cold-start users/items in val/test receive the global training mean for
        stat columns.

        Args:
            train_df: Training split.
            val_df:   Validation split.
            test_df:  Test split.
            ref_timestamp: Reference timestamp for user recency calculation.
                Defaults to ``train_df["timestamp"].max()``.

        Returns:
            (train_feats, val_feats, test_feats) — three DataFrames with all features.
        """
        logger.info("Starting feature engineering (no-leakage mode)...")

        # ── Step 1: Static movie metadata features ──────────────────────
        # create_movie_features selects top genres from the provided data.
        # We compute it on train so genre vocabulary is train-sized
        train_df = self.create_movie_features(train_df)
        val_df   = self.apply_movie_features(val_df)
        test_df  = self.apply_movie_features(test_df)

        # ── Step 2: User stats from train only ──────────────────────────
        user_lookup = self.compute_user_stats(train_df, ref_timestamp=ref_timestamp)
        train_df = self.apply_user_stats(train_df, user_lookup)
        val_df   = self.apply_user_stats(val_df,   user_lookup)
        test_df  = self.apply_user_stats(test_df,  user_lookup)

        # ── Step 3: Movie popularity from train only ─────────────────────
        movie_lookup = self.compute_movie_popularity_stats(train_df)
        train_df = self.apply_movie_popularity_stats(train_df, movie_lookup)
        val_df   = self.apply_movie_popularity_stats(val_df,   movie_lookup)
        test_df  = self.apply_movie_popularity_stats(test_df,  movie_lookup)

        # ── Step 4: Interaction features (derived, per-split) ────────────
        train_df = self.create_interaction_features(train_df)
        val_df   = self.create_interaction_features(val_df)
        test_df  = self.create_interaction_features(test_df)

        logger.info(
            f"No-leakage feature engineering done. "
            f"train={train_df.shape}, val={val_df.shape}, test={test_df.shape}"
        )
        return train_df, val_df, test_df

    # ------------------------------------------------------------------
    # Private helpers (in-place variants used by legacy engineer_features)
    # ------------------------------------------------------------------

    def _create_user_features_inplace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legacy: compute user features from ALL rows in df (leaky on full data)."""
        user_lookup = self.compute_user_stats(df, ref_timestamp=df["timestamp"].max())
        return self.apply_user_stats(df, user_lookup)

    def _create_movie_popularity_inplace(self, df: pd.DataFrame) -> pd.DataFrame:
        """Legacy: compute movie popularity from ALL rows in df (leaky on full data)."""
        movie_lookup = self.compute_movie_popularity_stats(df)
        return self.apply_movie_popularity_stats(df, movie_lookup)

    # kept for direct callers that imported these methods by name
    def create_user_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deprecated: use compute_user_stats + apply_user_stats instead."""
        return self._create_user_features_inplace(df)

    def create_movie_popularity_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Deprecated: use compute_movie_popularity_stats + apply_movie_popularity_stats."""
        return self._create_movie_popularity_inplace(df)
