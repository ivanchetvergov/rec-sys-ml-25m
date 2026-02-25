"""Recommender models."""

from src.models.popularity_based import PopularityRecommender
from src.models.als_recommender import ExplicitALSRecommender, SVDRecommender
from src.models.ials_recommender import ImplicitALSRecommender, ALSRecommender
from src.models.catboost_ranker import CatBoostRanker
from src.models.two_stage_recommender import TwoStageRecommender

__all__ = [
    "PopularityRecommender",
    # Explicit ALS (observed ratings only, SVD-solved)
    "ExplicitALSRecommender",
    "SVDRecommender",          # backward-compat alias
    # Implicit ALS (full matrix, confidence-weighted)
    "ImplicitALSRecommender",
    "ALSRecommender",          # backward-compat alias
    "CatBoostRanker",
    "TwoStageRecommender",
]
