"""Recommender models."""

from src.models.popularity_based import PopularityRecommender
from src.models.collaborative_filtering import SVDRecommender

__all__ = ["PopularityRecommender", "SVDRecommender"]
