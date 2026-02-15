"""Configuration for MovieLens data preprocessing pipeline."""
from pathlib import Path
from typing import Dict, List

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw" / "ml-25m"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
FEATURE_STORE_PATH = DATA_PROCESSED / "feature_store"

# Ensure directories exist
FEATURE_STORE_PATH.mkdir(parents=True, exist_ok=True)

# MovieLens 25M Dataset files
DATASETS = {
    "ratings": "ratings.csv",
    "movies": "movies.csv",
    "tags": "tags.csv",
    "links": "links.csv",
    "genome_scores": "genome-scores.csv",
    "genome_tags": "genome-tags.csv",
}

# Columns for each dataset
RATINGS_COLS = ["userId", "movieId", "rating", "timestamp"]
MOVIES_COLS = ["movieId", "title", "genres"]
TAGS_COLS = ["userId", "movieId", "tag", "timestamp"]
LINKS_COLS = ["movieId", "imdbId", "tmdbId"]

# Filter criteria
MIN_USER_RATINGS = 20  # Minimum ratings per user (active users)
MIN_MOVIE_RATINGS = 10  # Minimum ratings per movie (popular movies)
MIN_RATING = 0.5  # Minimum valid rating
MAX_RATING = 5.0  # Maximum valid rating

# Feature engineering parameters
TOP_GENRES = 20  # Number of top genres to create binary features for
MIN_YEAR = 1970  # Extract year from title and filter
MAX_YEAR = 2030  # Maximum reasonable year

# Temporal split (critical for recommender systems!)
# Split by timestamp percentile, not by date
TRAIN_SPLIT = 0.7  # 70% oldest interactions for training
VAL_SPLIT = 0.15  # 15% for validation
TEST_SPLIT = 0.15  # 15% newest interactions for testing

# User-based split (prevent leakage)
USER_SPLIT_SEED = 42  # For reproducible user splits
VALIDATION_FRACTION = 0.15  # Fraction of train users for validation
