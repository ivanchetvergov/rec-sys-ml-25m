"""
extract_movies.py — extract unique movie catalogue from the feature store.

Reads train.parquet (25M interaction rows), deduplicates to unique movies,
merges IMDB/TMDB links, and writes a lightweight movies.parquet (~17K rows)
to data/processed/movies.parquet.

This file is the single source of truth for movie metadata used by the
backend at startup — it is ~300× smaller than the full feature store split.

Usage:
    python -m src.pipeline.extract_movies [--dataset-tag TAG]
"""
import argparse
import logging
from pathlib import Path

import pandas as pd

from src.config import DATA_PROCESSED, DATA_RAW, FEATURE_STORE_PATH

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# Columns to pull from the interaction parquet — all other 50+ columns ignored
_MOVIE_COLS = [
    "movieId",
    "title",
    "genres",
    "year",
    "movie_avg_rating",
    "movie_num_ratings",
    "movie_popularity",
]

OUTPUT_PATH = DATA_PROCESSED / "movies.parquet"
LINKS_CSV = DATA_RAW / "links.csv"


def extract_movies(dataset_tag: str) -> pd.DataFrame:
    parquet = FEATURE_STORE_PATH / dataset_tag / "train.parquet"
    if not parquet.exists():
        raise FileNotFoundError(f"Feature store not found: {parquet}")

    log.info(f"Reading {parquet} …")
    df = pd.read_parquet(parquet, columns=_MOVIE_COLS)
    log.info(f"  rows loaded : {len(df):,}")

    # One row per movie
    movies = (
        df.drop_duplicates("movieId")
        .sort_values("movie_popularity", ascending=False)
        .reset_index(drop=True)
    )
    log.info(f"  unique movies: {len(movies):,}")

    # Merge IMDB / TMDB links
    if LINKS_CSV.exists():
        links = pd.read_csv(LINKS_CSV, dtype={"imdbId": str})
        movies = movies.merge(
            links[["movieId", "imdbId", "tmdbId"]], on="movieId", how="left"
        )
        log.info("  merged links.csv — imdb_id / tmdb_id available")
    else:
        movies["imdbId"] = None
        movies["tmdbId"] = None
        log.warning(f"  links.csv not found at {LINKS_CSV}")

    return movies


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract movie catalogue parquet")
    parser.add_argument(
        "--dataset-tag",
        default="ml_v_20260215_184134",
        help="Feature-store dataset tag (subfolder name)",
    )
    args = parser.parse_args()

    movies = extract_movies(args.dataset_tag)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    movies.to_parquet(OUTPUT_PATH, index=False)
    log.info(f"Saved {len(movies):,} movies → {OUTPUT_PATH}")
    log.info(f"File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")


if __name__ == "__main__":
    main()
