"""
build_similarity_index.py — pre-compute item-item cosine similarity from ALS factors.

Reads the trained ALSRecommender item factors, computes top-N most similar
movies for every movie via batched cosine similarity, then writes a compact
Parquet file that the backend can load at startup.

The index is ~3 MB (17K movies × top-20 neighbours) and is independent of
the `implicit` / catboost stack — the backend only needs pandas + pyarrow.

Usage:
    python -m src.pipeline.build_similarity_index [--n-similar 20] [--batch-size 2000]

Output:
    data/processed/similarity_index.parquet
        movieId          int64
        similar_ids      object  (list of int)
        similarity_scores object (list of float, descending)
"""
import argparse
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

_ALS_DIR = PROJECT_ROOT / "data" / "models" / "two_stage_ranker" / "als"
_MOVIES_PARQUET = PROJECT_ROOT / "data" / "processed" / "movies.parquet"
OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "similarity_index.parquet"


# ---------------------------------------------------------------------------
# Genre-based fallback (Jaccard similarity on multi-hot genre vectors)
# ---------------------------------------------------------------------------
def _genre_similarity_index(movies: pd.DataFrame, n_similar: int) -> pd.DataFrame:
    """Build similarity index from genre overlap (Jaccard) as ALS fallback."""
    log.info("Building genre-based similarity index (ALS not available) …")
    genre_sets = movies.set_index("movieId")["genres"].fillna("").apply(
        lambda g: set(g.split("|")) if g else set()
    )
    movie_ids = genre_sets.index.tolist()
    n = len(movie_ids)
    rows = []
    for i, mid in enumerate(movie_ids):
        g_i = genre_sets[mid]
        if not g_i:
            rows.append({"movieId": mid, "similar_ids": [], "similarity_scores": []})
            continue
        scores = []
        for j, other in enumerate(movie_ids):
            if i == j:
                continue
            g_j = genre_sets[other]
            if not g_j:
                continue
            jaccard = len(g_i & g_j) / len(g_i | g_j)
            if jaccard > 0:
                scores.append((other, jaccard))
        scores.sort(key=lambda x: -x[1])
        top = scores[:n_similar]
        rows.append({
            "movieId": mid,
            "similar_ids": [x[0] for x in top],
            "similarity_scores": [round(x[1], 4) for x in top],
        })
        if i % 1000 == 0:
            log.info(f"  processed {i}/{n} …")
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# ALS-based (cosine similarity on item latent factors)
# ---------------------------------------------------------------------------
def _als_similarity_index(n_similar: int, batch_size: int) -> pd.DataFrame:
    """Build similarity index from ALS item vectors via batched cosine sim."""
    log.info(f"Loading ALS model from {_ALS_DIR} …")

    with open(_ALS_DIR / "implicit_model.pkl", "rb") as f:
        implicit_model = pickle.load(f)
    with open(_ALS_DIR / "id_maps.pkl", "rb") as f:
        id_maps = pickle.load(f)

    item_factors: np.ndarray = implicit_model.item_factors  # (n_items, factors)
    idx_to_item: dict = id_maps["idx_to_item"]

    n_items, n_factors = item_factors.shape
    log.info(f"  item factors: {n_items:,} items × {n_factors} factors")

    # L2-normalise for cosine via dot product
    norms = np.linalg.norm(item_factors, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normed = (item_factors / norms).astype(np.float32)

    rows = []
    for start in range(0, n_items, batch_size):
        end = min(start + batch_size, n_items)
        batch = normed[start:end]               # (B, factors)
        sims = batch @ normed.T                 # (B, n_items)
        # zero out self-similarity
        for bi in range(end - start):
            sims[bi, start + bi] = -1.0

        top_indices = np.argpartition(sims, -n_similar, axis=1)[:, -n_similar:]
        for bi in range(end - start):
            item_idx = start + bi
            ti = top_indices[bi]
            ti = ti[np.argsort(-sims[bi, ti])]     # sort descending by score
            rows.append({
                "movieId": idx_to_item[item_idx],
                "similar_ids": [idx_to_item[int(i)] for i in ti],
                "similarity_scores": [round(float(sims[bi, i]), 4) for i in ti],
            })

        log.info(f"  {min(end, n_items):,}/{n_items:,} items processed …")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Build item-item similarity index")
    parser.add_argument("--n-similar", type=int, default=20,
                        help="Number of similar items to store per movie")
    parser.add_argument("--batch-size", type=int, default=2000,
                        help="Batch size for matrix multiplication")
    args = parser.parse_args()

    if _ALS_DIR.exists() and (_ALS_DIR / "implicit_model.pkl").exists():
        df = _als_similarity_index(args.n_similar, args.batch_size)
        method = "ALS cosine"
    else:
        log.warning("ALS model not found — falling back to genre-based Jaccard similarity.")
        log.warning("Run `make train-ranker` first to get ALS item vectors.")
        if not _MOVIES_PARQUET.exists():
            raise FileNotFoundError(
                f"movies.parquet not found at {_MOVIES_PARQUET}. "
                "Run `make extract-movies` first."
            )
        movies = pd.read_parquet(_MOVIES_PARQUET, columns=["movieId", "genres"])
        df = _genre_similarity_index(movies, args.n_similar)
        method = "genre Jaccard"

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PATH, index=False)
    size_kb = OUTPUT_PATH.stat().st_size // 1024
    log.info(f"Saved {len(df):,} rows → {OUTPUT_PATH}  ({size_kb} KB)  [{method}]")


if __name__ == "__main__":
    main()
