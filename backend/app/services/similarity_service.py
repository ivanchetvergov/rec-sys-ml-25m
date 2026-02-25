"""
SimilarityService — serves pre-computed item-item similarity from disk.

Reads data/processed/similarity_index.parquet produced by
`make build-similarity` and answers O(1) queries for similar movies.

No ML dependencies required at runtime — just pandas + pyarrow.
"""
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT") or Path(__file__).resolve().parents[3])
_INDEX_PATH = _PROJECT_ROOT / "data" / "processed" / "similarity_index.parquet"


class SimilarityService:
    def __init__(self, index_path: Optional[Path] = None) -> None:
        self._path = index_path or _INDEX_PATH
        self._index: Optional[dict[int, list[int]]] = None
        self.available: bool = False

    def _load(self) -> None:
        if not self._path.exists():
            logger.warning(
                "similarity_index.parquet not found at %s. "
                "Run `make build-similarity` to generate it. "
                "Similar-movies endpoint will return empty results.",
                self._path,
            )
            self._index = {}
            return

        logger.info("Loading similarity index from %s …", self._path)
        df = pd.read_parquet(self._path, columns=["movieId", "similar_ids"])
        self._index = {
            int(row.movieId): [int(x) for x in row.similar_ids]
            for row in df.itertuples(index=False)
        }
        self.available = True
        logger.info(
            "Similarity index ready — %d movies indexed (%d KB)",
            len(self._index),
            self._path.stat().st_size // 1024,
        )

    def _ensure_loaded(self) -> None:
        if self._index is None:
            self._load()

    def get_similar_ids(self, movie_id: int, n: int = 20) -> list[int]:
        """Return up to `n` similar movie IDs for `movie_id`."""
        self._ensure_loaded()
        return (self._index or {}).get(movie_id, [])[:n]


@lru_cache(maxsize=1)
def get_similarity_service() -> SimilarityService:
    return SimilarityService()
