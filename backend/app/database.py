"""Database migration runner — uses psycopg2 (sync, simple)."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

DATABASE_URL = os.environ.get(
    "DATABASE_URL",
    "postgresql://recsys:recsys_secret@localhost:5432/recsys",
)

# Strip asyncpg prefix if present
_SYNC_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://", 1)

MIGRATIONS_DIR = Path(__file__).resolve().parents[1] / "migrations"


def run_migrations() -> None:
    """Execute all .sql files in migrations/ in order. Idempotent."""
    import psycopg2
    from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

    logger.info("Connecting to postgres: %s", _SYNC_URL)
    try:
        conn = psycopg2.connect(_SYNC_URL, connect_timeout=10)
        conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
        cursor = conn.cursor()
    except Exception as exc:
        logger.error("Failed to connect to postgres: %s", exc)
        raise

    sql_files = sorted(MIGRATIONS_DIR.glob("*.sql"))
    if not sql_files:
        logger.warning("No .sql migration files found in %s", MIGRATIONS_DIR)
        cursor.close()
        conn.close()
        return

    for sql_file in sql_files:
        logger.info("Applying migration: %s", sql_file.name)
        try:
            cursor.execute(sql_file.read_text(encoding="utf-8"))
            logger.info("Done: %s", sql_file.name)
        except Exception as exc:
            logger.error("Migration %s failed: %s", sql_file.name, exc)
            cursor.close()
            conn.close()
            raise

    cursor.close()
    conn.close()
    logger.info("All migrations applied.")
