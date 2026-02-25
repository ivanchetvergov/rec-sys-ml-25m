"""Run Alembic migrations programmatically.

Usage:
    python migrate.py
"""

import asyncio
import os

from alembic import command
from alembic.config import Config


def run_migrations() -> None:
    alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))

    # Override DB URL from env if set
    db_url = os.environ.get("DATABASE_URL", "")
    if db_url:
        if db_url.startswith("postgresql://"):
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
        alembic_cfg.set_main_option("sqlalchemy.url", db_url)

    command.upgrade(alembic_cfg, "head")
    print("✓ Migrations applied successfully.")


if __name__ == "__main__":
    run_migrations()
