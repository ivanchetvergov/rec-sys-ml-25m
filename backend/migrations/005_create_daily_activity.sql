-- Migration 005: daily activity snapshot table
-- Pre-aggregated per-day counters for fast admin dashboard queries.

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '005_create_daily_activity') THEN

        CREATE TABLE IF NOT EXISTS daily_activity (
            id              SERIAL PRIMARY KEY,
            date            DATE        NOT NULL UNIQUE,
            new_users       INTEGER     NOT NULL DEFAULT 0,
            new_watched     INTEGER     NOT NULL DEFAULT 0,
            new_reviews     INTEGER     NOT NULL DEFAULT 0,
            new_watchlist   INTEGER     NOT NULL DEFAULT 0,
            created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS ix_daily_activity_date ON daily_activity (date DESC);

        INSERT INTO schema_migrations (version) VALUES ('005_create_daily_activity');

    END IF;
END $$;
