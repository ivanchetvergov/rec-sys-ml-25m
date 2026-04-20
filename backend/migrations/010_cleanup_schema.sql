-- Migration 010: schema cleanup
--
-- 1. Drop daily_activity — table was created in 005 but is never written to
--    or queried. Admin dashboard computes metrics directly from source tables.
-- 2. Add NOT NULL to reviews.rating — the API enforces 1-5 but the column
--    lacked a DB-level constraint.
-- 3. Add index on rec_events(user_id) — FK column without an index.

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '010_cleanup_schema') THEN

        -- 1. Remove unused pre-aggregation table
        DROP TABLE IF EXISTS daily_activity;

        -- 2. Enforce NOT NULL on reviews.rating at DB level
        --    Backfill any legacy NULLs to 0 first (safety; should be none in practice)
        UPDATE reviews SET rating = 0 WHERE rating IS NULL;
        ALTER TABLE reviews ALTER COLUMN rating SET NOT NULL;

        -- 3. Index the FK column on rec_events
        CREATE INDEX IF NOT EXISTS ix_rec_events_user_id ON rec_events (user_id);

        INSERT INTO schema_migrations (version) VALUES ('010_cleanup_schema');

    END IF;
END $$;
