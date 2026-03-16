-- Migration 007: harden profile privacy column presence
-- Covers cases where migration history was marked applied but column is missing.

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '007_ensure_profile_privacy_column') THEN

        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS is_profile_private BOOLEAN NOT NULL DEFAULT FALSE;

        INSERT INTO schema_migrations (version) VALUES ('007_ensure_profile_privacy_column');

    END IF;
END $$;
