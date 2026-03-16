-- Migration 006: add user profile privacy flag

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '006_add_profile_privacy') THEN

        ALTER TABLE users
            ADD COLUMN IF NOT EXISTS is_profile_private BOOLEAN NOT NULL DEFAULT FALSE;

        INSERT INTO schema_migrations (version) VALUES ('006_add_profile_privacy');

    END IF;
END $$;
