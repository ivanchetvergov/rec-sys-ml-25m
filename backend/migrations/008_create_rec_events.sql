-- Migration 008: recommendation and session KPI events

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '008_create_rec_events') THEN

        CREATE TABLE IF NOT EXISTS rec_events (
            id          SERIAL PRIMARY KEY,
            session_id  VARCHAR(128) NOT NULL,
            user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
            event_type  VARCHAR(64)  NOT NULL,
            block       VARCHAR(64),
            movie_id    INTEGER,
            created_at  TIMESTAMPTZ  NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS ix_rec_events_created_at ON rec_events (created_at DESC);
        CREATE INDEX IF NOT EXISTS ix_rec_events_event_type ON rec_events (event_type);
        CREATE INDEX IF NOT EXISTS ix_rec_events_session_id ON rec_events (session_id);

        INSERT INTO schema_migrations (version) VALUES ('008_create_rec_events');

    END IF;
END $$;
