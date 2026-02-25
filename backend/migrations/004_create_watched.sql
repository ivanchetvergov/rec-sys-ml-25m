-- Migration 004: create watched table

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '004_create_watched') THEN

        CREATE TABLE IF NOT EXISTS watched (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            movie_id    INTEGER      NOT NULL,
            title       VARCHAR(512) NOT NULL,
            genres      VARCHAR(512),
            year        SMALLINT,
            avg_rating  REAL,
            num_ratings INTEGER,
            popularity_score REAL,
            tmdb_id     INTEGER,
            imdb_id     VARCHAR(16),
            watched_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (user_id, movie_id)
        );

        CREATE INDEX IF NOT EXISTS ix_watched_user_id  ON watched (user_id);
        CREATE INDEX IF NOT EXISTS ix_watched_movie_id ON watched (movie_id);

        INSERT INTO schema_migrations (version) VALUES ('004_create_watched');

    END IF;
END $$;
