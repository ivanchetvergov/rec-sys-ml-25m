-- Migration 002: create watchlist table

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '002_create_watchlist') THEN

        CREATE TABLE IF NOT EXISTS watchlist (
            id               SERIAL PRIMARY KEY,
            user_id          INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            movie_id         INTEGER      NOT NULL,
            title            VARCHAR(512) NOT NULL,
            genres           VARCHAR(512),
            year             SMALLINT,
            avg_rating       NUMERIC(3,2),
            num_ratings      INTEGER,
            popularity_score NUMERIC(10,4),
            tmdb_id          INTEGER,
            imdb_id          VARCHAR(16),
            added_at         TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (user_id, movie_id)
        );

        CREATE INDEX IF NOT EXISTS ix_watchlist_user_id  ON watchlist (user_id);
        CREATE INDEX IF NOT EXISTS ix_watchlist_movie_id ON watchlist (movie_id);

        INSERT INTO schema_migrations (version) VALUES ('002_create_watchlist');

    END IF;
END $$;
