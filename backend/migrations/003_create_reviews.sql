-- Migration 003: create reviews table

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '003_create_reviews') THEN

        CREATE TABLE IF NOT EXISTS reviews (
            id          SERIAL PRIMARY KEY,
            user_id     INTEGER      NOT NULL REFERENCES users(id) ON DELETE CASCADE,
            movie_id    INTEGER      NOT NULL,
            title       VARCHAR(512) NOT NULL,
            rating      SMALLINT     CHECK (rating BETWEEN 1 AND 5),
            review_text TEXT,
            created_at  TIMESTAMPTZ  NOT NULL DEFAULT now(),
            UNIQUE (user_id, movie_id)
        );

        CREATE INDEX IF NOT EXISTS ix_reviews_user_id  ON reviews (user_id);
        CREATE INDEX IF NOT EXISTS ix_reviews_movie_id ON reviews (movie_id);

        INSERT INTO schema_migrations (version) VALUES ('003_create_reviews');

    END IF;
END $$;
