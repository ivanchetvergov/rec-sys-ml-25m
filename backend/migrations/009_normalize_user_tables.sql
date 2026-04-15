-- Migration 009: normalize user interaction tables
-- Remove movie metadata columns from watchlist, watched, and reviews.
-- Movie data is the sole responsibility of movies.parquet / PopularityService.
-- Interaction tables store only what is user-specific: user_id, movie_id,
-- timestamps, ratings and review text.

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '009_normalize_user_tables') THEN

        -- watchlist: keep only (id, user_id, movie_id, added_at)
        ALTER TABLE watchlist
            DROP COLUMN IF EXISTS title,
            DROP COLUMN IF EXISTS genres,
            DROP COLUMN IF EXISTS year,
            DROP COLUMN IF EXISTS avg_rating,
            DROP COLUMN IF EXISTS num_ratings,
            DROP COLUMN IF EXISTS popularity_score,
            DROP COLUMN IF EXISTS tmdb_id,
            DROP COLUMN IF EXISTS imdb_id;

        -- watched: keep only (id, user_id, movie_id, watched_at)
        ALTER TABLE watched
            DROP COLUMN IF EXISTS title,
            DROP COLUMN IF EXISTS genres,
            DROP COLUMN IF EXISTS year,
            DROP COLUMN IF EXISTS avg_rating,
            DROP COLUMN IF EXISTS num_ratings,
            DROP COLUMN IF EXISTS popularity_score,
            DROP COLUMN IF EXISTS tmdb_id,
            DROP COLUMN IF EXISTS imdb_id;

        -- reviews: keep only (id, user_id, movie_id, rating, review_text, created_at)
        ALTER TABLE reviews
            DROP COLUMN IF EXISTS title;

        INSERT INTO schema_migrations (version) VALUES ('009_normalize_user_tables');

    END IF;
END $$;
