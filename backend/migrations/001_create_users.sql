-- Migration 001: create users table
-- Idempotent: safe to run multiple times

CREATE TABLE IF NOT EXISTS schema_migrations (
    version     VARCHAR(64) PRIMARY KEY,
    applied_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

DO $$ BEGIN
    IF NOT EXISTS (SELECT 1 FROM schema_migrations WHERE version = '001_create_users') THEN

        -- Enum type
        IF NOT EXISTS (SELECT 1 FROM pg_type WHERE typname = 'user_role') THEN
            CREATE TYPE user_role AS ENUM ('user', 'admin');
        END IF;

        -- Users table
        CREATE TABLE IF NOT EXISTS users (
            id            SERIAL PRIMARY KEY,
            login         VARCHAR(64)  NOT NULL UNIQUE,
            email         VARCHAR(255) NOT NULL UNIQUE,
            password_hash VARCHAR(255) NOT NULL,
            role          user_role    NOT NULL DEFAULT 'user',
            created_at    TIMESTAMPTZ  NOT NULL DEFAULT now()
        );

        CREATE INDEX IF NOT EXISTS ix_users_login ON users (login);
        CREATE INDEX IF NOT EXISTS ix_users_email ON users (email);

        INSERT INTO schema_migrations (version) VALUES ('001_create_users');

    END IF;
END $$;
