CREATE TABLE IF NOT EXISTS web_sessions (
    id UUID PRIMARY KEY,
    agent_id UUID NOT NULL,
    owner_type VARCHAR NOT NULL,
    owner_id VARCHAR NOT NULL,
    user_type_id INTEGER,
    last_question TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_web_sessions_owner
    ON web_sessions (owner_type, owner_id);

CREATE TABLE IF NOT EXISTS external_identities (
    id UUID PRIMARY KEY,
    identity_type VARCHAR NOT NULL,
    external_id VARCHAR NOT NULL,
    display_name TEXT,
    user_type_id INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_external_identities_unique
    ON external_identities (identity_type, external_id);

CREATE TABLE IF NOT EXISTS ai_config (
    key VARCHAR PRIMARY KEY,
    value TEXT NOT NULL,
    value_type VARCHAR NOT NULL,
    category VARCHAR NOT NULL,
    description TEXT,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS ai_config_user_type_overrides (
    id UUID PRIMARY KEY,
    ai_config_key VARCHAR NOT NULL,
    user_type_id INTEGER NOT NULL,
    value TEXT NOT NULL,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_ai_config_overrides_unique
    ON ai_config_user_type_overrides (ai_config_key, user_type_id);
