-- Accounts system. `email` is the only PII column stored anywhere in this
-- feature by design — do not add name/avatar/IP columns without revisiting
-- that decision (see CLAUDE.md / plan discussion on PII minimization).

CREATE TABLE IF NOT EXISTS user (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  email TEXT NOT NULL,
  password_hash TEXT,
  password_salt TEXT,
  email_verified_at TEXT,
  google_sub TEXT,
  created_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS user_email_idx ON user (email);
CREATE UNIQUE INDEX IF NOT EXISTS user_google_sub_idx ON user (google_sub);

CREATE TABLE IF NOT EXISTS session (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES user (id) ON DELETE CASCADE,
  token_hash TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (current_timestamp),
  last_seen_at TEXT NOT NULL DEFAULT (current_timestamp),
  expires_at TEXT NOT NULL,
  revoked_at TEXT
);
CREATE UNIQUE INDEX IF NOT EXISTS session_token_hash_idx ON session (token_hash);
CREATE INDEX IF NOT EXISTS session_user_idx ON session (user_id);

CREATE TABLE IF NOT EXISTS auth_token (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES user (id) ON DELETE CASCADE,
  purpose TEXT NOT NULL CHECK (purpose IN ('email_verify', 'password_reset')),
  token_hash TEXT NOT NULL,
  expires_at TEXT NOT NULL,
  consumed_at TEXT,
  created_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS auth_token_hash_idx ON auth_token (token_hash);
CREATE INDEX IF NOT EXISTS auth_token_user_purpose_idx ON auth_token (user_id, purpose);

CREATE TABLE IF NOT EXISTS watchlist_item (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES user (id) ON DELETE CASCADE,
  symbol TEXT NOT NULL,
  company_name TEXT NOT NULL,
  added_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS watchlist_item_user_symbol_idx ON watchlist_item (user_id, symbol);

CREATE TABLE IF NOT EXISTS search_history_item (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id INTEGER NOT NULL REFERENCES user (id) ON DELETE CASCADE,
  query TEXT NOT NULL,
  searched_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE INDEX IF NOT EXISTS search_history_item_user_date_idx ON search_history_item (user_id, searched_at);
