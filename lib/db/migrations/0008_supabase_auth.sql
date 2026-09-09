-- Identity/sessions move to Supabase Auth. Drop the local user/session/
-- auth_token tables and re-key watchlist_item/search_history_item to
-- Supabase's UUID user ids (text) instead of the old local integer ids.
-- SQLite can't alter a column's type in place, and the old integer ids
-- don't map to anything post-migration anyway, so these two tables are
-- dropped and recreated empty.

DROP TABLE IF EXISTS watchlist_item;
DROP TABLE IF EXISTS search_history_item;
DROP TABLE IF EXISTS auth_token;
DROP TABLE IF EXISTS session;
DROP TABLE IF EXISTS user;

CREATE TABLE IF NOT EXISTS watchlist_item (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  symbol TEXT NOT NULL,
  company_name TEXT NOT NULL,
  added_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS watchlist_item_user_symbol_idx ON watchlist_item (user_id, symbol);

CREATE TABLE IF NOT EXISTS search_history_item (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  user_id TEXT NOT NULL,
  query TEXT NOT NULL,
  searched_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE INDEX IF NOT EXISTS search_history_item_user_date_idx ON search_history_item (user_id, searched_at);
