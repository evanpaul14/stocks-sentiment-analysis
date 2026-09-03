CREATE TABLE IF NOT EXISTS article_image (
  key TEXT PRIMARY KEY,
  unsplash_id TEXT,
  url TEXT NOT NULL,
  thumbnail_url TEXT,
  download_location TEXT,
  attribution_json TEXT,
  created_at TEXT NOT NULL DEFAULT (current_timestamp)
);

CREATE TABLE IF NOT EXISTS market_summary (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  date TEXT NOT NULL,
  slug TEXT NOT NULL,
  title TEXT NOT NULL,
  body TEXT NOT NULL,
  index_snapshot_json TEXT NOT NULL,
  headlines_json TEXT NOT NULL,
  created_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS market_summary_date_idx ON market_summary (date);
CREATE UNIQUE INDEX IF NOT EXISTS market_summary_slug_idx ON market_summary (slug);

CREATE TABLE IF NOT EXISTS market_summary_send_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  market_summary_id INTEGER NOT NULL REFERENCES market_summary (id),
  status TEXT NOT NULL CHECK (status IN ('sent', 'failed')),
  error_message TEXT,
  sent_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS market_summary_send_log_summary_idx ON market_summary_send_log (market_summary_id);

CREATE TABLE IF NOT EXISTS sentiment_history (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ticker TEXT NOT NULL,
  article_title TEXT NOT NULL,
  article_link TEXT,
  article_source TEXT,
  article_published_at TEXT,
  sentiment TEXT NOT NULL CHECK (sentiment IN ('positive', 'negative', 'neutral')),
  analyzed_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS sentiment_history_ticker_link_idx ON sentiment_history (ticker, article_link);
CREATE INDEX IF NOT EXISTS sentiment_history_ticker_date_idx ON sentiment_history (ticker, analyzed_at);

CREATE TABLE IF NOT EXISTS sentiment_page_cache (
  slug TEXT PRIMARY KEY,
  ticker TEXT NOT NULL,
  sections_json TEXT NOT NULL,
  price_json TEXT,
  sentiment_json TEXT,
  generated_at TEXT NOT NULL DEFAULT (current_timestamp),
  expires_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS job_run_log (
  job_name TEXT NOT NULL,
  run_date TEXT NOT NULL,
  ran_at TEXT NOT NULL DEFAULT (current_timestamp)
);
CREATE UNIQUE INDEX IF NOT EXISTS job_run_log_job_date_idx ON job_run_log (job_name, run_date);
