ALTER TABLE market_summary RENAME TO market_wrap;
ALTER TABLE market_summary_send_log RENAME TO market_wrap_send_log;
ALTER TABLE market_wrap_send_log RENAME COLUMN market_summary_id TO market_wrap_id;

DROP INDEX IF EXISTS market_summary_date_idx;
DROP INDEX IF EXISTS market_summary_slug_idx;
DROP INDEX IF EXISTS market_summary_send_log_summary_idx;

CREATE UNIQUE INDEX IF NOT EXISTS market_wrap_date_idx ON market_wrap (date);
CREATE UNIQUE INDEX IF NOT EXISTS market_wrap_slug_idx ON market_wrap (slug);
CREATE UNIQUE INDEX IF NOT EXISTS market_wrap_send_log_summary_idx ON market_wrap_send_log (market_wrap_id);
