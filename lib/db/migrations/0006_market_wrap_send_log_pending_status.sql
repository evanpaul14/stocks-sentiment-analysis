-- SQLite can't ALTER a CHECK constraint in place, so rebuild the table to
-- allow a 'pending' status — used as an atomic in-flight marker to close the
-- check-then-send-then-record race that let the same market wrap email go
-- out twice (see lib/db/queries/marketWrapSendLog.ts).
CREATE TABLE IF NOT EXISTS market_wrap_send_log_new (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  market_wrap_id INTEGER NOT NULL REFERENCES market_wrap (id),
  status TEXT NOT NULL CHECK (status IN ('sent', 'failed', 'pending')),
  error_message TEXT,
  sent_at TEXT NOT NULL DEFAULT (current_timestamp)
);

INSERT INTO market_wrap_send_log_new (id, market_wrap_id, status, error_message, sent_at)
SELECT id, market_wrap_id, status, error_message, sent_at FROM market_wrap_send_log;

DROP TABLE market_wrap_send_log;
ALTER TABLE market_wrap_send_log_new RENAME TO market_wrap_send_log;

CREATE UNIQUE INDEX IF NOT EXISTS market_wrap_send_log_summary_idx ON market_wrap_send_log (market_wrap_id);
