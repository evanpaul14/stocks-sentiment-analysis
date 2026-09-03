#!/usr/bin/env bash
# Backs up the SQLite database using SQLite's online .backup command (safe
# to run while the app is live, even in WAL mode). Retains the last 14
# daily backups. Intended to run via cron, e.g.:
#   0 3 * * * /opt/stocks-sentiment-analysis-v2/deploy/backup-db.sh
set -euo pipefail

APP_DIR="/opt/stocks-sentiment-analysis-v2"
DB_PATH="${APP_DIR}/data/app.db"
BACKUP_DIR="${APP_DIR}/data/backups"
RETENTION_DAYS=14

mkdir -p "$BACKUP_DIR"

timestamp=$(date +%Y-%m-%d_%H%M%S)
dest="${BACKUP_DIR}/app-${timestamp}.db"

sqlite3 "$DB_PATH" ".backup '${dest}'"
gzip "$dest"

find "$BACKUP_DIR" -name "app-*.db.gz" -mtime "+${RETENTION_DAYS}" -delete

echo "[backup] wrote ${dest}.gz"
