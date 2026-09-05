#!/usr/bin/env bash
# Backs up the SQLite database using SQLite's online .backup command (safe
# to run while the app is live, even in WAL mode), then mirrors it to Google
# Drive via rclone (remote configured in ~/.config/rclone/rclone.conf for
# whichever user runs this). Retains the last 14 daily backups locally and
# on Drive. Run via a systemd timer — see deploy/stocks-db-backup.timer.
set -euo pipefail
cd /

APP_DIR="/opt/stocks-sentiment-analysis-v2"
DB_PATH="${APP_DIR}/data/app.db"
BACKUP_DIR="${APP_DIR}/data/backups"
RETENTION_DAYS=14
RCLONE_REMOTE="drive:Backups/stocks-sentiment-db"

mkdir -p "$BACKUP_DIR"

timestamp=$(date +%Y-%m-%d_%H%M%S)
dest="${BACKUP_DIR}/app-${timestamp}.db"

sqlite3 "$DB_PATH" ".backup '${dest}'"
gzip "$dest"

find "$BACKUP_DIR" -name "app-*.db.gz" -mtime "+${RETENTION_DAYS}" -delete

rclone copy "${dest}.gz" "$RCLONE_REMOTE" --no-traverse
rclone delete "$RCLONE_REMOTE" --min-age "${RETENTION_DAYS}d" --no-traverse

echo "[backup] wrote ${dest}.gz and synced to ${RCLONE_REMOTE}"
