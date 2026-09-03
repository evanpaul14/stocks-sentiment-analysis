# Deployment

This app is designed to run as a single long-lived Node process on a self-hosted VPS — no
serverless platform, no external database. SQLite lives on local disk; cron jobs run in-process.

## 1. Server prerequisites

- Node.js 20+ (the app was built and tested against Node 22/26; anything 20+ should work)
- A domain pointed at the server, if you want a real hostname
- `sqlite3` CLI installed (used by `deploy/backup-db.sh`, not required by the app itself)

## 2. Get the code onto the server

```bash
git clone <your-repo-url> /opt/stocks-sentiment-analysis-v2
cd /opt/stocks-sentiment-analysis-v2
npm install
cp .env.example .env   # then fill in real values — see below
```

Fill in `.env` with real API credentials. At minimum you need `GOOGLE_API_KEY` (Gemini backup
sentiment classifier) for the app to build/run cleanly; everything else degrades gracefully if
unset (see the comments in `.env.example`). `ADMIN_API_TOKEN` and `INDEXNOW_KEY` should be
random strings — generate with `openssl rand -hex 32`.

## 3. Build and do a one-time sanity check

```bash
npm run build
npm run start   # Ctrl+C once you've confirmed http://localhost:3000 works
```

The first boot runs pending DB migrations automatically and starts the cron scheduler
(daily market summary, weekly MAG7 sentiment backfill).

## 4. Run it as a service (systemd)

Copy `deploy/stocks-sentiment.service` to `/etc/systemd/system/stocks-sentiment.service`,
adjusting `WorkingDirectory`, `User`, and `Group` for your server. Then:

```bash
sudo useradd -r -s /usr/sbin/nologin stocksentiment   # if it doesn't exist yet
sudo chown -R stocksentiment:stocksentiment /opt/stocks-sentiment-analysis-v2
sudo systemctl daemon-reload
sudo systemctl enable --now stocks-sentiment
sudo systemctl status stocks-sentiment
journalctl -u stocks-sentiment -f   # tail logs
```

Log rotation for the systemd journal is handled by `journald` automatically on most distros
(see `journalctl --disk-usage` / `/etc/systemd/journald.conf` if you want to tune retention) —
no separate logrotate config is needed.

## 5. Put a reverse proxy in front of it

The app trusts `X-Forwarded-For` for per-IP rate limiting, so whatever proxy you use **must**
set that header. `deploy/Caddyfile` is the simplest option (automatic HTTPS via Let's Encrypt);
an equivalent nginx block is included as a comment in that file if you prefer nginx.

```bash
sudo cp deploy/Caddyfile /etc/caddy/Caddyfile   # edit the domain first
sudo systemctl reload caddy
```

404s are handled by the app itself (`app/not-found.tsx`). A 502 means the Node process is down
or unreachable — the app can't serve a page for that, so `deploy/Caddyfile` points Caddy at
`deploy/error-pages/502.html`, a static page served straight from disk (the nginx equivalent is
in the Caddyfile's comment block).

## 6. Database backups

`deploy/backup-db.sh` does an online SQLite backup (safe with WAL mode) and keeps the last 14
days, gzip-compressed, under `data/backups/`. Wire it into cron:

```bash
crontab -e
# add:
0 3 * * * /opt/stocks-sentiment-analysis-v2/deploy/backup-db.sh >> /var/log/stocks-sentiment-backup.log 2>&1
```

## 7. Redeploying (code or content updates)

```bash
cd /opt/stocks-sentiment-analysis-v2
git pull
npm install        # only if dependencies changed
npm run build       # runs pending DB migrations as a side effect of build-time DB access,
                     # and re-runs at boot too — safe either way
sudo systemctl restart stocks-sentiment
```

Publishing a new blog post is just adding a `.mdx` file under `content/blog/` and redeploying —
there's no admin UI or database write involved (see `content/blog/*.mdx` for the frontmatter
schema).

## What's intentionally NOT here

- No Docker/container setup — a bare Node process behind a reverse proxy was simpler for a
  single-instance SQLite-backed app and avoids an extra layer to debug.
- No CI/CD pipeline — add one if/when you want automated deploys; `npm run build` + `npm test`
  (once tests exist) is the relevant check to gate on.
- No horizontal scaling story — the in-memory rate limiter and in-process cron scheduler both
  assume a single Node process. If you ever need multiple instances, see the comment in
  `lib/ratelimit/tokenBucket.ts` for what has to change first.
