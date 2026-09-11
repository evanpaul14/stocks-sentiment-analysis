@AGENTS.md

# Project notes for AI agents

Ground-up Next.js rebuild of the old Flask `stocks-sentiment-analysis` app. No code was ported —
only the feature set and external API shapes were carried over. See `README.md` for stack/scripts.
Runs as a systemd service on a self-hosted VPS behind Caddy, no serverless; `.github/workflows/deploy.yml`
builds and deploys automatically on every push to `master`.

## Commands

```bash
npm run dev         # dev server (Turbopack)
npm run build        # production build (runs prebuild: writes IndexNow key file)
npm run start         # run the production build
npm run lint           # ESLint
npm run db:migrate      # apply pending SQLite migrations (also runs automatically on first DB open)
```

There is no test suite (the original Flask app had none either). There's no single-test command
because there are no tests to target.

## Key architectural decisions (don't relitigate these without cause)

- **SQLite via better-sqlite3 + Drizzle**, not Postgres — single-VPS deployment, no external DB.
  The client (`lib/db/client.ts`) self-migrates on first connection open (see the comment there
  for why — it fixes a real build-time race that used to break `next build`).
- **In-memory rate limiting and caching** (`lib/ratelimit`, `lib/cache`) — acceptable because this
  runs as one Node process, not horizontally scaled. If that ever changes, these need a shared
  store (Redis) first.
- **Accounts are Supabase Auth** (email/password + Google sign-in; `lib/supabase/*`,
  `lib/auth/currentUser.ts`, `app/login`, `app/signup`, `app/account`). Identity lives entirely
  in Supabase — this app's own SQLite `user`/`session`/`auth_token` tables were dropped in
  migration `0008_supabase_auth.sql` in favor of it. Watchlist and search history are DB-backed
  (`watchlist_item`/`search_history_item`, keyed by the Supabase UUID) for signed-in users, and
  still `localStorage`-only for anonymous visitors; `POST /api/account/merge-local-data` does a
  one-time, idempotent merge of local data into the account on login. Account deletion
  (`DELETE /api/account`) requires `SUPABASE_SERVICE_ROLE_KEY` — the regular client SDK can't
  delete Supabase users, only the admin client can. Separately, the one privileged *admin*
  action (`POST /api/market-summary/generate`) is still just a single static `ADMIN_API_TOKEN`
  bearer token, unrelated to user accounts — don't conflate the two auth mechanisms.
- **Blog is MDX files in `content/blog/`, not a CMS.** Publishing = adding a file + redeploying.
- **Pages that read SQLite or call live external APIs use `export const dynamic =
  "force-dynamic"`**, not ISR (`revalidate`). Static prerendering these at build time caused a
  real multi-process SQLite lock race — see the git history / `lib/db/migrate.ts` comments if
  you're tempted to add `revalidate` back to a DB-backed page.
- **`/blog/[slug]` handles two different things**: real MDX posts AND programmatic SEO sentiment
  pages (`/blog/sentiment-of-<company>-stock`). They're consolidated into one route on purpose —
  Next.js's App Router doesn't reliably disambiguate two competing dynamic segments
  (`[slug]` vs `sentiment-of-[company]-stock`) at the same directory level; it silently picked
  the wrong one when they were separate routes. Don't split them back out without solving that.

## Extending the app

New data source → `lib/integrations/<source>/*.ts` (typed functions, no Next.js imports there) →
a cache wrapper in `lib/cache` if it needs one → one thin `app/api/.../route.ts` that composes
integration + cache + `withRateLimit` → a page that calls it server-side. Keep route handlers
compose-only.

## VPS access

The production server is reachable via `ssh digitalocean` (host alias already configured).
App lives at `/opt/stocks-sentiment-analysis-v2`, owned by `flaskuser`. The live systemd service
is named `stocks-nextjs` (not `stocks-sentiment`, despite `deploy/stocks-sentiment.service`'s
example name) — use `systemctl restart stocks-nextjs` after editing `.env` on the server, and
`journalctl -u stocks-nextjs -f` to tail logs. There's also a leftover failed `stocks.service`
(the old Flask app) — ignore it.

## Git commit messages

Do not mention Claude, Anthropic, or AI generation in commit message subjects or bodies.
