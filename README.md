[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Stock Sentiment

Real-time stock prices, historical charts, AI-powered news sentiment analysis, trending
dashboards, an automated daily market summary, and a blog — a single Next.js app (App Router,
TypeScript), self-hosted on a VPS with SQLite for storage.

This is a ground-up rewrite of the original Flask app of the same name. Only the feature set and
external API shapes were carried over; none of the original code was reused.

## Features

- Search stocks by ticker or company name, with live price, historical chart (Day/5D/1M/YTD/1Y/5Y
  ranges), and key stats
- AI-powered news sentiment analysis per article (Gemini, with a Cloudflare Workers AI fallback)
- Trending dashboards sourced from StockTwits and Reddit (via ApeWisdom)
- Automated daily market summary (LLM-written wrap-up) with email subscription via Mailgun
- AI-generated stock movement insight (why a stock moved, in plain language)
- Sentiment-vs-price overlay chart per stock (90-day rolling view)
- Programmatic SEO pages for the Magnificent 7 + Netflix (`/blog/sentiment-of-<company>-stock`)
- Accounts (Supabase Auth — email/password + Google sign-in); watchlist and search history are
  DB-backed for signed-in users and fall back to `localStorage` for anonymous visitors, merged
  into the account on login
- Blog (MDX files, no CMS)
- Rate-limited API routes, in-memory (single-process deployment)

## Stack

- **Next.js 16** (App Router, TypeScript, Turbopack)
- **SQLite** via `better-sqlite3` + a typed Drizzle query layer — no external database
- **Tailwind CSS + shadcn/ui** (`base-nova` style)
- **Recharts** for charts
- **node-cron** for the daily market-summary and weekly sentiment-backfill jobs, running
  in-process (see `instrumentation.ts` / `lib/cron/`)

## Quickstart

```bash
git clone https://github.com/evanpaul14/stocks-sentiment-analysis.git
cd stocks-sentiment-analysis
npm install
cp .env.example .env   # fill in real API keys — see comments in the file
npm run dev
```

Open http://localhost:3000.

## API Endpoints

### Pages (HTML)

| Path | Description |
| --- | --- |
| `/` | Home — stock search, popular stocks, trending preview. |
| `/stock/<symbol>` | Stock detail: price chart, stats, news sentiment, StockTwits, sentiment/price overlay. |
| `/watchlist` | Watchlist view (DB-backed when signed in, `localStorage`-backed otherwise). |
| `/trending` | Trending dashboards landing page. |
| `/trending/<source>` | Trending dashboard for a specific source (`stocktwits`, `reddit`). |
| `/market-summary` | Market summary landing page. |
| `/market-summary/stock-market-today` | Always shows the latest market summary. |
| `/market-summary/<slug>` | A specific market summary article. |
| `/blog` | Blog listing. |
| `/blog/<slug>` | Blog post, or a programmatic SEO sentiment page for a covered company. |
| `/login` | Sign in (email/password or Google). |
| `/signup` | Create an account. |
| `/reset-password` | Password reset flow. |
| `/account` | Signed-in account management (delete account, etc.). |
| `/privacy` | Privacy policy page. |

### JSON APIs

| Method | Path | Auth | Description |
| --- | --- | --- | --- |
| POST | `/api/search` | No | Search for a stock; returns stock info, historical data, and news articles. |
| POST | `/api/sentiment` | No | Analyze sentiment for one news article. |
| POST | `/api/movement-insight` | No | Generate a plain-language explanation for a stock's price movement. |
| GET | `/api/historical/<symbol>/<period>` | No | Historical price series for a symbol and period. |
| GET | `/api/quote/<symbol>` | No | Quick price/quote lookup. |
| GET | `/api/sentiment-history/<symbol>` | No | Rolling sentiment-vs-price history for the overlay chart. |
| GET | `/api/trending` | No | Trending stocks, default source. |
| GET | `/api/trending/<source>` | No | Trending stocks from a specific source. |
| POST | `/api/trending/prices` | No | Batched price refresh for a list of symbols (watchlist/trending). |
| GET | `/api/stocktwits/<symbol>/summary` | No | StockTwits sentiment summary for a symbol. |
| GET | `/api/stocktwits/<symbol>/sentiment` | No | StockTwits sentiment detail for a symbol. |
| GET | `/api/market-summary/latest` | No | Latest market summary payload. |
| GET | `/api/market-summary/week-glance` | No | Weekly index snapshots for the market summary dashboard. |
| GET | `/api/market-summary/archive` | No | Market summary archive payload. |
| GET | `/api/market-summary/<slug>` | No | A specific market summary by slug. |
| POST | `/api/market-summary/subscribe` | No | Subscribe to market summary email updates (Mailgun). |
| POST | `/api/market-summary/generate` | **Yes** — `ADMIN_API_TOKEN` bearer token | Force-regenerate the market summary. |
| GET/POST | `/api/watchlist` | **Yes** — Supabase session | List / add a watchlist entry for the signed-in user. |
| DELETE | `/api/watchlist/<symbol>` | **Yes** — Supabase session | Remove a watchlist entry. |
| GET/POST | `/api/search-history` | **Yes** — Supabase session | List / record search history for the signed-in user. |
| POST | `/api/account/merge-local-data` | **Yes** — Supabase session | One-time merge of a device's `localStorage` watchlist/search history into the account. |
| DELETE | `/api/account` | **Yes** — Supabase session | Delete the caller's account and their local watchlist/search-history rows. |

### Static / SEO

| Path | Description |
| --- | --- |
| `/robots.txt` | Robots file. |
| `/sitemap.xml` | Sitemap. |
| `/manifest.webmanifest` | Web app manifest. |

## Architecture

- **SQLite via better-sqlite3 + Drizzle, not Postgres** — single-VPS deployment, no external DB.
  Self-migrates on first connection open.
- **In-memory rate limiting and caching** — fine for a single Node process; would need a shared
  store (Redis) to scale horizontally.
- **Accounts via Supabase Auth** (email/password + Google sign-in) — identity lives entirely in
  Supabase, not this app's own SQLite tables. Watchlist and search history are DB-backed
  (keyed by the Supabase user id) for signed-in users, still `localStorage`-only for anonymous
  visitors, and merged into the account once on login. Account deletion uses the Supabase
  service-role key (the regular client SDK can't delete users). Separately, the one privileged
  *admin* action (`POST /api/market-summary/generate`) is protected by a single static bearer
  token — unrelated to user accounts.
- **Blog is MDX files in `content/blog/`, not a CMS.** Publishing = adding a file + redeploying.
- Pages that read SQLite or call live external APIs render dynamically (`force-dynamic`), not via
  ISR — static prerendering these at build time caused a real multi-process SQLite lock race.

**Adding a new data source**: put typed functions in `lib/integrations/<source>/*.ts` (no
Next.js imports there), add a cache wrapper in `lib/cache` if it needs one, expose it via one
thin `app/api/.../route.ts`, and call it from a page. Route handlers stay compose-only — that's
the pattern the whole app follows.

## Project structure

```
app/                    Pages and API routes (App Router)
lib/
  db/                    SQLite client, schema (Drizzle), migrations, per-table query modules
  integrations/          One module per external API (Yahoo, StockTwits, Mailgun, etc.)
  cron/                  node-cron scheduler + the two scheduled jobs
  cache/                 In-memory TTL cache for hot short-lived data
  ratelimit/              In-memory per-IP token bucket
  auth/                  Supabase session helpers + the admin bearer-token check
  supabase/              Supabase client factories (browser, server, admin/service-role)
  seo/                   Structured data (JSON-LD), CSP nonce helper, sitemap key file writer
content/blog/*.mdx       Blog posts — see "Publishing a blog post" below
components/              UI, organized by feature area
```

## Publishing a blog post

There's no admin UI — posts are `.mdx` files with frontmatter:

```mdx
---
title: "Post Title"
description: "One-sentence summary for SEO/social."
author: "Your Name"
publishedAt: "2026-08-22"
tags: ["announcements"]
---

Post body in Markdown/MDX goes here.
```

Drop it in `content/blog/`, commit, redeploy. It'll appear at `/blog/<filename-without-extension>`.

## Environment variables

See `.env.example` for the full list with comments. Everything is optional except
`GOOGLE_API_KEY` (Gemini sentiment classifier) — every other integration degrades gracefully
(returns empty results, or skips the feature) if its keys are unset. Notable ones:

- `GOOGLE_API_KEY` — required; Gemini sentiment classification
- `CLOUDFLARE_API_TOKEN` / `CLOUDFLARE_ACCOUNT_ID` — fallback sentiment classifier
- `LLM7_API_KEY` — movement insight + market summary text generation
- `FINNHUB_API_KEY`, `ALPACA_API_KEY_ID` / `ALPACA_API_SECRET_KEY` — market data / news
- `UNSPLASH_ACCESS_KEY` — article thumbnail images
- `MAILGUN_API_KEY`, `MAILGUN_DOMAIN` — market summary email subscription
- `ADMIN_API_TOKEN` — protects `POST /api/market-summary/generate`
- `NEXT_PUBLIC_SUPABASE_URL`, `NEXT_PUBLIC_SUPABASE_ANON_KEY` — Supabase Auth (accounts, watchlist,
  search history)
- `SUPABASE_SERVICE_ROLE_KEY` — server-only; used for account deletion, never exposed client-side
- `INDEXNOW_KEY` — IndexNow push-to-index verification
- `SITE_BASE_URL`, `DATABASE_FILENAME` — site/infra config
- `ENABLE_MARKET_SUMMARY`, `ENABLE_MAG7_SENTIMENT` — feature toggles for the cron jobs
- `UMAMI_SCRIPT_URL`, `UMAMI_WEBSITE_ID` — optional self-hosted analytics

## Scripts

| Command | What it does |
| --- | --- |
| `npm run dev` | Start the dev server |
| `npm run build` | Production build (also runs `prebuild`, which writes the IndexNow verification file) |
| `npm run start` | Run the production build |
| `npm run db:migrate` | Apply any pending SQLite migrations (also happens automatically on first DB connection) |
| `npm run lint` | ESLint |

There is no automated test suite — the original Flask app had none either.

## Rate limits

All mutating and data-fetching API routes go through `lib/ratelimit` (in-memory, per-IP token
bucket) via the `withRateLimit` wrapper — see individual route handlers under `app/api/` for
per-route limits.

## Deployment

Runs as a systemd service on a self-hosted VPS behind Caddy (HTTP/2 + HTTP/3, automatic HTTPS),
with SQLite backed up via `deploy/backup-db.sh`. `.github/workflows/deploy.yml` builds and
deploys automatically on every push to `master`.

## License

[MIT](LICENSE)
