# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Setup & Running

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py  # binds to 0.0.0.0:5000
```

**Required env vars** (app fails to start without these):
- `GOOGLE_API_KEY` — Gemma LLM client
- `FLASK_SECRET_KEY` — session/auth signing

All other env vars (LLM7, Alpaca, Finnhub, Cloudflare, Mailgun, Unsplash, blog admin) are optional and degrade gracefully. See README.md for the full `.env` template.

**Sitemap generation:**
```bash
python scripts/generate_sitemap.py [--db <path>] [--output <path>] [--base-url <url>]
```

## Testing

No automated tests. Exercise flows manually via browser and curl:
- `/`, `/results`, `/search`, `/sentiment`, `/historical/<symbol>/<period>`
- `/trending`, `/trending/<source>`, `/market-summary`

Check server logs for `"Error ..."` prints — those are the primary debugging breadcrumbs.

## Architecture

**Monolithic Flask SPA** (`main.py` ~161KB) serving `templates/index.html` as a single-page shell. `static/index.js` contains all SPA logic; no frontend build step.

### Key data flows

1. **Search**: `/search` (POST) → resolve ticker via `yahooquery.search` → `get_stock_info()` + `get_historical_data()` + `get_news_articles()` → lightweight JSON. Sentiment is **not** bundled here.

2. **Sentiment pipeline**: Frontend calls `/sentiment` (POST) per article after search results render (`progressivelyAnalyzeSentiment` in `index.js`). Backend runs `analyze_sentiment()` — Cloudflare is primary, Gemma is backup. All new Gemma calls must go through `wait_for_ai_rate_slot()` to avoid quota bans.

3. **Movement insight**: `build_movement_insight()` gates on ±3% price swings. Fetches Finnhub headlines (falls back to Google News) → `summarize_stock_movement()` via LLM7 or `fallback_movement_summary()`.

4. **Trending dashboards**: Page routes `/trending-list` and `/trending-list/<source>`; data APIs `/trending` and `/trending/<source>`, both backed by `get_trending_source_data()` pulling from ApeWisdom (Reddit), StockTwits (cloudscraper), and Alpaca.

5. **Market summary (automated)**: APScheduler runs daily at configurable time → fetches headlines → generates AI summary → stores in `ip_log.db` → emails via Mailgun. Disable with `FLASK_SKIP_SCHEDULER=1` or `ENABLE_MARKET_SUMMARY=0`.

6. **Blog writer workspace**: Auth-gated `/write` page with a rich text editor. Articles saved to `blog.db` (separate SQLAlchemy bind). APIs at `/api/blog/*`.

### Persistence

- `ip_log.db` — app state: `ArticleImage` (Unsplash cache), `MarketSummary`, `MarketSummarySendLog`
- `blog.db` — `BlogArticle` records (drafts + published)

No ORM migrations — schema created via `db.create_all()` on startup.

### Rate limiting

Flask-Limiter on all endpoints (100/hr default). Adjust via `Limiter` config or per-route decorators when adding endpoints.

## Coding Patterns

- **Reuse existing helpers** (`get_trending_source_data`, `analyze_sentiment`, `build_movement_insight`) instead of duplicating HTTP/normalization logic.
- **Keep `/search` fast** — don't bundle per-article sentiment work; preserve the current deferred-streaming pattern.
- **Frontend contract**: `search_stock()` JSON expects `stock_info`, `historical_data`, `articles`, and optional `movement_insight`. Format numbers client-side (`formatNumber`, `formatCurrency`); keep backend responses as plain dicts/lists.
- **Historical data intervals**: `get_historical_data()` uses mixed intervals (5m intraday, 1h weekly, etc.). Follow the branching there when extending — Chart.js expects uniform `{date, price}` tuples.
- **Trending integrations**: wrap with short `timeout` and `except` blocks returning `[]` to match current graceful-fail UX.
- **UI changes**: edit `static/index.js` and `templates/index.html` only; no build step. Reuse existing DOM IDs.
