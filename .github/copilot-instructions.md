# Copilot Instructions

## Project snapshot
- Single Flask app in `main.py` renders `templates/index.html` and serves `static/index.js`; most interactions are client-side fetch flows.
- SQLAlchemy + SQLite are used for app persistence (`ip_log.db`) and the blog writer bind (`blog.db`), plus cached records (market summaries, article image metadata).
- Frontend search flow expects `search_stock()` JSON payloads with `stock_info`, `historical_data`, `articles`, and optional `movement_insight`; sentiment totals are computed client-side via `/sentiment` per article.
- Runtime includes APScheduler bootstrap for market-summary automation and scheduled delivery.

## Architecture map
1. **Search flow**: `/search` resolves tickers via `yahooquery.search`, hydrates metrics with `get_stock_info()`, loads chart points via `get_historical_data()`, and returns news from `get_news_articles()`.
2. **Sentiment pipeline**: frontend progressively calls `/sentiment` for each article (`progressivelyAnalyzeSentiment` in `static/index.js`); backend uses `analyze_sentiment()` with Cloudflare as primary and Gemma backup (`wait_for_ai_rate_slot()` + retry helpers).
3. **Movement insight**: `build_movement_insight()` gates on ±3% swings, fetches Finnhub (falls back to Google News) and summarizes through `summarize_stock_movement()` (LLM7) or `fallback_movement_summary()`.
4. **Trending dashboards**: page routes are `/trending-list` and `/trending-list/<source>`; data APIs are `/trending` and `/trending/<source>`, both backed by `get_trending_source_data()` for ApeWisdom/StockTwits (`cloudscraper`) /Alpaca.
5. **Additional surfaces**: market summaries (`/market-summary` + `/api/market-summary/*`), quote hydration (`/quote/<symbol>`), StockTwits summaries (`/stocktwits/<symbol>/summary`), and blog writer/admin APIs (`/write`, `/api/blog/*`).
6. **Frontend contract**: `templates/index.html` is the shell and `static/index.js` contains the SPA logic (search, trending tabs, watchlist, market-summary rendering, sentiment streaming).

## Dev workflows
- **Setup**: `python -m venv .venv && source .venv/bin/activate`, `pip install -r requirements.txt`, create `.env` matching README keys.
- **Required env**: `GOOGLE_API_KEY` (Gemma client init) and `FLASK_SECRET_KEY` (session/auth) must exist or startup fails.
- **Run**: `python main.py` (binds to `0.0.0.0:5000`). Use browser + `curl` to sanity check `/search`, `/sentiment`, `/historical/<symbol>/<period>`, `/trending`, and market-summary endpoints.
- **Optional integrations**: missing LLM7/Finnhub/Cloudflare/Mailgun/Alpaca degrade gracefully via fallback responses and/or feature disablement.
- **Rate limits**: Flask-Limiter caps each route (see decorators in `main.py`); adjust via the `Limiter` config or per-route decorators when adding endpoints.
- **Dependency changes**: add libraries to `requirements.txt`; nothing else manages deps.

## Coding patterns & gotchas
- Reuse existing helpers (e.g., `get_trending_source_data`, `analyze_sentiment`, `build_movement_insight`) instead of duplicating HTTP logic; they already normalize payloads for frontend expectations.
- Keep responses small: the frontend renders everything client-side, so attach only serializable primitives (dicts/lists) and format numbers client-side (see `formatNumber`, `formatCurrency`).
- AI calls must be throttled—if you add new Gemma usage, feed through `wait_for_ai_rate_slot()` or reuse existing queues to avoid quota bans.
- `/search` should stay fast and avoid bundling expensive per-article sentiment work; preserve the current pattern where sentiment is streamed separately by `/sentiment`.
- Trending APIs sometimes rate-limit; wrap new integrations with short `timeout` and graceful `except` blocks returning `[]`, matching current UX.
- Historical data uses mixed intervals (5m intraday, 1h weekly, etc.); extend by following the branching inside `get_historical_data()` so Chart.js keeps uniform `{date, price}` points.
- When adding UI behaviors, edit `static/index.js` (and `templates/index.html` only for markup/data attributes); there’s no build step, so keep helpers modular and reuse existing DOM IDs.

## Verification
- No automated tests currently; run the Flask server locally and exercise `/`, `/results`, `/search`, `/sentiment`, `/historical/<symbol>/<period>`, `/trending`, `/trending/<source>`, and `/market-summary` flows manually.
- Check server logs for "Error ..." prints—those are the primary debugging breadcrumbs today.
- Confirm sentiment counts (streamed via `/sentiment`), movement highlights, watchlist hydration, and market-summary widgets render correctly after backend changes; mismatched keys immediately surface in the dashboard.
