# Copilot Instructions

## Project snapshot
- Single Flask app in `main.py` renders `templates/index.html`; no ORM or background workers—everything is on-demand HTTP + fetch()
- Frontend expects JSON payloads shaped exactly like `search_stock()` returns (`stock_info`, `historical_data`, `articles`, `sentiment_summary`, `overall_sentiment`, optional `movement_insight`)
- No database; every call fans out to third-party APIs (Yahoo Finance, ApeWisdom, StockTwits, Alpaca, Google News/Gemma, Finnhub, LLM7)

## Architecture map
1. **Search flow**: `/search` resolves tickers via `yahooquery.search`, hydrates metrics with `get_stock_info()`, charts via `get_historical_data()`, news from `get_news_articles()`, then sentiment via `build_sentiment_payload()` (Gemma).
2. **Sentiment pipeline**: `build_sentiment_payload()` batches article scraping + `analyze_sentiment_gemma()`; respect `wait_for_ai_rate_slot()` and retry helpers before touching Gemma.
3. **Movement insight**: `build_movement_insight()` gates on ±3% swings, fetches Finnhub (falls back to Google News) and summarizes through `summarize_stock_movement()` (LLM7) or `fallback_movement_summary()`.
4. **Trending dashboards**: `/trending` + `/trending/<source>` fan into `get_trending_source_data()`, which delegates to ApeWisdom/StockTwits (via `cloudscraper`) /Alpaca helpers; `lookup_company_name()` is LRU-cached to avoid duplicate yfinance calls.
5. **Frontend contract**: `templates/index.html` drives everything with vanilla JS + Chart.js; period tabs reuse cached `historicalData` map, and trending cards re-trigger `searchStock()` with their ticker symbol.

## Dev workflows
- **Setup**: `python -m venv .venv && source .venv/bin/activate`, `pip install -r requirements.txt`, create `.env` matching the README (GOOGLE_API_KEY, LLM7 keys, Alpaca, Finnhub, etc.).
- **Run**: `python main.py` (binds to `0.0.0.0:5000`). Use `curl http://127.0.0.1:5000/trending` or the browser UI to sanity check responses.
- **API keys**: missing keys degrade gracefully (e.g., movement summaries return fallback text), but Gemma key is mandatory—raise early.
- **Rate limits**: Flask-Limiter caps each route (see decorators in `main.py`); adjust via the `Limiter` config or per-route decorators when adding endpoints.
- **Dependency changes**: add libraries to `requirements.txt`; nothing else manages deps.

## Coding patterns & gotchas
- Reuse existing helpers (e.g., `get_trending_source_data`, `build_sentiment_payload`) instead of duplicating HTTP logic; they already normalize payloads for the frontend expectations.
- Keep responses small: the frontend renders everything client-side, so attach only serializable primitives (dicts/lists) and format numbers client-side (see `formatNumber`, `formatCurrency`).
- AI calls must be throttled—if you add new Gemma/LLM7 usage, feed through `wait_for_ai_rate_slot()` or reuse existing queues to avoid quota bans.
- Trending APIs sometimes rate-limit; wrap new integrations with short `timeout` and graceful `except` blocks returning `[]`, matching current UX.
- Historical data uses mixed intervals (5m intraday, 1h weekly, etc.); extend by following the branching inside `get_historical_data()` so Chart.js keeps uniform `{date, price}` points.
- When adding UI behaviors, edit `templates/index.html` directly—there’s no build step, but keep inline JS modular (helper per concern) and hook into existing DOM IDs to avoid breaking search/trending toggles.

## Verification
- No automated tests; run the Flask server locally and exercise `/`, `/search`, `/historical/<symbol>/<period>`, and `/trending/<source>` manually.
- Check server logs for "Error ..." prints—those are the primary debugging breadcrumbs today.
- Confirm sentiment counts and movement highlights render in the UI after backend changes; mismatched keys immediately show as missing fields in the dashboard.
