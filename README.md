
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Stocks Sentiment Analysis

A Flask web app for real-time stock search, trending dashboards, sentiment analysis, and an authenticated writer workspace using Yahoo Finance, ApeWisdom, StockTwits, Alpaca, Google News, Finnhub, and LLM-powered summaries. SQLite backs the Unsplash image cache, market summary history, and the internal blog editor; everything else is fetched live and rendered client-side.

## Features
- Search stocks by ticker or company name
- Trending stocks from Reddit (ApeWisdom), StockTwits, and Alpaca
- Real-time price charts (Chart.js)
- News aggregation and AI-powered sentiment analysis (Gemma, LLM7, Cloudflare)
- Market summary dashboard (optional, with daily wrap)
- Unsplash-powered article images
- Private `/write` page with a mini word processor that saves long-form articles into `blog.db`
- Rate-limited endpoints for API safety

## Quickstart

1. **Clone & setup:**
  ```bash
  git clone https://github.com/evanpaul/stocks-sentiment-analysis.git
  cd stocks-sentiment-analysis
  python -m venv .venv && source .venv/bin/activate
  pip install -r requirements.txt
  ```

2. **Configure `.env`:**
  Create a `.env` file with these keys:
  ```
  GOOGLE_API_KEY=your_google_gemma_key
  LLM7_API_KEY=your_llm7_key
  LLM7_BASE_URL=https://api.llm7.io/v1
  ALPACA_API_KEY_ID=your_alpaca_key
  ALPACA_API_SECRET_KEY=your_alpaca_secret
  FINNHUB_API_KEY=your_finnhub_key
  UNSPLASH_ACCESS_KEY=your_unsplash_key
  # Optional: Cloudflare AI
  CLOUDFLARE_ACCOUNT_ID=your_cf_account
  CLOUDFLARE_API_TOKEN=your_cf_token
  # Authenticated writer workspace
  # Required: use a long, random string and keep it stable across deployments
  FLASK_SECRET_KEY=replace_with_random_string
  BLOG_ADMIN_USERNAME=choose_a_username
  BLOG_ADMIN_PASSWORD=choose_a_password
  BLOG_DEFAULT_AUTHOR=stocksentimentapp.com Team
  ```

3. **Run the app:**
  ```bash
  python main.py
  # Visit http://127.0.0.1:5000/
  ```

## API Endpoints

- `/search` (POST): Search for a stock, returns JSON with `stock_info`, `historical_data`, `articles`, `sentiment_summary`, `overall_sentiment`, and optional `movement_insight`.
- `/historical/<symbol>/<period>` (GET): Get historical price data for a symbol and period.
- `/trending` (GET): Get trending stocks (Reddit/ApeWisdom).
- `/trending/<source>` (GET): Trending stocks from `stocktwits` or `alpaca`.
- `/sentiment` (POST): Analyze sentiment for a stock/news article.
- `/market-summary` (GET): Landing page for the latest market summary editions (if enabled).
- `/market-summary/stock-market-today` (GET): SEO-friendly page that always shows the latest market summary article.
- `/market-summary/<slug>` (GET): Dedicated article page for a specific daily summary (e.g., `/market-summary/2024-07-08`).
- `/api/market-summary/latest` (GET): Latest market summary (JSON).
- `/api/market-summary/archive` (GET): Market summary archive (JSON).
- `/api/market-summary/<slug>` (GET): Fetch a specific market summary by slug (ISO date or `id-<pk>`).
- `/quote/<symbol>` (GET): Quick price/quote lookup.
- `/stocktwits/<symbol>/summary` (GET): StockTwits summary for a symbol.
- `/write` (GET): Auth-only writer workspace with the mini word processor UI.
- `/write/login` (POST): JSON login endpoint that unlocks the writer workspace (`BLOG_ADMIN_*` credentials).
- `/write/logout` (POST): Ends the writer session.
- `/api/blog/articles` (GET/POST): Manage private blog articles stored in `blog.db` (requires an authenticated writer session).

## Frontend

- Single-page app in `templates/index.html` (vanilla JS + Chart.js)
- Trending cards and search bar trigger backend endpoints
- All formatting (currency, numbers) is client-side

## Writer Workspace

- Visit `/write` and unlock the page with `BLOG_ADMIN_USERNAME` / `BLOG_ADMIN_PASSWORD` (set in `.env`).
- The page bundles a lightweight word processor (contenteditable + formatting toolbar) where admins can craft long-form posts, specify a hero image URL, and save directly to SQLite.
- Articles persist inside `blog.db` through the SQLAlchemy `blog` bind, alongside created/updated timestamps and auto-generated slugs.
- `/api/blog/articles` responds with JSON so you can preview or repurpose drafts elsewhere; the endpoint stays locked behind the same session to avoid public exposure.

## Architecture

- **No database required for public features** (SQLite only stores Unsplash cache, market summaries, and the authenticated blog workspace)
- All data fetched live from third-party APIs
- Sentiment and movement summaries use Gemma, LLM7, or Cloudflare (rate-limited)
- Trending APIs are wrapped with timeouts and error handling
- Flask-Limiter caps all endpoints (see `main.py`)

## Development

- Add new dependencies to `requirements.txt`
- Use `curl` or browser to test endpoints
- Check server logs for errors
- No automated tests; verify manually

## Environment Variables

Required:
- `GOOGLE_API_KEY` (Gemma)
- `FLASK_SECRET_KEY` (session signing; must be the same across all workers)

Optional (degrades gracefully if missing):
- `LLM7_API_KEY`, `LLM7_BASE_URL`, `LLM7_MODEL`
- `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY`
- `FINNHUB_API_KEY`
- `UNSPLASH_ACCESS_KEY`, `UNSPLASH_APP_NAME`, `UNSPLASH_DEFAULT_QUERY`
- `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_SENTIMENT_MODEL`
- `BLOG_ADMIN_USERNAME`, `BLOG_ADMIN_PASSWORD`, `BLOG_DEFAULT_AUTHOR` (unlock the `/write` workspace)

## Rate Limits

- All endpoints are rate-limited via Flask-Limiter (see decorators in `main.py`)
- Adjust limits via environment or per-route decorators

## License

[MIT](LICENSE)