
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Stocks Sentiment Analysis

A Flask web app for real-time stock search, trending dashboards, sentiment analysis, and an authenticated writer workspace using Yahoo Finance, ApeWisdom, StockTwits, Alpaca, Google News, Finnhub, and LLM-powered summaries. SQLite backs the Unsplash image cache, market summary history, and the internal blog editor; everything else is fetched live and rendered client-side.

## Features
- Search stocks by ticker or company name
- Dedicated pages for watchlist (`/watchlist`), search results (`/results`), and trending dashboards (`/trending-list`)
- Trending stocks from Reddit (ApeWisdom), StockTwits, and Alpaca
- Real-time price charts (Chart.js)
- News aggregation and AI-powered sentiment analysis (Gemma, LLM7, Cloudflare)
- Market summary dashboard with daily wrap + email subscription (Mailgun)
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
  FLASK_SECRET_KEY=replace_with_random_string

  # Optional: AI
  LLM7_API_KEY=your_llm7_key
  LLM7_BASE_URL=https://api.llm7.io/v1
  LLM7_MODEL=fast

  # Optional: data sources
  ALPACA_API_KEY_ID=your_alpaca_key
  ALPACA_API_SECRET_KEY=your_alpaca_secret
  FINNHUB_API_KEY=your_finnhub_key

  # Optional: Unsplash image cache (used for article thumbnails)
  UNSPLASH_ACCESS_KEY=your_unsplash_key
  UNSPLASH_APP_NAME=stocks-sentiment-analysis
  UNSPLASH_DEFAULT_QUERY="stock market"

  # Optional: Cloudflare AI fallback sentiment
  CLOUDFLARE_ACCOUNT_ID=your_cf_account
  CLOUDFLARE_API_TOKEN=your_cf_token
  CLOUDFLARE_SENTIMENT_MODEL=@cf/meta/llama-3-8b-instruct

  # Optional: Market summary email subscription (Mailgun)
  MAILGUN_API_KEY=your_mailgun_key
  MAILGUN_DOMAIN=your_mailgun_domain
  MAILGUN_MARKET_LIST_ADDRESS=marketsummary@your_mailgun_domain

  # Optional: Toggle market summary automation
  ENABLE_MARKET_SUMMARY=1

  # Authenticated writer workspace
  # Required: use a long, random string and keep it stable across deployments
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

### Pages (HTML)
- `/` (GET): Main app UI.
- `/privacy` (GET): Privacy policy page.
- `/watchlist` (GET): Watchlist view.
- `/results` (GET): Dedicated search results page.
- `/trending-list` (GET): Trending dashboards page.
- `/trending-list/<source>` (GET): Trending dashboard for a specific source.
- `/market-summary` (GET): Market summary landing page.
- `/market-summary/stock-market-today` (GET): Always show latest market summary.
- `/market-summary/<slug>` (GET): Dedicated market summary article page.
- `/blog` (GET): Blog listing.
- `/blog/<slug>` (GET): Blog article detail page.
- `/write` (GET): Auth-only writer workspace with the mini word processor UI.
- `/confirm` (GET): Market summary subscription confirmation page.

### Public JSON APIs
- `/search` (POST): Search for a stock; returns JSON with `stock_info`, `historical_data`, and `articles`.
- `/movement-insight` (POST): Build movement insight from `stock_info` or `symbol`.
- `/historical/<symbol>/<period>` (GET): Get historical price data for a symbol and period.
- `/trending` (GET): Get trending stocks (Reddit/ApeWisdom).
- `/trending/<source>` (GET): Trending stocks from `stocktwits`, `reddit`, or `volume`.
- `/trending/prices` (POST): Batch quote hydration for trending symbols.
- `/sentiment` (POST): Analyze sentiment for a stock/news article.
- `/quote/<symbol>` (GET): Quick price/quote lookup.
- `/stocktwits/<symbol>/summary` (GET): StockTwits summary for a symbol.
- `/api/market-summary/latest` (GET): Latest market summary (JSON).
- `/api/market-summary/week-glance` (GET): Weekly index snapshots used by the market summary dashboard.
- `/api/market-summary/archive` (GET): Market summary archive (JSON).
- `/api/market-summary/<slug>` (GET): Fetch a specific market summary by slug (ISO date or `id-<pk>`).
- `/api/market-summary/subscribe` (POST): Subscribe to market summary email updates (Mailgun).

### Authenticated APIs (writer/admin)
- `/write/login` (POST): Authenticate writer/admin session.
- `/write/logout` (POST): End writer/admin session.
- `/api/blog/articles` (GET/POST): Manage private blog articles stored in `blog.db`.
- `/api/blog/articles/<article_identifier>` (PUT/PATCH/DELETE): Update or delete a draft.
- `/api/blog/articles/<article_identifier>/publish` (POST): Publish a draft to the public blog.
- `/api/blog/articles/<article_identifier>/unpublish` (POST): Revert a post back to draft.
- `/api/market-summary/generate` (POST): Force regenerate the market summary (admin only).

### Static assets
- `/robots.txt` (GET): Robots file.
- `/sitemap.xml` (GET): Sitemap file.

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

- **No external database service required** (SQLite stores Unsplash cache, market summaries, and the authenticated blog workspace)
- All data fetched live from third-party APIs
- Sentiment and movement summaries use Gemma, LLM7, or Cloudflare (rate-limited)
- Trending APIs are wrapped with timeouts and error handling
- Flask-Limiter caps all endpoints (see `main.py`)

## Development

- Add new dependencies to `requirements.txt`
- Use `curl` or browser to test endpoints
- Check server logs for errors

## Environment Variables

Required:
- `GOOGLE_API_KEY` (Gemma)
- `FLASK_SECRET_KEY` or `SECRET_KEY` (session signing; must be the same across all workers)

Optional (degrades gracefully if missing):
- `LLM7_API_KEY`, `LLM7_BASE_URL`, `LLM7_MODEL`
- `ALPACA_API_KEY_ID`, `ALPACA_API_SECRET_KEY` (or `ALPACA_API_KEY`, `ALPACA_SECRET_KEY` aliases)
- `FINNHUB_API_KEY`
- `UNSPLASH_ACCESS_KEY`, `UNSPLASH_APP_NAME`, `UNSPLASH_DEFAULT_QUERY`
- `CLOUDFLARE_ACCOUNT_ID`, `CLOUDFLARE_API_TOKEN`, `CLOUDFLARE_SENTIMENT_MODEL`
- `MAILGUN_API_KEY`, `MAILGUN_DOMAIN`, `MAILGUN_MARKET_LIST_ADDRESS`, `MAILGUN_FROM_EMAIL` (email market summary subscription)
- `ENABLE_MARKET_SUMMARY` (set to `0` to disable market summary generation)
- `FLASK_SKIP_SCHEDULER` (set to `1` to disable APScheduler startup in a process)
- `MARKET_SUMMARY_RELEASE_HOUR`, `MARKET_SUMMARY_RELEASE_MINUTE`, `MARKET_SUMMARY_RETENTION_DAYS`, `MARKET_SUMMARY_MAX_HEADLINES`
- `BLOG_ARTICLE_FETCH_LIMIT`
- `GEMMA_MAX_CALLS_PER_MINUTE`, `GEMMA_RATE_WINDOW_SECONDS`, `GEMMA_SENTIMENT_TIMEOUT_SECONDS`
- `BLOG_ADMIN_USERNAME`, `BLOG_ADMIN_PASSWORD`, `BLOG_DEFAULT_AUTHOR` (unlock the `/write` workspace)

## Rate Limits

- All endpoints are rate-limited via Flask-Limiter (see decorators in `main.py`)
- Adjust limits via environment or per-route decorators

## License

[MIT](LICENSE)
