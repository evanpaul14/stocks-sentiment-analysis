
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
  git clone https://github.com/evanpaul14/stocks-sentiment-analysis.git
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
| Method | Path | Auth Required | Description |
| --- | --- | --- | --- |
| GET | `/` | No | Main app UI. |
| GET | `/privacy` | No | Privacy policy page. |
| GET | `/watchlist` | No | Watchlist view. |
| GET | `/results` | No | Dedicated search results page. |
| GET | `/trending-list` | No | Trending dashboards page. |
| GET | `/trending-list/<source>` | No | Trending dashboard for a specific source. |
| GET | `/market-summary` | No | Market summary landing page. |
| GET | `/market-summary/stock-market-today` | No | Always show latest market summary. |
| GET | `/market-summary/<slug>` | No | Dedicated market summary article page. |
| GET | `/blog` | No | Blog listing. |
| GET | `/blog/<slug>` | No | Blog article detail page. |
| GET | `/write` | No | Writer workspace page (prompts for auth in UI). |
| GET | `/confirm` | No | Market summary subscription confirmation page. |

### Public JSON APIs
| Method | Path | Auth Required | Description |
| --- | --- | --- | --- |
| POST | `/search` | No | Search for a stock; returns `stock_info`, `historical_data`, and `articles`. |
| POST | `/movement-insight` | No | Build movement insight from `stock_info` or `symbol`. |
| GET | `/historical/<symbol>/<period>` | No | Get historical price data for a symbol and period. |
| GET | `/trending` | No | Get trending stocks (Reddit/ApeWisdom). |
| GET | `/trending/<source>` | No | Get trending stocks from `stocktwits`, `reddit`, or `volume`. |
| POST | `/trending/prices` | No | Batch quote hydration for trending symbols. |
| POST | `/sentiment` | No | Analyze sentiment for a stock/news article. |
| GET | `/quote/<symbol>` | No | Quick price/quote lookup. |
| GET | `/stocktwits/<symbol>/summary` | No | StockTwits summary for a symbol. |
| GET | `/api/market-summary/latest` | No | Latest market summary payload. |
| GET | `/api/market-summary/week-glance` | No | Weekly index snapshots for the market summary dashboard. |
| GET | `/api/market-summary/archive` | No | Market summary archive payload. |
| GET | `/api/market-summary/<slug>` | No | Specific market summary by slug (`YYYY-MM-DD` or `id-<pk>`). |
| POST | `/api/market-summary/subscribe` | No | Subscribe to market summary email updates (Mailgun). |

### Authenticated APIs (writer/admin)
| Method | Path | Auth Required | Description |
| --- | --- | --- | --- |
| POST | `/write/login` | No | Authenticate writer/admin session. |
| POST | `/write/logout` | No | End writer/admin session. |
| GET, POST | `/api/blog/articles` | Yes | List or create private blog articles in `blog.db`. |
| PUT, PATCH, DELETE | `/api/blog/articles/<article_identifier>` | Yes | Update or delete a draft. |
| POST | `/api/blog/articles/<article_identifier>/publish` | Yes | Publish a draft to the public blog. |
| POST | `/api/blog/articles/<article_identifier>/unpublish` | Yes | Revert a post back to draft. |
| POST | `/api/market-summary/generate` | Yes | Force regenerate the market summary. |

### Static assets
| Method | Path | Auth Required | Description |
| --- | --- | --- | --- |
| GET | `/robots.txt` | No | Robots file. |
| GET | `/sitemap.xml` | No | Sitemap file. |

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
