# Stock Sentiment Analysis

A real-time stock analysis dashboard combining financial data, AI-powered news sentiment, and multi-source trending tickers. Search a company to view live prices, historical charts, key stats, and sentiment analysis of recent news. Explore trending stocks from Reddit, StockTwits, and Alpaca volume leaders—all in a modern, responsive UI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- Multi-source trending dashboards: Reddit (ApeWisdom), StockTwits and Alpaca volume leaders.
- Search by company name to resolve a ticker and retrieve detailed stock data.
- Real-time price and after-hours information, market statistics, and a company overview.
- Interactive historical charts powered by Chart.js (1D, 1W, 1M, 3M, YTD, 1Y, 5Y, ALL).
- AI-powered news sentiment per article using Google Gemma (via google-genai) and a sentiment summary.
- Intraday movement insights: when a stock moves more than 3% intraday, the app summarizes key catalysts using Finnhub headlines and LLM7 (if available).
- Scheduled “Market Summary” page: every weekday at 4:15 PM ET the app captures top index moves + Google News headlines, runs them through LLM7, saves the article, and exposes a browsable archive.
- Rate-limited endpoints to protect third-party quotas and avoid abuse.

## Technology Stack

**Backend:** Flask, Flask-Limiter, Flask-SQLAlchemy (SQLite), yfinance, yahooquery, pygooglenews, google-genai (Gemma), openai (LLM7 client), finnhub-python, cloudscraper, requests, APScheduler, python-dotenv, gunicorn

**Frontend:** HTML5/CSS3, Chart.js, Vanilla JavaScript (no build step)

**External APIs:** ApeWisdom (Reddit), StockTwits, Alpaca, Yahoo Finance, Google News, Google Gemma AI, Finnhub, LLM7

## Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemma AI) — REQUIRED. The application will raise a ValueError and refuse to start if this is not set.
- Alpaca API Key/Secret — optional for the volume dashboard. The app accepts either `ALPACA_API_KEY_ID` (or `ALPACA_API_KEY`) and `ALPACA_API_SECRET_KEY` (or `ALPACA_SECRET_KEY`).
- FINNHUB_API_KEY — optional; Finnhub headlines are used to explain moves when present (fallback to Google News if not set).
- LLM7_API_KEY — optional; used to generate concise movement summaries. If missing the app will fall back to a shorter message.
- An Internet connection for real-time data

## Installation

1. Clone the repository
```bash
git clone https://github.com/yourusername/stocks-sentiment-analysis.git
cd stocks-sentiment-analysis
```
2. Create a virtual environment and activate it
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```
3. Install dependencies
```bash
pip install -r requirements.txt
```
4. Create a `.env` file in the project root. Example:
```ini
GOOGLE_API_KEY=your_google_api_key_here
# Optional Alpaca keys (either of the two naming sets will be accepted)
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
FINNHUB_API_KEY=your_finnhub_api_key_here
LLM7_API_KEY=your_llm7_api_key_here
LLM7_BASE_URL=https://api.llm7.io/v1  # optional override
LLM7_MODEL=fast                      # optional override
GEMMA_MAX_CALLS_PER_MINUTE=45       # (optional) in-process throttle for Gemma calls
GEMMA_RATE_WINDOW_SECONDS=60
TRENDING_PRICE_TTL_SECONDS=120      # cache TTL for small price snapshots used on trending cards
```

Note: The app requires a valid `GOOGLE_API_KEY`; it will raise an error if unset. Everything else is optional but enhances features.

## Usage

Start the Flask server locally
```bash
python main.py
```
The server will bind to `0.0.0.0:5000` by default.

Open `http://127.0.0.1:5000` in your browser.

## API Endpoints

This app exposes a small JSON API consumed by the front-end. All endpoints are rate limited (configured via Flask-Limiter in `main.py`).

- `GET /` — render the main UI (index.html). Rate Limit: 50/min
- `GET /watchlist` — same app with watchlist page view. Rate Limit: 50/min
- `GET /results?q=<query>` — search results page (client-side flow). Rate Limit: 50/min
- `GET /trending-list` — render the dedicated trending dashboards page. Rate Limit: 50/min
- `GET /market-summary` — render the Market Summary page (archives + latest wrap). Rate Limit: 50/min
- `POST /search` — search for a company name and return a combined payload containing `stock_info`, `historical_data`, `articles`, `sentiment_summary`, `overall_sentiment` and optional `movement_insight`. Rate Limit: 10/min
  Request JSON: `{ "company_name": "Apple" }`

- `GET /historical/<symbol>/<period>` — return historical price data for Chart.js in that timeframe. Periods include: `1d`, `7d`/`1w`, `1mo`, `3mo`, `ytd`, `1y`, `5y`, `max`. Rate Limit: 50/min

- `GET /trending` — return combined trending data for `stocktwits`, `reddit`, and `volume` (Alpaca). Rate Limit: 30/min
- `GET /trending/<source>` — return a single source's trending data: `stocktwits`, `reddit`, or `volume`. Rate Limit: 30/min
- `GET /quote/<symbol>` — small snapshot used by the frontend to refresh watchlist prices `{symbol, price, change_percent, timestamp}`. Rate Limit: 50/min
- `GET /api/market-summary/latest` — latest published market summary article + metadata. Rate Limit: 30/min
- `GET /api/market-summary/archive?limit=20` — most recent archived summaries (default 20, max 60). Rate Limit: 30/min

### Response shapes (high level)

The `POST /search` response includes the following shape:
```json
{
  "stock_info": { /* ticker metadata from yfinance */ },
  "historical_data": [ /* {date, price} points */ ],
  "articles": [ /* articles with sentiment */ ],
  "sentiment_summary": { "positive": 0, "negative": 0, "neutral": 0 },
  "overall_sentiment": "positive|negative|neutral",
  "movement_insight": { /* optional when intraday move >= 3% */ }
}
```

## Configuration / Tuning

- Change the Gemma model in `main.py` by editing `gemma_model` (default `gemma-3-27b-it`).
- Adjust in-process Gemma throttling with `GEMMA_MAX_CALLS_PER_MINUTE` and `GEMMA_RATE_WINDOW_SECONDS` in `.env`.
- The number of news articles analyzed is controlled in code via `get_news_articles(symbol, num_articles)` (default is 10).

### Market Summary automation

- The Market Summary page uses APScheduler to run a background job each weekday at **4:15 PM Eastern**. The job collects S&P 500 / Nasdaq / Dow closing stats, scrapes Google News for “stock market today” style headlines, and asks LLM7 to write a short wrap. Articles are stored in SQLite and exposed via `/api/market-summary/latest` plus `/api/market-summary/archive`.
- Set `ENABLE_MARKET_SUMMARY=0` to disable the feature entirely, or `FLASK_SKIP_SCHEDULER=1` to keep the scheduler from running (useful on secondary workers or during development). The publish slot and headline count can be tuned with `MARKET_SUMMARY_RELEASE_HOUR`, `MARKET_SUMMARY_RELEASE_MINUTE`, and `MARKET_SUMMARY_MAX_HEADLINES`.
- Google News and LLM7 are both required for the richest article, but the job will fall back to a deterministic text summary if LLM7 is unavailable.

## Tips & Notes

- `GOOGLE_API_KEY` is mandatory; after setting it the app initializes a `google-genai` client. A missing key results in startup failure.
- `LLM7_API_KEY` and `FINNHUB_API_KEY` are optional but improve the movement insight box when present.
- Alpaca keys are optional; when present the `/trending` volume leaderboard includes Alpaca data.
- Rate limiting is configured using `Flask-Limiter` in `main.py`; you can modify the `default_limits` or per-route limits there.
- The app uses an in-process queue (`wait_for_ai_rate_slot`) to smooth Gemma requests and avoid bursts that may exhaust quotas.

## Project Structure

```
stocks-sentiment-analysis/
├── main.py               # Flask backend application
├── templates/
│   └── index.html        # Frontend interface
├── static/
│   └── favicon.ico       # Website icon
├── .env                  # Environment variables (create this)
├── requirements.txt      # Python dependencies
├── LICENSE               # MIT License
└── README.md             # This file
```

## Running locally & troubleshooting

- Run `python main.py` and check the server logs for helpful messages. If a required env var is missing, the app prints an error and exits.
- If the Gemma client hits quota errors you will see `Retry-After` headers or exceptions and the app implements a retry/backoff using `extract_retry_delay_seconds`.

## Acknowledgements & License

This project is MIT licensed.