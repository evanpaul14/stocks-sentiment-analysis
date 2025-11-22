# Copilot Instructions for `stocks-sentiment-analysis`

## Project Architecture

- **Backend:** `main.py` (Flask app) is the entry point. It handles all API endpoints, sentiment analysis, trending stock aggregation, and background cache management.
- **Background Worker:** Defined in `worker.py`, started via `start_worker_if_enabled()` (called on Flask startup and in `__main__`). It maintains a sentiment cache in SQLite, refreshing hourly and pruning old entries.
- **Frontend:** `templates/index.html` (Jinja2/HTML) and static assets in `static/`.
- **Data Sources:** Integrates with yfinance, Google News, ApeWisdom, StockTwits, Alpaca, and Google Gemma AI for data and sentiment.
- **Database:** Uses SQLite (`sentiment_cache.db`) for caching sentiment summaries.

## Deployment & Environment

- **Production:** Must work on Ubuntu VPS with Gunicorn and systemctl. The background worker must start reliably in Gunicorn worker processes (see `@app.before_first_request` in `main.py`).
- **Environment Variables:** All API keys and model configs are loaded from `.env` (see README for required keys).
- **Disabling Worker:** Set `DISABLE_BACKGROUND_WORKER=1` to skip background tasks (important for tests and some deployments).

## Key Patterns & Conventions

- **Background Tasks:** Use `start_worker_if_enabled()` and never block the main thread. Always ensure Flask and Gunicorn can start/stop cleanly.
- **Rate Limiting:** All endpoints are rate-limited via Flask-Limiter.
- **Sentiment Analysis:** Direct user queries use Google Gemma AI; background cache uses LLM7 API for higher throughput.
- **Trending Aggregation:** Trending tickers are collected from multiple sources and merged.
- **Testing:** Use `pytest`. The test suite expects the worker to be disabled (`DISABLE_BACKGROUND_WORKER=1`).
- **Error Handling:** Log and gracefully handle all external API failures; never crash the server on third-party errors.

## Examples

- To add a new background task, extend `worker.py` and wire it via the dependency injection pattern used in `WorkerDependencies`.
- To add a new API endpoint, define it in `main.py` and document its rate limit and usage in the README.

## Do/Don't

- **Do**: Ensure all code is compatible with Gunicorn/systemctl on Ubuntu VPS.
- **Do**: Use dependency injection for background worker logic.
- **Do**: Keep all API keys and secrets out of source control.
- **Don't**: Use patterns that require a persistent main thread or block Gunicorn workers.
- **Don't**: Assume a specific Flask server (must work with Gunicorn).
