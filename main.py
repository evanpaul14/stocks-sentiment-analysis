from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_sqlalchemy import SQLAlchemy
from google import genai
from pygooglenews import GoogleNews
from yahooquery import search
import yfinance as yf
import re
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from dotenv import load_dotenv
import os
import logging
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import time
from collections import deque
from functools import lru_cache
import cloudscraper
from openai import OpenAI
import finnhub

load_dotenv()


log_level_name = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, log_level_name, logging.INFO))
summary_logger = logging.getLogger("stocktwits.summary")


app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///ip_log.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


def _utcnow_naive():
    """Return a timezone-naive UTC datetime without using deprecated utcnow."""
    return datetime.now(timezone.utc).replace(tzinfo=None)

# IP logging model
class IPLog(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    ip_address = db.Column(db.String(64), unique=True, nullable=False)
    first_seen = db.Column(db.DateTime, nullable=False, default=_utcnow_naive)

with app.app_context():
    db.create_all()

_last_ip_cleanup_run = None
IP_CLEANUP_INTERVAL = timedelta(days=1)

@app.before_request
def log_ip_address():
    global _last_ip_cleanup_run

    # run cleanup once per day
    now = _utcnow_naive()
    if _last_ip_cleanup_run is None or (now - _last_ip_cleanup_run) >= IP_CLEANUP_INTERVAL:
        cutoff = now - timedelta(days=30)
        IPLog.query.filter(IPLog.first_seen < cutoff).delete()
        db.session.commit()
        _last_ip_cleanup_run = now

    # determine client ip
    raw_ip = request.headers.get('X-Forwarded-For', request.remote_addr)
    if not raw_ip:
        return

    # take first IP if multiple
    ip = raw_ip.split(',')[0].strip()

    existing = IPLog.query.filter_by(ip_address=ip).first()
    if not existing:
        log = IPLog()
        log.ip_address = ip
        log.first_seen = _utcnow_naive()
        db.session.add(log)
        try:
            db.session.commit()
        except Exception as e:
            print(f"Error committing IPLog: {e}")
            db.session.rollback()


# set up rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"] 
)

# Initialize Google Gemini client
apikey = os.getenv('GOOGLE_API_KEY')
if not apikey:
    raise ValueError("Secret key not set in environment!")
client = genai.Client(api_key=apikey)
gemma_model = "gemma-3-27b-it"

LLM7_API_KEY = os.getenv("LLM7_API_KEY")
LLM7_BASE_URL = os.getenv("LLM7_BASE_URL", "https://api.llm7.io/v1")
LLM7_MODEL = os.getenv("LLM7_MODEL", "fast")
llm7_client = None
if LLM7_API_KEY:
    try:
        llm7_client = OpenAI(
            base_url=LLM7_BASE_URL,
            api_key=LLM7_API_KEY
        )
    except Exception as e:
        print(f"Error initializing LLM7 client: {e}")

FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY")
finnhub_client = None
if FINNHUB_API_KEY:
    try:
        finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)
    except Exception as e:
        print(f"Error initializing Finnhub client: {e}")

# Trending stocks API
import requests
import html
APEWISDOM_API_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
ALPACA_MOST_ACTIVE_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=20"


def _get_int_env(var_name, default):
    value = os.getenv(var_name)
    if value is None:
        return default
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


AI_RATE_LIMIT_PER_MINUTE = max(0, _get_int_env("GEMMA_MAX_CALLS_PER_MINUTE", 45))
AI_RATE_WINDOW_SECONDS = max(1, _get_int_env("GEMMA_RATE_WINDOW_SECONDS", 60))
_ai_call_timestamps = deque()


def wait_for_ai_rate_slot():
    """Respect a simple in-process rate limit for Gemma API calls."""
    if AI_RATE_LIMIT_PER_MINUTE == 0:
        return

    while True:
        now = time.time()
        window_floor = now - AI_RATE_WINDOW_SECONDS
        while _ai_call_timestamps and _ai_call_timestamps[0] < window_floor:
            _ai_call_timestamps.popleft()

        if len(_ai_call_timestamps) < AI_RATE_LIMIT_PER_MINUTE:
            _ai_call_timestamps.append(now)
            return

        wait_for = AI_RATE_WINDOW_SECONDS - (now - _ai_call_timestamps[0])
        if wait_for <= 0:
            _ai_call_timestamps.popleft()
            continue
        time.sleep(wait_for)


PRICE_CACHE_TTL_SECONDS = max(5, _get_int_env("TRENDING_PRICE_TTL_SECONDS", 120))
_price_snapshot_cache = {}

STOCKTWITS_SUMMARY_CACHE_TTL_SECONDS = max(
    60,
    _get_int_env("STOCKTWITS_SUMMARY_CACHE_TTL_SECONDS", 600)
)
_stocktwits_summary_cache = {}


def _extract_stocktwits_summary_parts(symbol_payload):
    """Return (summary_text, summary_meta, company_name) from a StockTwits payload."""
    trends = symbol_payload.get("trends") or {}
    raw_summary = (
        trends.get("summary")
        or symbol_payload.get("summary")
        or symbol_payload.get("watchlist_description")
        or symbol_payload.get("body")
        or ""
    )
    summary_meta = {
        "source": "stocktwits",
        "summary_at": trends.get("summary_at") or symbol_payload.get("summary_at")
    }
    company_name = symbol_payload.get("title") or symbol_payload.get("symbol") or "Unknown"
    return raw_summary, summary_meta, company_name


def _normalize_stocktwits_summary(summary_payload):
    """Coerce StockTwits summary payloads (string/dict/list) into clean text."""
    if summary_payload is None:
        return None

    if isinstance(summary_payload, str):
        normalized = summary_payload.strip()
        return normalized or None

    if isinstance(summary_payload, dict):
        for key in ("text", "body", "summary", "content"):
            value = summary_payload.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return None

    if isinstance(summary_payload, (list, tuple)):
        parts = []
        for part in summary_payload:
            normalized = _normalize_stocktwits_summary(part)
            if normalized:
                parts.append(normalized)
        if parts:
            joined = " ".join(parts).strip()
            return joined or None
        return None

    try:
        normalized = str(summary_payload).strip()
    except Exception:
        return None
    return normalized or None


def get_price_change_snapshot(symbol):
    if not symbol:
        return {"price": None, "change_percent": None}

    normalized = symbol.upper()
    now = time.time()
    cached = _price_snapshot_cache.get(normalized)
    if cached and (now - cached["timestamp"]) < PRICE_CACHE_TTL_SECONDS:
        return cached["data"]

    current_price = None
    previous_close = None

    try:
        ticker = yf.Ticker(normalized)
        fast_info = getattr(ticker, "fast_info", None) or {}
        current_price = (
            fast_info.get("last_price")
            or fast_info.get("lastPrice")
            or fast_info.get("regular_market_price")
            or fast_info.get("regularMarketPrice")
        )
        previous_close = (
            fast_info.get("previous_close")
            or fast_info.get("previousClose")
            or fast_info.get("regular_market_previous_close")
            or fast_info.get("regularMarketPreviousClose")
        )

        if current_price is None or previous_close is None:
            info = ticker.info or {}
            if current_price is None:
                current_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if previous_close is None:
                previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
    except Exception as exc:
        print(f"Error fetching realtime price for {normalized}: {exc}")

    change_percent = None
    if current_price is not None and previous_close not in (None, 0):
        try:
            change_percent = ((current_price - previous_close) / previous_close) * 100
        except Exception:
            change_percent = None

    snapshot = {"price": current_price, "change_percent": change_percent}
    _price_snapshot_cache[normalized] = {"timestamp": now, "data": snapshot}
    return snapshot


def _parse_retry_after_value(value):
    try:
        seconds = float(value)
        return max(0.0, seconds)
    except (TypeError, ValueError):
        pass

    try:
        retry_dt = parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None

    if retry_dt is None:
        return None

    if retry_dt.tzinfo is None:
        retry_dt = retry_dt.replace(tzinfo=timezone.utc)

    delta = (retry_dt - datetime.now(timezone.utc)).total_seconds()
    return max(0.0, delta)


def extract_retry_delay_seconds(exc):
    """Extract retry delay from a Gemma API exception, if provided."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if isinstance(headers, dict):
        retry_after = headers.get("Retry-After") or headers.get("retry-after")
        if retry_after:
            parsed = _parse_retry_after_value(retry_after)
            if parsed is not None:
                return parsed

    message = str(exc).lower()
    match = re.search(r"(\d+(?:\.\d+)?)\s*(?:seconds|secs|s)", message)
    if match:
        try:
            return max(0.0, float(match.group(1)))
        except ValueError:
            return None

    return None


def fetch_top_stocks():
    """Fetch top trending Reddit stocks from ApeWisdom API"""
    try:
        resp = requests.get(APEWISDOM_API_URL, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        stocks_list = data.get("results", [])
        top20 = sorted(stocks_list, key=lambda x: x.get("rank", 999))[:20]
        return top20
    except Exception as e:
        print(f"Error fetching trending stocks: {e}")
        return []

def analyze_trending(top10):
    """Analyze trending stocks and apply tags based on percent increase and rank changes"""
    results = []
    for rec in top10:
        ticker = rec.get("ticker")
        if not ticker:
            continue
        name   = html.unescape(rec.get("name", ""))
        mentions_now = rec.get("mentions", 0)
        mentions_24h = rec.get("mentions_24h_ago", 0)
        rank_24h     = rec.get("rank_24h_ago", None)
        rank_now     = rec.get("rank", None)

        # percent increase
        if mentions_24h > 0:
            pct_increase = ((mentions_now - mentions_24h) / mentions_24h) * 100
        else:
            pct_increase = 0

        tag = ""
        if mentions_24h > 0 and pct_increase > 50:
            tag = "Trending"
        elif rank_24h is not None and rank_now is not None and rank_now < rank_24h - 5:
            tag = f"Up {rank_24h - rank_now} Spots"

        price_snapshot = get_price_change_snapshot(ticker)
        results.append({
            "ticker": ticker,
            "name": name,
            "pct_increase": pct_increase,
            "tag": tag,
            "price_change_percent": price_snapshot.get("change_percent"),
            "price": price_snapshot.get("price")
        })
    return results


@lru_cache(maxsize=256)
def lookup_company_name(symbol):
    """Resolve ticker to company name via yfinance"""
    if not symbol:
        return None
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}
        return info.get("longName") or info.get("shortName") or info.get("shortName")
    except Exception as e:
        print(f"Error looking up company name for {symbol}: {e}")
        return None


def _cache_stocktwits_summary(symbol, name, summary_text, summary_meta, watchlist_count=None):
    normalized = (symbol or "").upper()
    if not normalized:
        return
    normalized_summary = _normalize_stocktwits_summary(summary_text)
    _stocktwits_summary_cache[normalized] = {
        "ticker": normalized,
        "name": name,
        "summary": normalized_summary,
        "summary_meta": summary_meta or {},
        "watchlist_count": watchlist_count,
        "timestamp": time.time()
    }
    return normalized_summary


def get_stocktwits_summary_entry(symbol):
    normalized = (symbol or "").upper()
    if not normalized:
        return None

    cached = _stocktwits_summary_cache.get(normalized)
    now = time.time()
    if cached and (now - cached.get("timestamp", 0)) < STOCKTWITS_SUMMARY_CACHE_TTL_SECONDS:
        summary_logger.debug(
            "[StockTwits] Serving cached summary for %s (age=%.1fs)",
            normalized,
            now - cached.get("timestamp", 0)
        )
        return cached

    summary_logger.info(
        "[StockTwits] Refreshing summary cache for %s (stale or missing)",
        normalized
    )
    return _refresh_stocktwits_summary(normalized)


def _fetch_stocktwits_symbols(limit=20):
    scraper = cloudscraper.create_scraper(
        browser={
            "browser": "chrome",
            "platform": "darwin",
            "mobile": False
        }
    )
    resp = scraper.get(STOCKTWITS_TRENDING_URL, timeout=10)
    resp.raise_for_status()
    payload = resp.json()
    symbols = payload.get("symbols", [])

    filtered = [
        s for s in symbols
        if (s.get("exchange") or "").upper() != "CRYPTO"
        and (s.get("instrument_class") or "").lower() == "stock"
    ]
    return filtered[:limit]


def _refresh_stocktwits_summary(symbol, *, limit=60):
    normalized = (symbol or "").upper()
    if not normalized:
        return None

    try:
        summary_logger.info(
            "[StockTwits] On-demand summary fetch for %s (limit=%d)",
            normalized,
            limit
        )
        symbols = _fetch_stocktwits_symbols(limit=limit)
    except Exception as exc:
        summary_logger.exception(
            "[StockTwits] Error refreshing summary for %s: %s",
            normalized,
            exc
        )
        return None

    for sym in symbols:
        current_symbol = (sym.get("symbol") or "").upper()
        if current_symbol != normalized:
            continue
        raw_summary, summary_meta, company_name = _extract_stocktwits_summary_parts(sym)
        _cache_stocktwits_summary(
            normalized,
            company_name,
            raw_summary,
            summary_meta,
            sym.get("watchlist_count")
        )
        return _stocktwits_summary_cache.get(normalized)

    summary_logger.warning(
        "[StockTwits] Summary refresh miss for %s (not in payload)",
        normalized
    )
    return None


def fetch_stocktwits_trending(limit=20):
    """Fetch trending symbols from StockTwits"""
    try:
        trimmed = _fetch_stocktwits_symbols(limit=limit)
        results = []
        for idx, sym in enumerate(trimmed, start=1):
            price_info = sym.get("price")
            if isinstance(price_info, dict):
                last_price = price_info.get("last")
                change_pct = price_info.get("change_percent")
            else:
                last_price = sym.get("price")
                change_pct = sym.get("change_percent")

            symbol = (sym.get("symbol") or "").upper()
            resolved_price = last_price
            resolved_change_pct = change_pct
            if resolved_price is None or resolved_change_pct is None:
                snapshot = get_price_change_snapshot(symbol)
                if resolved_price is None:
                    resolved_price = snapshot.get("price")
                if resolved_change_pct is None:
                    resolved_change_pct = snapshot.get("change_percent")

            raw_summary, summary_meta, company_name = _extract_stocktwits_summary_parts(sym)
            cached_summary = None
            has_summary = False
            if idx <= 10:
                cached_summary = _cache_stocktwits_summary(
                    symbol,
                    company_name,
                    raw_summary,
                    summary_meta,
                    sym.get("watchlist_count")
                )
                has_summary = bool(cached_summary)

            results.append({
                "rank": idx,
                "ticker": symbol,
                "name": company_name,
                "watchlist_count": sym.get("watchlist_count"),
                "change_percent": resolved_change_pct,
                "price": resolved_price,
                "price_change_percent": resolved_change_pct,
                "has_summary": has_summary
            })
        return results
    except Exception:
        return []


def fetch_alpaca_most_actives(limit=20):
    """Fetch most active stocks by volume from Alpaca"""
    headers = {}
    alpaca_key = os.getenv("ALPACA_API_KEY_ID") or os.getenv("ALPACA_API_KEY")
    alpaca_secret = os.getenv("ALPACA_API_SECRET_KEY") or os.getenv("ALPACA_SECRET_KEY")
    if alpaca_key and alpaca_secret:
        headers["Apca-Api-Key-Id"] = alpaca_key
        headers["Apca-Api-Secret-Key"] = alpaca_secret

    try:
        resp = requests.get(ALPACA_MOST_ACTIVE_URL, headers=headers, timeout=10)
        resp.raise_for_status()
        payload = resp.json()
        stocks = (
            payload.get("most_actives")
            or payload.get("stocks")
            or payload.get("results")
            or payload.get("data")
            or []
        )

        trimmed = stocks[:limit]
        results = []
        for idx, rec in enumerate(trimmed, start=1):
            change_pct = rec.get("change_percent") or rec.get("percent_change")
            price = rec.get("price") or rec.get("last") or rec.get("close")
            volume = rec.get("volume") or rec.get("volume_1d")
            symbol = (rec.get("symbol") or "").upper()
            resolved_name = rec.get("name") or lookup_company_name(symbol) or symbol or "Unknown"
            if price is None or change_pct is None:
                snapshot = get_price_change_snapshot(symbol)
                if price is None:
                    price = snapshot.get("price")
                if change_pct is None:
                    change_pct = snapshot.get("change_percent")
            results.append({
                "rank": idx,
                "ticker": symbol,
                "name": resolved_name,
                "volume": volume,
                "price": price,
                "change_percent": change_pct,
                "price_change_percent": change_pct
            })
        return results
    except Exception as e:
        print(f"Error fetching Alpaca most actives: {e}")
        return []


def get_trending_source_data(source):
    """Return trending data for a specific source identifier."""
    normalized = (source or "").lower()

    if normalized == "stocktwits":
        return fetch_stocktwits_trending()
    if normalized == "reddit":
        try:
            return analyze_trending(fetch_top_stocks())
        except Exception as e:
            print(f"Error in reddit trending: {e}")
            return []
    if normalized == "volume":
        return fetch_alpaca_most_actives()

    return None


def get_stock_info(symbol):
    """Get stock information using yfinance"""
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info

        # Get current price data
        current_price = info.get('currentPrice') or info.get('regularMarketPrice', 0)
        previous_close = info.get('previousClose', 0)

        # Calculate change
        change = current_price - previous_close if previous_close else 0
        change_percent = (change / previous_close * 100) if previous_close else 0

        # Extract the year founded from the business summary using regex
        description = info.get('longBusinessSummary', '')
        year_founded = 'N/A'
        if description:
            match = re.search(r"in (\b(?:19|20)\d{2}\b)", description)
            if match:
                year_founded = int(match.group(1))
        #Get company officers to find CEO
        ceo_name = 'N/A'
        officers = info.get('companyOfficers', [])
        for officer in officers:
            title = officer.get('title', '').lower()
            if 'ceo' in title:
                ceo_name = officer.get('name', 'N/A')
                break
        return {
            'symbol': symbol,
            'price': current_price,
            'change': change,
            'changePercent': change_percent,
            'afterHoursPrice': info.get('postMarketPrice'),
            'afterHoursChange': info.get('postMarketChange'),
            'afterHoursChangePercent': info.get('postMarketChangePercent'),
            'marketCap': info.get('marketCap', 0),
            'peRatio': info.get('forwardPE') or info.get('trailingPE', 0),
            'dividendYield': info.get('dividendYield', 0) if info.get('dividendYield') else 0,
            'avgVolume': info.get('averageDailyVolume10Day', 0),
            'dayHigh': info.get('dayHigh', 0),
            'dayLow': info.get('dayLow', 0),
            'openPrice': info.get('open') or info.get('regularMarketOpen', 0),
            'volume': info.get('volume', 0),
            'fiftyTwoWeekHigh': info.get('fiftyTwoWeekHigh', 0),
            'fiftyTwoWeekLow': info.get('fiftyTwoWeekLow', 0),
            'companyName': info.get('longName') or info.get('shortName', symbol),
            'ceo': ceo_name,
            'employees': info.get('fullTimeEmployees'),
            'city': info.get('city'),
            'state': info.get('state'),
            'country': info.get('country'),
            'industry': info.get('industry'),
            'sector': info.get('sector'),
            'website': info.get('website'),
            'description': description,
            'yearFounded': year_founded
        }
    except Exception as e:
        print(f"Error getting stock info: {e}")
        return None


def get_historical_data(symbol, period='1d'):
    """Get historical price data using yfinance"""
    try:
        ticker = yf.Ticker(symbol)

        if period == '1d':
            hist = ticker.history(period='5d', interval='5m')
            if not hist.empty:
                last_date = hist.index[-1].date()
                hist = hist[hist.index.date == last_date]

        elif period in ['7d', '1w']:
            end = datetime.now()
            start = end - timedelta(days=7)
            hist = yf.download(
                symbol,
                start=start,
                end=end,
                interval='1h',
                prepost=True,
                progress=False
            )

        else:
            interval = None
            if period == '1mo':
                interval = '90m'

            if interval:
                hist = ticker.history(period=period, interval=interval)
            else:
                hist = ticker.history(period=period)

        if hist.empty:
            return []

        data = [
            {"date": idx.strftime("%Y-%m-%d %H:%M:%S"), "price": float(row["Close"])}
            for idx, row in hist.iterrows()
        ]
        return data

    except Exception as e:
        print(f"Error getting historical data: {e}")
        return []


def get_news_articles(stock_symbol, num_articles=10):
    """Get news articles using Google News"""
    try:
        gn = GoogleNews(lang='en', country='US')
        search_result = gn.search(stock_symbol + " stock")
        entries = search_result.get('entries', [])

        articles = []
        for entry in entries[:num_articles]:
            articles.append({
                'title': entry.get('title', 'No title'),
                'description': entry.get('summary', ''),
                'link': entry.get('link', ''),
                'publishedAt': entry.get('published', ''),
                'source': entry.get('source', {}).get('title', 'Unknown')
            })
        return articles
    except Exception as e:
        print(f"Error getting news: {e}")
        return []


def fetch_finnhub_company_news(symbol, max_articles=6, lookback_days=5):
    if not finnhub_client or not symbol:
        return []

    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=lookback_days)

    try:
        raw_articles = finnhub_client.company_news(
            symbol,
            _from=start_date.isoformat(),
            to=end_date.isoformat()
        ) or []
    except Exception as e:
        print(f"Error fetching Finnhub news for {symbol}: {e}")
        return []

    sanitized = []
    seen = set()
    for article in raw_articles:
        ts = article.get("datetime")
        iso_time = None
        if isinstance(ts, (int, float)):
            try:
                iso_time = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
            except Exception:
                iso_time = None

        headline = (article.get("headline") or "").strip()
        url = (article.get("url") or "").strip()
        if not headline:
            continue
        dedupe_key = (headline.lower(), url.lower())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)

        sanitized.append({
            "headline": headline,
            "summary": article.get("summary", ""),
            "source": article.get("source", ""),
            "url": url,
            "publishedAt": iso_time,
            "datetime": ts
        })

    sanitized.sort(key=lambda item: item.get("datetime") or 0, reverse=True)
    trimmed = [s for s in sanitized if s["headline"]][:max_articles]
    return trimmed


def convert_google_news_for_movement(symbol, max_articles=6):
    google_articles = get_news_articles(symbol, max_articles)
    normalized = []
    for entry in google_articles:
        headline = (entry.get('title') or '').strip()
        if not headline:
            continue
        normalized.append({
            "headline": headline,
            "summary": entry.get('description', ''),
            "source": entry.get('source', 'Google News'),
            "url": entry.get('link'),
            "publishedAt": entry.get('publishedAt') or entry.get('published')
        })
    return normalized


def summarize_stock_movement(symbol, company_name, change_percent, articles):
    if not llm7_client or not articles:
        return None

    bullet_lines = []
    for idx, article in enumerate(articles, start=1):
        published = article.get("publishedAt") or "Unknown time"
        summary = article.get("summary") or ""
        bullet_lines.append(
            f"{idx}. {article.get('headline')} ({article.get('source')}) on {published}: {summary}"
        )

    change_direction = "up" if change_percent >= 0 else "down"
    change_abs = abs(change_percent)
    prompt = (
        f"Company: {company_name} ({symbol})\n"
        f"Intraday move: {change_direction} {change_abs:.2f}% today.\n\n"
        "Recent catalysts:\n"
        + "\n".join(bullet_lines)
        + "\n\n"
          "Write a concise 2-3 sentence summary explaining the most likely reasons for the move. "
          "Blend the headlines into a cohesive narrative and mention key drivers."
          "Try to incorporate quantitative details where possible."
          "If the articles do not explain the move, clearly state that."
          "Avoid speculation and do not invent any facts."
          "Do not include a headline or summary section indicators, just give the summary."
          "Do not specifically mention the stock price or percentage change in the summary."
          "Do not mention any specific articles."
          "Refer to the company by its ticker symbol rather than by name"
          "Do not directly refer to what the articles say or do not say."
    )

    try:
        response = llm7_client.chat.completions.create(
            model=LLM7_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "You are a sharp markets reporter who explains price action using headlines."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,
            max_tokens=220
        )
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"Error generating movement summary: {e}")
        return None


def fallback_movement_summary(company_name, change_percent, articles, source_label):
    direction = "higher" if change_percent >= 0 else "lower"
    base = f"{company_name} is trading {direction} by roughly {abs(change_percent):.2f}% today."

    if articles:
        highlights = [article.get('headline') for article in articles[:3] if article.get('headline')]
        if highlights:
            joined = "; ".join(highlights)
            return f"{base} Recent {source_label} headlines include {joined}."

    return f"{base} No fresh {source_label} headlines surfaced yet, so keep an eye on upcoming catalysts or filings."


def build_movement_insight(stock_info):
    change_percent = stock_info.get('changePercent')
    if change_percent is None or abs(change_percent) < 3:
        return None

    symbol = stock_info.get('symbol')
    company_name = stock_info.get('companyName') or symbol
    articles = fetch_finnhub_company_news(symbol)
    article_source_label = "Finnhub"
    if not articles:
        articles = convert_google_news_for_movement(symbol)
        article_source_label = "news"

    summary = summarize_stock_movement(symbol, company_name, change_percent, articles)
    if not summary:
        summary = fallback_movement_summary(company_name, change_percent, articles, article_source_label)
    if not summary:
        summary = f"{company_name} is moving sharply today, but no catalysts were found."

    sources = []
    for article in articles:
        if not article.get('headline'):
            continue
        sources.append({
            "headline": article.get('headline'),
            "source": article.get('source'),
            "url": article.get('url'),
            "publishedAt": article.get('publishedAt')
        })

    return {
        "summary": summary,
        "changePercent": change_percent,
        "sources": sources
    }


def analyze_sentiment_gemma(article_title, article_description, company_name):
    """Analyze sentiment using Google Gemma AI"""
    prompt = (
        f"Analyze the sentiment (positive, negative, or neutral) of this news article strictly in reference "
        f"to the company {company_name}.\n\n"
        f"Title: {article_title}\n"
        f"Description: {article_description}\n\n"
        f"Do not assume anything not explicitly stated in the title or description.\n"
        f"Respond with only one word and nothing else: positive, negative, or neutral."
    )

    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            wait_for_ai_rate_slot()
            response = client.models.generate_content(
                model=gemma_model,
                contents=prompt
            )

            sentiment = response.text.strip().lower()
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'
            return sentiment
        except Exception as e:
            retry_delay = extract_retry_delay_seconds(e)
            if retry_delay:
                print(f"Gemma quota hit, waiting {retry_delay:.2f}s before retrying...")
                time.sleep(retry_delay)
                continue
            print(f"Error analyzing sentiment: {e}")
            break

    return 'neutral'


def build_sentiment_payload(symbol, company_name=None):
    articles = get_news_articles(symbol, 10)
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
    for article in articles:
        sentiment = analyze_sentiment_gemma(
            article['title'],
            article['description'],
            company_name or symbol
        )
        article['sentiment'] = sentiment
        if sentiment in sentiment_summary:
            sentiment_summary[sentiment] += 1

    if all(value == 0 for value in sentiment_summary.values()):
        overall_sentiment = 'neutral'
    else:
        overall_sentiment = max(sentiment_summary, key=sentiment_summary.get)

    return articles, sentiment_summary, overall_sentiment


@app.route('/')
@limiter.limit("50 per minute")
def index():
    '''Render the main page'''
    return render_template('index.html', page_view='home', initial_query='')


@app.route('/watchlist')
@limiter.limit("50 per minute")
def watchlist_page():
    """Render the watchlist page"""
    return render_template('index.html', page_view='watchlist', initial_query='')


@app.route('/results')
@limiter.limit("50 per minute")
def search_results_page():
    """Render the dedicated search results page"""
    query = (request.args.get('q') or '').strip()
    return render_template('index.html', page_view='search', initial_query=query)


@app.route('/trending-list')
@limiter.limit("50 per minute")
def trending_board_page():
    """Render the dedicated trending dashboards page"""
    return render_template('index.html', page_view='trending', initial_query='')

@app.route('/search', methods=['POST'])
@limiter.limit("10 per minute")
def search_stock():
    '''Search for stock and return info, historical data, news, and sentiment analysis'''
    try:
        company_name = request.json.get('company_name') if request.json else None
        if not company_name:
            return jsonify({'error': 'Company name is required'}), 400

        # Search for stock symbol
        results = search(company_name)
        if not results.get('quotes') or len(results['quotes']) == 0:
            return jsonify({'error': 'Company not found'}), 404

        stock_symbol = results['quotes'][0]['symbol'].upper()

        # Get stock information
        stock_info = get_stock_info(stock_symbol)
        if not stock_info:
            return jsonify({'error': 'Failed to retrieve stock information'}), 500

        # Get historical data for 1 day
        historical_data = get_historical_data(stock_symbol, '1d')

        response_data = {
            'stock_info': stock_info,
            'historical_data': historical_data
        }

        articles, sentiment_summary, overall_sentiment = build_sentiment_payload(
            stock_symbol,
            stock_info['companyName']
        )
        response_data['articles'] = articles
        response_data['sentiment_summary'] = sentiment_summary
        response_data['overall_sentiment'] = overall_sentiment

        movement_insight = build_movement_insight(stock_info)
        if movement_insight:
            response_data['movement_insight'] = movement_insight

        return jsonify(response_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Error in search: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/historical/<symbol>/<period>', methods=['GET'])
@limiter.limit("50 per minute")
def get_historical(symbol, period):
    """Get historical data for a specific period"""
    try:
        data = get_historical_data(symbol, period)
        return jsonify({'data': data})
    except Exception as e:
        print(f"Error getting historical data: {e}")
        return jsonify({'error': str(e)}), 500


# Trending stocks API endpoint
@app.route('/trending', methods=['GET'])
@limiter.limit("30 per minute")
def trending_stocks():
    return jsonify({
        "stocktwits": get_trending_source_data("stocktwits") or [],
        "reddit": get_trending_source_data("reddit") or [],
        "volume": get_trending_source_data("volume") or []
    })


@app.route('/quote/<symbol>', methods=['GET'])
@limiter.limit("50 per minute")
def quote(symbol):
    normalized = (symbol or "").strip().upper()
    if not normalized:
        return jsonify({"error": "Symbol is required"}), 400

    snapshot = get_price_change_snapshot(normalized)
    price = snapshot.get("price")
    if price is None:
        return jsonify({"error": "Price unavailable"}), 404

    return jsonify({
        "symbol": normalized,
        "price": price,
        "change_percent": snapshot.get("change_percent"),
        "timestamp": datetime.now(timezone.utc).isoformat()
    })


@app.route('/trending/<source>', methods=['GET'])
@limiter.limit("30 per minute")
def trending_stocks_source(source):
    data = get_trending_source_data(source)
    if data is None:
        return jsonify({"error": "Unknown trending source"}), 404
    return jsonify({
        "source": (source or "").lower(),
        "data": data
    })


@app.route('/stocktwits/<symbol>/summary', methods=['GET'])
@limiter.limit("30 per minute")
def stocktwits_summary(symbol):
    normalized = (symbol or "").upper()
    entry = get_stocktwits_summary_entry(normalized)
    if not entry:
        summary_logger.warning(
            "[StockTwits] Summary endpoint miss for %s (no cache entry)",
            normalized
        )
        return jsonify({"error": "Summary unavailable"}), 404
    summary_text = (entry.get("summary") or "").strip()
    used_fallback = False
    if not summary_text:
        summary_text = "No StockTwits summary available yet."
        used_fallback = True
    summary_logger.info(
        "[StockTwits] Served summary for %s (fallback=%s)",
        normalized,
        used_fallback
    )
    return jsonify({
        "ticker": entry.get("ticker"),
        "name": entry.get("name"),
        "summary": summary_text,
        "summary_meta": entry.get("summary_meta"),
        "watchlist_count": entry.get("watchlist_count")
    })

@app.route('/robots.txt', methods=['GET'])
@limiter.limit("100 per minute")
def robots_txt():
    '''Serve robots.txt file'''
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "robots.txt")

@app.route('/sitemap.xml', methods=['GET'])
@limiter.limit("100 per minute")
def sitemap_xml():
    '''Serve sitemap.xml file'''
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "sitemap.xml")

@app.route('/google-verification.html', methods=['GET'])
@limiter.limit("100 per minute")
def google_verification():
    '''Serve google verification file'''
    return send_from_directory(os.path.dirname(os.path.abspath(__file__)), "google-verification.html")



if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        debug=False, 
    )