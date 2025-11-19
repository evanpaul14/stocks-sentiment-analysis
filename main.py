from flask import Flask, render_template, request, jsonify, send_from_directory
from google import genai
from pygooglenews import GoogleNews
from yahooquery import search
import yfinance as yf
import json
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import sqlite3
import threading
import time
from collections import deque
from functools import lru_cache
import cloudscraper
from openai import OpenAI

load_dotenv()
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sentiment_cache.db")
BACKGROUND_REFRESH_SECONDS = 60 * 60  # 1 hour
CACHE_HEALTH_CHECK_SECONDS = 10 * 60  # 10 minutes
AI_RATE_LIMIT_PER_MINUTE = 50
AI_RATE_WINDOW_SECONDS = 60
STATIC_SYMBOLS = {
    "GOOGL",  # Alphabet Class A
    "GOOG",   # Alphabet Class C
    "AMZN",   # Amazon
    "AAPL",   # Apple
    "META",   # Meta
    "MSFT",   # Microsoft
    "NVDA",   # Nvidia
    "TSLA",   # Tesla
    "SOFI"    # SoFi
}
tracked_trending_symbols = set()
db_lock = threading.Lock()
background_thread_started = False
ai_rate_lock = threading.Lock()
ai_rate_timestamps = deque()
llm7_client = None
llm7_client_lock = threading.Lock()
LLM7_BASE_URL = os.getenv("LLM7_BASE_URL", "https://api.llm7.io/v1")
LLM7_MODEL = os.getenv("LLM7_MODEL_NAME", "gpt-4.1-nano-2025-04-14")

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

# Trending stocks API
import requests
import html
APEWISDOM_API_URL = "https://apewisdom.io/api/v1.0/filter/all-stocks"
STOCKTWITS_TRENDING_URL = "https://api.stocktwits.com/api/2/trending/symbols.json"
ALPACA_MOST_ACTIVE_URL = "https://data.alpaca.markets/v1beta1/screener/stocks/most-actives?by=volume&top=20"


def get_db_connection():
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return conn


def init_db():
    with db_lock:
        conn = get_db_connection()
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS sentiment_cache (
                symbol TEXT PRIMARY KEY,
                articles TEXT NOT NULL,
                sentiment_summary TEXT NOT NULL,
                overall_sentiment TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        conn.close()


def wait_for_ai_rate_slot():
    while True:
        with ai_rate_lock:
            now = time.time()
            while ai_rate_timestamps and now - ai_rate_timestamps[0] >= AI_RATE_WINDOW_SECONDS:
                ai_rate_timestamps.popleft()

            if len(ai_rate_timestamps) < AI_RATE_LIMIT_PER_MINUTE:
                ai_rate_timestamps.append(now)
                return

            wait_seconds = AI_RATE_WINDOW_SECONDS - (now - ai_rate_timestamps[0])

        time.sleep(max(wait_seconds, 0.1))


def extract_retry_delay_seconds(error):
    message = str(error)
    patterns = [
        r"retryDelay['\"]?:\s*'?(?P<delay>\d+(?:\.\d+)?)s",
        r"Please retry in (?P<delay>\d+(?:\.\d+)?)s"
    ]
    for pattern in patterns:
        match = re.search(pattern, message)
        if match:
            try:
                return float(match.group('delay'))
            except (ValueError, TypeError):
                continue
    return None


def save_sentiment_to_db(symbol, articles, summary, overall_sentiment):
    payload = (
        symbol.upper(),
        json.dumps(articles),
        json.dumps(summary),
        overall_sentiment,
        datetime.utcnow().isoformat()
    )
    with db_lock:
        conn = get_db_connection()
        conn.execute(
            """
            INSERT INTO sentiment_cache (symbol, articles, sentiment_summary, overall_sentiment, updated_at)
            VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(symbol) DO UPDATE SET
                articles = excluded.articles,
                sentiment_summary = excluded.sentiment_summary,
                overall_sentiment = excluded.overall_sentiment,
                updated_at = excluded.updated_at
            """,
            payload
        )
        conn.commit()
        conn.close()


def get_cached_sentiment(symbol):
    with db_lock:
        conn = get_db_connection()
        row = conn.execute(
            "SELECT articles, sentiment_summary, overall_sentiment, updated_at FROM sentiment_cache WHERE symbol = ?",
            (symbol.upper(),)
        ).fetchone()
        conn.close()
    if not row:
        return None
    return {
        "articles": json.loads(row[0]),
        "sentiment_summary": json.loads(row[1]),
        "overall_sentiment": row[2],
        "updated_at": row[3]
    }


def get_tracked_symbols():
    return STATIC_SYMBOLS.union(tracked_trending_symbols)


def get_cached_symbols():
    with db_lock:
        conn = get_db_connection()
        rows = conn.execute("SELECT symbol FROM sentiment_cache").fetchall()
        conn.close()
    return {row[0] for row in rows}


def purge_stale_sentiment_records(valid_symbols):
    if not valid_symbols:
        return
    valid = {symbol.upper() for symbol in valid_symbols}
    with db_lock:
        conn = get_db_connection()
        rows = conn.execute("SELECT symbol FROM sentiment_cache").fetchall()
        stale = [row[0] for row in rows if row[0] not in valid]
        for symbol in stale:
            conn.execute("DELETE FROM sentiment_cache WHERE symbol = ?", (symbol,))
        conn.commit()
        conn.close()

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

        results.append({
            "ticker": ticker,
            "name": name,
            "pct_increase": pct_increase,
            "tag": tag
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


def fetch_stocktwits_trending(limit=20):
    """Fetch trending symbols from StockTwits"""
    try:
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

        non_crypto = [s for s in symbols if s.get("exchange") != "CRYPTO"]
        trimmed = non_crypto[:limit]
        results = []
        for idx, sym in enumerate(trimmed, start=1):
            price_info = sym.get("price")
            if isinstance(price_info, dict):
                last_price = price_info.get("last")
                change_pct = price_info.get("change_percent")
            else:
                last_price = sym.get("price")
                change_pct = sym.get("change_percent")

            results.append({
                "rank": idx,
                "ticker": (sym.get("symbol") or "").upper(),
                "name": sym.get("title") or sym.get("symbol") or "Unknown",
                "watchlist_count": sym.get("watchlist_count"),
                "change_percent": change_pct,
                "price": last_price,
                "summary": sym.get("summary")
                    or sym.get("watchlist_description")
                    or sym.get("body")
                    or "Trending now on StockTwits"
            })
        return results
    except Exception as e:
        print(f"Error fetching StockTwits trending: {e}")
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
            results.append({
                "rank": idx,
                "ticker": symbol,
                "name": resolved_name,
                "volume": volume,
                "price": price,
                "change_percent": change_pct
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


def collect_trending_symbols_from_all_sources():
    """Aggregate tickers appearing across all configured trending sources."""
    symbol_set = set()

    def _extract_symbols(records, source_name):
        extracted = set()
        for rec in records or []:
            ticker = rec.get("ticker") or rec.get("symbol")
            if ticker:
                extracted.add(ticker.upper())
        if not extracted:
            print(f"No trending tickers extracted from {source_name} response")
        return extracted

    try:
        reddit_records = fetch_top_stocks()
        symbol_set |= _extract_symbols(reddit_records, "reddit")
    except Exception as e:
        print(f"Error collecting reddit trending symbols: {e}")

    try:
        stocktwits_records = fetch_stocktwits_trending()
        symbol_set |= _extract_symbols(stocktwits_records, "stocktwits")
    except Exception as e:
        print(f"Error collecting stocktwits trending symbols: {e}")

    try:
        volume_records = fetch_alpaca_most_actives()
        symbol_set |= _extract_symbols(volume_records, "volume")
    except Exception as e:
        print(f"Error collecting volume trending symbols: {e}")

    return symbol_set


def refresh_tracked_trending_symbols():
    """Update the global tracked_trending_symbols set with current trending tickers."""
    global tracked_trending_symbols
    latest_symbols = collect_trending_symbols_from_all_sources()
    if latest_symbols:
        tracked_trending_symbols = latest_symbols
    else:
        print("Warning: Unable to refresh trending symbols; keeping previous tracked set")


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


def analyze_sentiment_gemma(article_title, article_description, company_name):
    """Analyze sentiment using Google Gemma AI"""
    prompt = build_sentiment_prompt(article_title, article_description, company_name)

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


def build_sentiment_prompt(article_title, article_description, company_name):
    return (
        f"Analyze the sentiment (positive, negative, or neutral) of this news article strictly in reference "
        f"to the company {company_name}.\n\n"
        f"Title: {article_title}\n"
        f"Description: {article_description}\n\n"
        f"Respond with only one word and nothing else: positive, negative, or neutral."
    )


def get_llm7_client():
    global llm7_client
    if not os.getenv("LLM7_API_KEY"):
        raise ValueError("LLM7_API_KEY is not set. Unable to analyze sentiment for trending cache.")

    with llm7_client_lock:
        if llm7_client is None:
            llm7_client = OpenAI(
                base_url=LLM7_BASE_URL,
                api_key=os.getenv("LLM7_API_KEY")
            )
    return llm7_client


def analyze_sentiment_llm7(article_title, article_description, company_name):
    """Analyze sentiment for cached trending stocks using the LLM7 API"""
    prompt = build_sentiment_prompt(article_title, article_description, company_name)
    max_attempts = 3
    backoff_seconds = 1

    for attempt in range(max_attempts):
        try:
            client = get_llm7_client()
            response = client.chat.completions.create(
                model=LLM7_MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            sentiment = (response.choices[0].message.content or "").strip().lower()
            if sentiment not in ['positive', 'negative', 'neutral']:
                sentiment = 'neutral'
            return sentiment
        except Exception as e:
            if attempt == max_attempts - 1:
                print(f"LLM7 sentiment analysis failed permanently: {e}")
                break
            sleep_for = min(backoff_seconds * (2 ** attempt), 8)
            print(f"LLM7 sentiment analysis failed (attempt {attempt + 1}); retrying in {sleep_for:.1f}s: {e}")
            time.sleep(sleep_for)

    return 'neutral'


def build_sentiment_payload(symbol, company_name=None, sentiment_analyzer=None):
    analyzer = sentiment_analyzer or analyze_sentiment_gemma
    articles = get_news_articles(symbol, 10)
    sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
    for article in articles:
        sentiment = analyzer(
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


def refresh_sentiment_cache():
    refresh_tracked_trending_symbols()
    symbols_to_update = get_tracked_symbols()
    for symbol in symbols_to_update:
        if not symbol:
            continue
        try:
            stock_info = get_stock_info(symbol)
            company_name = stock_info['companyName'] if stock_info else symbol
            articles, sentiment_summary, overall_sentiment = build_sentiment_payload(
                symbol,
                company_name,
                sentiment_analyzer=analyze_sentiment_llm7
            )
            save_sentiment_to_db(symbol, articles, sentiment_summary, overall_sentiment)
        except Exception as e:
            print(f"Error refreshing sentiment for {symbol}: {e}")

    purge_stale_sentiment_records(symbols_to_update)


def ensure_cache_completeness():
    refresh_tracked_trending_symbols()
    tracked_symbols = get_tracked_symbols()
    if not tracked_symbols:
        return

    cached_symbols = get_cached_symbols()
    missing_symbols = [symbol for symbol in tracked_symbols if symbol not in cached_symbols]

    for symbol in missing_symbols:
        try:
            stock_info = get_stock_info(symbol)
            company_name = stock_info['companyName'] if stock_info else symbol
            articles, sentiment_summary, overall_sentiment = build_sentiment_payload(
                symbol,
                company_name,
                sentiment_analyzer=analyze_sentiment_llm7
            )
            save_sentiment_to_db(symbol, articles, sentiment_summary, overall_sentiment)
        except Exception as e:
            print(f"Error ensuring sentiment for {symbol}: {e}")

    purge_stale_sentiment_records(tracked_symbols)


def background_worker_loop():
    last_full_refresh = 0
    while True:
        now = time.time()
        if now - last_full_refresh >= BACKGROUND_REFRESH_SECONDS:
            try:
                refresh_sentiment_cache()
            except Exception as e:
                print(f"Background refresh error: {e}")
            else:
                last_full_refresh = now

        try:
            ensure_cache_completeness()
        except Exception as e:
            print(f"Cache health check error: {e}")

        time.sleep(CACHE_HEALTH_CHECK_SECONDS)


def start_background_worker():
    global background_thread_started
    if background_thread_started:
        return
    init_db()
    thread = threading.Thread(target=background_worker_loop, daemon=True)
    thread.start()
    background_thread_started = True


@app.route('/')
@limiter.limit("50 per minute")
def index():
    '''Render the main page'''
    return render_template('index.html')

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'),
                               'favicon.ico', mimetype='image/vnd.microsoft.icon')

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

        tracked_symbols = get_tracked_symbols()
        cached_sentiment = get_cached_sentiment(stock_symbol) if stock_symbol in tracked_symbols else None

        if cached_sentiment:
            response_data['articles'] = cached_sentiment['articles']
            response_data['sentiment_summary'] = cached_sentiment['sentiment_summary']
            response_data['overall_sentiment'] = cached_sentiment['overall_sentiment']
        else:
            articles, sentiment_summary, overall_sentiment = build_sentiment_payload(
                stock_symbol,
                stock_info['companyName']
            )
            response_data['articles'] = articles
            response_data['sentiment_summary'] = sentiment_summary
            response_data['overall_sentiment'] = overall_sentiment

            if stock_symbol in tracked_symbols:
                save_sentiment_to_db(stock_symbol, articles, sentiment_summary, overall_sentiment)

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




if __name__ == '__main__':
    if os.getenv("DISABLE_BACKGROUND_WORKER") != "1":
        start_background_worker()
    app.run(
        host='0.0.0.0', 
        debug=False, 
    )
