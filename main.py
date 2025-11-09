from flask import Flask, render_template, request, jsonify
from google import genai
from pygooglenews import GoogleNews
import requests
import json
import re
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from yahooquery import search

load_dotenv()
app = Flask(__name__)

# Rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"]
)

# Google Gemini setup
apikey = os.getenv('GOOGLE_API_KEY')
if not apikey:
    raise ValueError("Secret key not set in environment!")
client = genai.Client(api_key=apikey)
gemma_model = "gemma-3-27b-it"

# Alpha Vantage API key
av_key = os.getenv("ALPHA_VANTAGE_KEY")
if not av_key:
    raise ValueError("Alpha Vantage key not set in environment!")

BASE_URL = "https://www.alphavantage.co/query"


def get_stock_info(symbol):
    """Get stock information using Alpha Vantage API (drop‑in replacing yfinance)"""
    try:
        # 1) Get quote
        quote_url = (
            f"https://www.alphavantage.co/query?function=GLOBAL_QUOTE"
            f"&symbol={symbol}&apikey={av_key}"
        )
        r1 = requests.get(quote_url)
        r1.raise_for_status()
        qdata = r1.json().get("Global Quote", {})

        current_price = float(qdata.get("05. price", 0))
        previous_close = float(qdata.get("08. previous close", 0))
        change = current_price - previous_close if previous_close else 0
        change_percent = (
            (change / previous_close * 100) if previous_close else 0
        )

        # 2) Get fundamentals/overview
        ov_url = (
            f"https://www.alphavantage.co/query?function=OVERVIEW"
            f"&symbol={symbol}&apikey={av_key}"
        )
        r2 = requests.get(ov_url)
        r2.raise_for_status()
        ov = r2.json()

        return {
            'symbol': symbol,
            'price': current_price,
            'change': change,
            'changePercent': change_percent,
            'afterHoursPrice': None,
            'afterHoursChange': None,
            'afterHoursChangePercent': None,
            'marketCap': float(ov.get('MarketCapitalization', 0)),
            'peRatio': float(ov.get('PERatio', 0)) if ov.get('PERatio') else 0,
            'dividendYield': float(ov.get('DividendYield', 0)) if ov.get('DividendYield') else 0,
            'avgVolume': None,
            'dayHigh': None,
            'dayLow': None,
            'openPrice': float(qdata.get("02. open", 0)),
            'volume': float(qdata.get("06. volume", 0)),
            'fiftyTwoWeekHigh': None,
            'fiftyTwoWeekLow': None,
            'companyName': ov.get('Name', symbol),
            'ceo': None,
            'employees': None,
            'city': None,
            'state': None,
            'country': None,
            'industry': ov.get('Industry'),
            'sector': ov.get('Sector'),
            'website': ov.get('Website'),
            'description': ov.get('Description'),
            'yearFounded': int(ov.get('Founded')) if ov.get('Founded') else 'N/A'
        }
    except Exception as e:
        print(f"Error getting stock info (AV): {e}")
        return None


def get_historical_data(symbol, period='1d'):
    """Get historical price data using Alpha Vantage API (drop‑in)"""
    try:
        if period == '1d':
            # use intraday 5min interval
            interval = '5min'
            func = 'TIME_SERIES_INTRADAY'
            url = (
                f"https://www.alphavantage.co/query?function={func}"
                f"&symbol={symbol}&interval={interval}&outputsize=compact"
                f"&apikey={av_key}"
            )
        elif period in ['7d', '1w']:
            # use daily adjusted and then filter last ~7 entries
            func = 'TIME_SERIES_DAILY_ADJUSTED'
            url = (
                f"https://www.alphavantage.co/query?function={func}"
                f"&symbol={symbol}&outputsize=compact"
                f"&apikey={av_key}"
            )
        elif period == '1mo':
            func = 'TIME_SERIES_DAILY_ADJUSTED'
            url = (
                f"https://www.alphavantage.co/query?function={func}"
                f"&symbol={symbol}&outputsize=full"
                f"&apikey={av_key}"
            )
        else:
            func = 'TIME_SERIES_DAILY_ADJUSTED'
            url = (
                f"https://www.alphavantage.co/query?function={func}"
                f"&symbol={symbol}&outputsize=compact"
                f"&apikey={av_key}"
            )

        r = requests.get(url)
        r.raise_for_status()
        data = r.json()

        # pick the relevant key based on function
        # For intraday: "Time Series (5min)"
        # For daily: "Time Series (Daily)"
        if func == 'TIME_SERIES_INTRADAY':
            ts_key = f"Time Series ({interval})"
        else:
            ts_key = "Time Series (Daily)"

        series = data.get(ts_key, {})
        result = []
        for timestamp, values in series.items():
            result.append({
                "date": timestamp,
                "price": float(values.get("4. close", 0))
            })

        # sort ascending if needed
        result = sorted(result, key=lambda x: x["date"])
        return result

    except Exception as e:
        print(f"Error getting historical data (AV): {e}")
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
    """Analyze sentiment using Google Gemini"""
    try:
        prompt = (
            f"Analyze the sentiment (positive, negative, or neutral) of this news article strictly in reference "
            f"to the company {company_name}.\n\n"
            f"Title: {article_title}\n"
            f"Description: {article_description}\n\n"
            f"Respond with only one word: positive, negative, or neutral."
        )

        response = client.models.generate_content(
            model=gemma_model,
            contents=prompt
        )

        sentiment = response.text.strip().lower()
        if sentiment not in ['positive', 'negative', 'neutral']:
            sentiment = 'neutral'
        return sentiment
    except Exception as e:
        print(f"Error analyzing sentiment: {e}")
        return 'neutral'


@app.route('/')
@limiter.limit("50 per minute")
def index():
    '''Render the main page'''
    return render_template('index.html')


@app.route('/search', methods=['POST'])
@limiter.limit("50 per minute")
def search_stock():
    '''Search for stock and return info, historical data, news, and sentiment analysis'''
    try:
        company_name = request.json.get('company_name')
        use_cache = request.json.get('use_cache', False)

        # Search for stock symbol
        search_url = (
        f"https://www.alphavantage.co/query?function=SYMBOL_SEARCH"
        f"&keywords={company_name}&apikey={av_key}"
)
        r = requests.get(search_url)
        r.raise_for_status()
        search_res = r.json().get("bestMatches", [])
        if not search_res:
            return jsonify({'error': 'Company not found'}), 404
        stock_symbol = search_res[0].get("1. symbol")


        # Get stock information
        stock_info = get_stock_info(stock_symbol)
        if not stock_info:
            return jsonify({'error': 'Failed to retrieve stock information'}), 500

        # Get historical data for 1 day
        historical_data = get_historical_data(stock_symbol, '1d')

        # Only fetch news if not using cache (frontend will handle cache logic)
        response_data = {
            'stock_info': stock_info,
            'historical_data': historical_data
        }

        if not use_cache:
            # Get news articles
            articles = get_news_articles(stock_symbol, 10)

            # Analyze sentiment for each article
            sentiment_summary = {"positive": 0, "negative": 0, "neutral": 0}
            for article in articles:
                sentiment = analyze_sentiment_gemma(
                    article['title'],
                    article['description'],
                    stock_info['companyName']
                )
                article['sentiment'] = sentiment
                sentiment_summary[sentiment] += 1

            # Determine overall sentiment
            most_common_sentiment = max(sentiment_summary, key=sentiment_summary.get)

            response_data['articles'] = articles
            response_data['sentiment_summary'] = sentiment_summary
            response_data['overall_sentiment'] = most_common_sentiment

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


if __name__ == '__main__':
    app.run(
        host='0.0.0.0', 
        debug=False, 
    )

