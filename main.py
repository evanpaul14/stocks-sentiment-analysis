from flask import Flask, render_template, request, jsonify
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
from alpaca.data.requests import StockLatestQuoteRequest, StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
from datetime import datetime, timedelta

load_dotenv()
app = Flask(__name__)
ALPACA_API_KEY = os.getenv('APCA_API_KEY_ID')
ALPACA_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')
client = StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET_KEY)
print(os.getenv("APCA_API_KEY_ID"))


# set up rate limiting
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["500 per hour"] 
)

# Initialize Google Gemini client
apikey = os.getenv('GOOGLE_API_KEY')
if not apikey:
    raise ValueError("Secret key not set in environment!")
client = genai.Client(api_key=apikey)
gemma_model = "gemma-3-27b-it"


def get_stock_info(symbol):
    try:
        # latest quote request
        quote_req = StockLatestQuoteRequest(symbol_or_symbols=[symbol])
        latest = client.get_stock_latest_quote(quote_req)

        current_price = latest.ask_price
        # If previousClose not available via quote, set fallback
        previous_close = None  
        if hasattr(latest, 'bp'):  # bid price attribute
            previous_close = latest.bp

        change = (current_price - previous_close) if previous_close else 0
        change_percent = (change / previous_close * 100) if previous_close else 0

        return {
            'symbol': symbol,
            'price': current_price,
            'change': change,
            'changePercent': change_percent,
            # fill others with None or default if not supported by Alpaca
            'afterHoursPrice': None,
            'afterHoursChange': None,
            'afterHoursChangePercent': None,
            'marketCap': None,
            'peRatio': None,
            'dividendYield': None,
            'avgVolume': None,
            'dayHigh': None,
            'dayLow': None,
            'openPrice': None,
            'volume': None,
            'fiftyTwoWeekHigh': None,
            'fiftyTwoWeekLow': None,
            'companyName': symbol,
            'ceo': 'N/A',
            'employees': None,
            'city': None,
            'state': None,
            'country': None,
            'industry': None,
            'sector': None,
            'website': None,
            'description': None,
            'yearFounded': None
        }
    except Exception as e:
        print(f"Error getting stock info (Alpaca): {e}")
        return None

def get_historical_data(symbol, period='1d'):
    try:
        end = datetime.now()
        if period == '1d':
            start = end - timedelta(days=1)
            timeframe = TimeFrame.Minute
        elif period in ['7d', '1w']:
            start = end - timedelta(days=7)
            timeframe = TimeFrame.Hour
        elif period == '1mo':
            start = end - timedelta(days=30)
            timeframe = TimeFrame.Day
        else:
            # fallback
            start = end - timedelta(days=7)
            timeframe = TimeFrame.Day

        bars_req = StockBarsRequest(
            symbol_or_symbols=[symbol],
            timeframe=timeframe,
            start=start,
            end=end
        )
        bars = client.get_stock_bars(bars_req)[symbol]

        data = [
            {"date": bar.timestamp.strftime("%Y-%m-%d %H:%M:%S"), "price": bar.close}
            for bar in bars
        ]
        return data
    except Exception as e:
        print(f"Error getting historical data (Alpaca): {e}")
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
        results = search(company_name)
        if not results.get('quotes') or len(results['quotes']) == 0:
            return jsonify({'error': 'Company not found'}), 404

        stock_symbol = results['quotes'][0]['symbol']

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

