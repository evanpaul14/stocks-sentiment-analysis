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

load_dotenv()
app = Flask(__name__)

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
            'ceo': info.get('companyOfficers', [{}])[0].get('name', 'N/A') if info.get('companyOfficers') else 'N/A',
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

        # For intraday data (1d), keep your existing logic
        if period == '1d':
            hist = ticker.history(period='5d', interval='5m')
            if not hist.empty:
                last_date = hist.index[-1].date()
                hist = hist[hist.index.date == last_date]

        # For 1 week, use explicit start/end dates and 1h interval
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
            # Default logic for other periods
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
    """Analyze sentiment using Google Gemini"""
    try:
        prompt = (
            f"Analyze the sentiment (positive, negative, or neutral) of this news article strictly in reference "
            f"to the company {company_name}.\n\n"
            f"Title: {article_title}\n"
            f"Description: {article_description}\n\n"
            f"Respond with only one word and nothing else: positive, negative, or neutral."
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

