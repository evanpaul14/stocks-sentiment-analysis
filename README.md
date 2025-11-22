# Stock Sentiment Analysis

A real-time stock analysis dashboard combining financial data, AI-powered news sentiment, and multi-source trending tickers. Search any company to view live prices, historical charts, key stats, and sentiment analysis of recent news. Explore trending stocks from Reddit, StockTwits, and Alpaca volume leaders—all in a modern, responsive UI.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Multi-Source Trending Dashboards:** View top stocks trending on Reddit (ApeWisdom), StockTwits, and Alpaca (most active by volume). Click any ticker to instantly analyze it.
- **Search by Company Name:** Enter any company name to resolve its ticker and view full details.
- **Comprehensive Stock Data:** Real-time price, after-hours data, market cap, P/E, dividend yield, volume, daily/52-week high/low, open price, company info (CEO, employees, headquarters, year founded, industry, sector, website, description).
- **Historical Data Visualization:** Interactive Chart.js charts for 1D, 1W, 1M, 3M, YTD, 1Y, 5Y, ALL timeframes.
- **AI-Powered News Sentiment:** News articles fetched via Google News and analyzed for sentiment using Google Gemma AI. Sentiment summary and badges shown for each article.
- **On-Demand Sentiment:** Each request analyzes the freshest set of articles through Google Gemma AI—no stale cache to worry about.
- **Rate Limiting:** All endpoints are rate-limited for abuse protection.
- **Modern Responsive UI:** Dark mode, trending dashboards, search, and detailed results.

## Technology Stack

**Backend:** Flask, Flask-Limiter, yfinance, yahooquery, pygooglenews, Google Gemma AI, requests, python-dotenv

**Frontend:** HTML5/CSS3, Chart.js, Vanilla JavaScript

**External APIs:** ApeWisdom (Reddit), StockTwits, Alpaca, Yahoo Finance, Google News, Google Gemma AI

## Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemma AI)
- Alpaca API Key/Secret (for volume dashboard)
- Internet connection for real-time data

## Installation

1. **Clone the repository**
  ```bash
  git clone https://github.com/yourusername/stocks-sentiment-analysis.git
  cd stocks-sentiment-analysis
  ```
2. **Create a virtual environment**
  ```bash
  python -m venv venv
  source venv/bin/activate  # On Windows: venv\Scripts\activate
  ```
3. **Install dependencies**
  ```bash
  pip install -r requirements.txt
  ```
4. **Set up environment variables**
  See below for required keys and a sample `.env` file.

## Environment Variables

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
```
**Descriptions:**
- `GOOGLE_API_KEY`: Used for Google Gemma AI sentiment analysis. [Get your key here](https://aistudio.google.com/)
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`: Used for Alpaca stock data API (for volume dashboard).

**Note:** Never share your `.env` file publicly. Keep your API keys secure.

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

## Usage

1. **Start the Flask server**
  ```bash
  python main.py
  ```
  The server will start on `http://0.0.0.0:5000`
  
2. **Open your browser**
  Navigate to `http://127.0.0.1:5000`

3. **Explore Trending Dashboards**
  - Home page shows three dashboards: StockTwits, Reddit, and Most Active by Volume (Alpaca)
  - Click any ticker to instantly search and analyze it

4. **Search for a Stock**
  - Enter a company name (e.g., "Apple", "Tesla", "Microsoft")
  - Click "Search" or press Enter
  - View stock data, charts, and sentiment analysis
  - Every search runs fresh sentiment analysis to ensure the latest tone from the news cycle

## API Endpoints

### `GET /`
Render the main application interface.
**Rate Limit:** 50 requests per minute

### `POST /search`
Search for a company and retrieve stock data with sentiment analysis.
**Rate Limit:** 10 requests per minute
**Request Body:**
```json
{
  "company_name": "Apple"
}
```
**Response:**
```json
{
  "stock_info": { ... },
  "historical_data": [ ... ],
  "articles": [ ... ],
  "sentiment_summary": { ... },
  "overall_sentiment": "positive"
}
```

### `GET /historical/<symbol>/<period>`
Retrieve historical price data for a specific timeframe.
**Rate Limit:** 50 requests per minute
**Parameters:**
- `symbol` (string): Stock ticker symbol (e.g., AAPL)
- `period` (string): One of `1d`, `5d`, `1w`, `1mo`, `3mo`, `ytd`, `1y`, `5y`, `max`
**Response:**
```json
{
  "data": [ ... ]
}
```

### `GET /trending`
Fetch trending stocks from Reddit, StockTwits, and Alpaca volume leaders.
**Rate Limit:** 30 requests per minute
**Response:**
```json
{
  "stocktwits": [ ... ],
  "reddit": [ ... ],
  "volume": [ ... ]
}
```

## Features Explained

### Trending Dashboards
- **Reddit (ApeWisdom):** Top 20 most-mentioned stocks, percent increase in mentions, "Trending" or "Up X Spots" tags
- **StockTwits:** Top tickers by watchlist count, price, and summary
- **Alpaca Volume Leaders:** Most active stocks by trading volume

### Stock Information
- Real-time price, after-hours data, market cap, P/E, dividend yield, volume, daily/52-week high/low, open price
- Company details: CEO, employees, headquarters, year founded, industry, sector, website, description

### Historical Data Visualization
- Chart.js interactive charts for 1D, 1W, 1M, 3M, YTD, 1Y, 5Y, ALL

### AI News Sentiment
- News articles fetched via Google News
- Sentiment analyzed using Google Gemma AI (Gemma 3 27B model) on every request
- Sentiment summary and badges for each article

### Rate Limiting
- All endpoints are rate-limited for abuse protection

### Background Worker
- No background worker is required now that caching has been removed. The application runs entirely on-demand.

## Configuration

### Sentiment Analysis Model
Default: `gemma-3-27b-it`. To change, edit `main.py`:
```python
gemma_model = "your-preferred-model"
```

### Number of News Articles
Default: 10. To change, modify the `get_news_articles` call in `main.py`:
```python
articles = get_news_articles(stock_symbol, 20)  # Analyze 20 articles
```

### Rate Limiting
To adjust, modify the limiter configuration in `main.py`:
```python
limiter = Limiter(
   get_remote_address,
   app=app,
   default_limits=["100 per hour"]  # Adjust as needed
)
```
