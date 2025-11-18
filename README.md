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
- **Smart Sentiment Caching:** Sentiment for top trending and flagship tickers is cached in SQLite and refreshed hourly. Health checks run every 10 minutes.
- **Rate Limiting:** All endpoints are rate-limited for abuse protection.
- **Background Worker:** Sentiment cache is maintained in the background unless disabled.
- **Modern Responsive UI:** Dark mode, trending dashboards, search, and detailed results.

## Technology Stack

**Backend:** Flask, Flask-Limiter, yfinance, yahooquery, pygooglenews, Google Gemma AI, requests, python-dotenv, SQLite

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
├── experimentation.py    # Example scripts
├── templates/
│   └── index.html        # Frontend interface
├── static/
│   └── favicon.ico       # Website icon
├── .env                  # Environment variables
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

4. **Explore Trending Dashboards**
  - Home page shows three dashboards: StockTwits, Reddit, and Most Active by Volume (Alpaca)
  - Click any ticker to instantly search and analyze it

4. **Search for a Stock**
  - Enter a company name (e.g., "Apple", "Tesla", "Microsoft")
  - Click "Search" or press Enter
  - View stock data, charts, and sentiment analysis
  - Tracked tickers load instantly thanks to the hourly backend sentiment cache

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
- Sentiment analyzed using Google Gemma AI (Gemma 3 27B model)
- Sentiment summary and badges for each article

### Smart Sentiment Caching
- Sentiment for top trending and flagship tickers cached in SQLite
- Refreshed hourly by background worker
- Health checks every 10 minutes
- Tracked tickers load instantly; non-tracked tickers analyzed on-demand

### Rate Limiting
- All endpoints are rate-limited for abuse protection

### Background Worker
- Sentiment cache maintained in the background unless `DISABLE_BACKGROUND_WORKER=1` is set

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

### Cache Health Interval
Default: 10 minutes. Update `CACHE_HEALTH_CHECK_SECONDS` in `main.py`.

### Rate Limiting
To adjust, modify the limiter configuration in `main.py`:
```python
limiter = Limiter(
   get_remote_address,
   app=app,
   default_limits=["100 per hour"]  # Adjust as needed
)
```

### Background Worker
To disable, set `DISABLE_BACKGROUND_WORKER=1` before launching the app or importing `main`.
# Stock Sentiment Analysis

A real-time stock analysis application that combines financial data with AI-powered news sentiment analysis. Search for any publicly traded company to view live stock prices, historical charts, key statistics, and sentiment analysis of recent news articles. Also features trending stocks from Reddit communities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-Time Stock Data**: Live prices, market cap, P/E ratio, volume, and more
- **Interactive Charts**: View historical data across multiple timeframes (1D, 1W, 1M, 3M, YTD, 1Y, 5Y, ALL)
- **AI-Powered Sentiment Analysis**: Analyze news sentiment using Google Gemma AI
- **Trending Stocks**: View top 20 trending stocks from Reddit with Trending indicators
- **News Aggregation**: Latest news articles from multiple sources
- **Automated Sentiment Harvesting**: Hourly background worker precomputes news sentiment for trending and flagship tickers
- **Company Information**: PE Ratio, dividend yield, market cap, average volume, and business description
- **After-Hours Trading**: Track after-hours price movements
- **Dark Mode UI**: Modern, responsive interface with Chart.js visualizations
- **Rate Limiting**: Built-in protection against API abuse

## Technology Stack

### Backend
- **Flask**: Web framework
- **Flask-Limiter**: Rate limiting for API endpoints
- **yfinance**: Real-time stock data
- **yahooquery**: Company search functionality
- **pygooglenews**: News article aggregation
- **Google Gemma AI**: Sentiment analysis (Gemma 3 27B model)
- **Requests**: HTTP library for trending stocks API
- **python-dotenv**: Environment variable management
- **SQLite (built-in)**: Lightweight sentiment cache for precomputed news analysis

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **Chart.js**: Interactive stock charts
- **Vanilla JavaScript**: Dynamic UI updates

### External APIs
- **ApeWisdom API**: Trending stocks data from Reddit
- **Yahoo Finance**: Stock market data
- **Google News**: News article aggregation
- **Google Gemma AI**: AI sentiment analysis

## Prerequisites

- Python 3.8 or higher
- Google API Key (for Gemma AI)
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

See the next section for required environment variables and a sample `.env` file.

## Environment Variables

Create a `.env` file in the project root with the following keys:

```
GOOGLE_API_KEY=your_google_api_key_here
ALPACA_API_KEY=your_alpaca_api_key_here
ALPACA_SECRET_KEY=your_alpaca_secret_key_here
```

**Descriptions:**
- `GOOGLE_API_KEY`: Used for Google Gemma AI sentiment analysis. [Get your key here](https://aistudio.google.com/)
- `ALPACA_API_KEY` and `ALPACA_SECRET_KEY`: Used for Alpaca stock data API (if enabled in your code).

## Project Structure

```
stock-sentiment-analysis/
│
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

3. **View trending stocks**
- The home page displays the top 20 trending stocks from Reddit
- Click any stock to instantly search and analyze it
- "Fast Rising" tags indicate stocks with 50%+ mention increase or significant rank jumps

4. **Search for a stock**
- Enter a company name (e.g., "Apple", "Tesla", "Microsoft")
- Click "Search" or press Enter
- View stock data, charts, and sentiment analysis
- Tracked tickers load instantly thanks to the hourly backend sentiment cache

## API Endpoints

### `GET /`
Render the main application interface.

**Rate Limit:** 50 requests per minute

---

### `POST /search`
Search for a company and retrieve stock data with sentiment analysis.

**Rate Limit:** 50 requests per minute

**Request Body:**
```json
{
  "company_name": "Apple"
}
```

**Parameters:**
- `company_name` (string, required): Company name or ticker symbol

**Response:**
```json
{
  "stock_info": {
    "symbol": "AAPL",
    "price": 150.25,
    "change": 2.50,
    "changePercent": 1.69,
    "afterHoursPrice": 150.75,
    "afterHoursChange": 0.50,
    "afterHoursChangePercent": 0.33,
    "marketCap": 2500000000000,
    "peRatio": 28.5,
    "dividendYield": 0.0054,
    "avgVolume": 75000000,
    "dayHigh": 151.00,
    "dayLow": 148.50,
    "openPrice": 149.00,
    "volume": 80000000,
    "fiftyTwoWeekHigh": 180.00,
    "fiftyTwoWeekLow": 120.00,
    "companyName": "Apple Inc.",
    "ceo": "Tim Cook",
    "employees": 164000,
    "city": "Cupertino",
    "state": "CA",
    "country": "United States",
    "industry": "Consumer Electronics",
    "sector": "Technology",
    "website": "https://www.apple.com",
    "description": "Apple Inc. designs, manufactures...",
    "yearFounded": 1976
  },
  "historical_data": [
    {"date": "2024-11-13 09:30:00", "price": 149.50},
    {"date": "2024-11-13 09:35:00", "price": 150.00}
  ],
  "articles": [
    {
      "title": "Apple Announces New Product",
      "description": "Apple has unveiled...",
      "link": "https://...",
      "publishedAt": "2024-11-13T10:00:00Z",
      "source": "TechCrunch",
      "sentiment": "positive"
    }
  ],
  "sentiment_summary": {
    "positive": 6,
    "negative": 2,
    "neutral": 2
  },
  "overall_sentiment": "positive"
}
```

---

### `GET /historical/<symbol>/<period>`
Retrieve historical price data for a specific timeframe.

**Rate Limit:** 50 requests per minute

**Parameters:**
- `symbol` (string): Stock ticker symbol (e.g., AAPL)
- `period` (string): Time period - one of:
  - `1d` - 5-minute intervals (last trading day)
  - `5d` or `1w` - Daily data (1 week)
  - `1mo` - 90-minute intervals (1 month)
  - `3mo` - Daily data (3 months)
  - `ytd` - Daily data (year-to-date)
  - `1y` - Daily data (1 year)
  - `5y` - Weekly data (5 years)
  - `max` - Monthly data (all available)

**Response:**
```json
{
  "data": [
    {"date": "2024-11-13 09:30:00", "price": 149.50},
    {"date": "2024-11-13 09:35:00", "price": 150.00}
  ]
}
```

---

### `GET /trending`
Fetch top 20 trending stocks from Reddit communities (via ApeWisdom API).

**Rate Limit:** 30 requests per minute

**Response:**
```json
{
  "trending": [
    {
      "ticker": "TSLA",
      "name": "Tesla Inc",
      "pct_increase": 125.5,
      "tag": "FAST RISING"
    },
    {
      "ticker": "AAPL",
      "name": "Apple Inc",
      "pct_increase": 15.2,
      "tag": ""
    }
  ]
}
```

**Trending Tags:**
- `FAST RISING`: Assigned when mentions increased by 50%+ in 24 hours OR rank jumped by 5+ positions

---

## Features Explained

### Stock Information
The application displays comprehensive stock data:
- Current price with real-time updates
- Price change (absolute and percentage)
- After-hours trading data (price, change, percentage)
- Market capitalization
- P/E ratio and dividend yield
- Trading volume and average volume
- Daily high/low and open price
- 52-week high/low
- Company details (CEO, employees, headquarters, founding year)
- Industry and sector information
- Complete business description

### Sentiment Analysis
The application uses Google's Gemma AI (Gemma 3 27B model) to analyze news headlines and descriptions:
- **Positive**: Favorable news about the company
- **Negative**: Concerning or critical news
- **Neutral**: Balanced reporting or unclear cases

Each article is individually analyzed and categorized. The overall sentiment is determined by the most common sentiment across all analyzed articles (default: 10 articles per search).

### Smart Caching System
To optimize API usage and improve performance:
- A background worker refreshes sentiment every hour for the top 20 Reddit tickers plus key large-cap symbols (Alphabet, Amazon, Apple, Meta, Microsoft, Nvidia, Tesla, SoFi)
- News results and sentiment summaries are stored centrally in a SQLite cache so tracked symbols load instantly
- Non-tracked tickers trigger an on-demand analysis, which is saved if that ticker later becomes tracked
- A lightweight health check runs every 10 minutes to ensure all tracked tickers have a fresh entry and to purge symbols that fall off the tracked list
- Stock price and historical data remain real-time, so you still see the freshest market moves

### Historical Data Visualization
Interactive Chart.js charts allow you to view price trends across different timeframes:
- **1D**: 5-minute intervals (intraday trading)
- **1W**: Hourly data (past week)
- **1M**: 90-minute intervals (past month)
- **3M**: Daily data (past 3 months)
- **YTD**: Daily data (year-to-date)
- **1Y**: Daily data (past year)
- **5Y**: Weekly aggregation (past 5 years)
- **ALL**: Maximum available historical data

### Trending Stocks
- Displays top 20 most-mentioned stocks on Reddit
- Data sourced from ApeWisdom API
- Shows percent increase in mentions over 24 hours
- Highlights "Fast Rising" stocks with special tags
- Click any trending stock to instantly search and analyze it
- Automatically refreshes when returning to home page

### Rate Limiting
Built-in protection against API abuse:
- Search endpoint: 50 requests/minute
- Historical data: 50 requests/minute
- Trending stocks: 30 requests/minute
- Main page: 50 requests/minute

## Configuration

### Sentiment Analysis Model
The default model is `gemma-3-27b-it`. To change the model, edit `main.py`:
```python
gemma_model = "your-preferred-model"
```

### Number of News Articles
To change the number of articles analyzed (default: 10), modify the `get_news_articles` call in `main.py`:
```python
articles = get_news_articles(stock_symbol, 20)  # Analyze 20 articles
```

### Cache Health Interval
To change how frequently the worker verifies cache completeness (default: 10 minutes), update `CACHE_HEALTH_CHECK_SECONDS` in `main.py`.

### Rate Limiting
To adjust rate limits, modify the limiter configuration in `main.py`:
```python
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"]  # Adjust as needed
)
```

### Background Worker
To keep the hourly sentiment harvester from running (for example, during unit tests), set the environment variable `DISABLE_BACKGROUND_WORKER=1` before launching the app or importing `main`.
