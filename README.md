# Stock Sentiment Analysis

A real-time stock analysis application that combines financial data with AI-powered news sentiment analysis. Search for any publicly traded company to view live stock prices, historical charts, key statistics, and sentiment analysis of recent news articles. Also features trending stocks from Reddit communities.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-Time Stock Data**: Live prices, market cap, P/E ratio, volume, and more
- **Interactive Charts**: View historical data across multiple timeframes (1D, 1W, 1M, 3M, YTD, 1Y, 5Y, ALL)
- **AI-Powered Sentiment Analysis**: Analyze news sentiment using Google Gemini AI
- **Trending Stocks**: View top 20 trending stocks from Reddit with "Fast Rising" indicators
- **News Aggregation**: Latest news articles from multiple sources
- **Smart Caching**: 12-hour cache for news and sentiment data to optimize API usage
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
- **Google Gemma AI**: Sentiment analysis (Gemini 3 27B model)
- **Requests**: HTTP library for trending stocks API
- **python-dotenv**: Environment variable management

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **Chart.js**: Interactive stock charts
- **Vanilla JavaScript**: Dynamic UI updates
- **LocalStorage API**: Client-side caching for news data

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

Create a `.env` file in the project root:
```
GOOGLE_API_KEY=your_google_api_key_here
```

To obtain a Google API key:
- Visit [Google AI Studio](https://aistudio.google.com/)
- Create a new API key
- Copy and paste it into your `.env` file

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
- News and sentiment data is cached for 12 hours to optimize performance

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
  "company_name": "Apple",
  "use_cache": false
}
```

**Parameters:**
- `company_name` (string, required): Company name or ticker symbol
- `use_cache` (boolean, optional): If true, skip news fetching (frontend handles cache logic)

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
- News articles and sentiment analysis are cached in browser localStorage
- Cache duration: 12 hours
- Stock price and historical data are always fetched fresh
- Cache is automatically invalidated after expiration
- Reduces unnecessary API calls to Google Gemini

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
The default model is `gemini-3-27b-it`. To change the model, edit `main.py`:
```python
gemma_model = "your-preferred-model"
```

### Number of News Articles
To change the number of articles analyzed (default: 10), modify the `get_news_articles` call in `main.py`:
```python
articles = get_news_articles(stock_symbol, 20)  # Analyze 20 articles
```

### Cache Duration
To adjust the news cache duration (default: 12 hours), edit the constant in `index.html`:
```javascript
const CACHE_DURATION_MS = 12 * 60 * 60 * 1000; // Change to desired milliseconds
```

### Rate Limiting
To adjust rate limits, modify the limiter configuration in `main.py`:
```python
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["100 per hour"]  # Adjust as needed
)
```
