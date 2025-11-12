# Stock Sentiment Analysis

A real-time stock analysis application that combines financial data with AI-powered news sentiment analysis. Search for any publicly traded company to view live stock prices, historical charts, key statistics, and sentiment analysis of recent news articles.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Flask](https://img.shields.io/badge/Flask-2.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

- **Real-Time Stock Data**: Live prices, market cap, P/E ratio, volume, and more
- **Interactive Charts**: View historical data across multiple timeframes (1D, 1W, 1M, 3M, YTD, 1Y, 5Y, ALL)
- **AI-Powered Sentiment Analysis**: Analyze news sentiment using Google Gemini AI
- **News Aggregation**: Latest news articles from multiple sources
- **Company Information**: PE Ratio, dividend yield, market cap, average volume, and business description
- **After-Hours Trading**: Track after-hours price movements
- **Dark Mode UI**: Modern, responsive interface with Chart.js visualizations

## Technology Stack

### Backend
- **Flask**: Web framework
- **yfinance**: Real-time stock data
- **yahooquery**: Company search functionality
- **pygooglenews**: News article aggregation
- **Google Gemma AI**: Sentiment analysis (Gemma 3 27B model)

### Frontend
- **HTML5/CSS3**: Modern responsive design
- **Chart.js**: Interactive stock charts
- **Vanilla JavaScript**: Dynamic UI updates

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
stock-sentiment-analyzer/
│
├── main.py               # Flask backend application
├── templates/
│   └── index.html        # Frontend interface
├── static/
│   └── favicon.ico 
├── .env                  # Environment variables (create this)
├── .LICENSE              # MIT License
└── README.md             # This file
```

## Usage

1. **Start the Flask server**
```bash
python main.py
```

2. **Open your browser**
Navigate to `https://127.0.0.1:5000`

3. **Search for a stock**
- Enter a company name (e.g., "Apple", "Tesla", "Microsoft")
- Click "Search" or press Enter
- View stock data, charts, and sentiment analysis

## API Endpoints

### `POST /search`
Search for a company and retrieve stock data with sentiment analysis.

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
  "sentiment_summary": {
    "positive": 5,
    "negative": 2,
    "neutral": 3
  },
  "overall_sentiment": "positive"
}
```

### `GET /historical/<symbol>/<period>`
Retrieve historical price data for a specific timeframe.

**Parameters:**
- `symbol`: Stock ticker symbol (e.g., AAPL)
- `period`: Time period (1d, 5d, 1mo, 3mo, ytd, 1y, 5y, max)

## Features Explained

### Stock Information
- Current price with real-time updates
- Price change (absolute and percentage)
- After-hours trading data
- Market capitalization
- P/E ratio and dividend yield
- Trading volume and ranges
- 52-week high/low

### Sentiment Analysis
The application uses Google's Gemma AI to analyze news headlines and descriptions:
- **Positive**: Favorable news about the company
- **Negative**: Concerning or critical news
- **Neutral**: Factual or balanced reporting

Each article is individually analyzed and categorized, with an overall sentiment score calculated based on the distribution.

### Historical Data Visualization
Interactive charts allow you to view price trends across different timeframes:
- **1D**: 5-minute intervals (intraday)
- **1W**: Daily data
- **1M**: 90-minute intervals
- **3M, YTD, 1Y**: Daily data
- **5Y, ALL**: Weekly/monthly aggregation

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
