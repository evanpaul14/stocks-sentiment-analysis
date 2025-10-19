import yfinance as yf
from datetime import datetime, timedelta

def get_historical_data(symbol):
    """Get last week's hourly data using yfinance"""
    try:
        end = datetime.now()
        start = end - timedelta(days=7)

        # Explicitly specify start/end instead of period
        hist = yf.download(
            symbol,
            start=start,
            end=end,
            interval="1h",
            prepost=True,   # include pre/post-market if available
            progress=False
        )

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
data = get_historical_data("AAPL")
print(len(data), "rows")
print(data)
