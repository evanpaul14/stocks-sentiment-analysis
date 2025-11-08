from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockLatestQuoteRequest
import os
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("APCA_API_KEY_ID"))

# keys required for stock historical data client
client = StockHistoricalDataClient(os.getenv('APCA_API_KEY_ID'), os.getenv('APCA_API_SECRET_KEY'))

# multi symbol request - single symbol is similar
multisymbol_request_params = StockLatestQuoteRequest(symbol_or_symbols=["AAPL"])

latest_multisymbol_quotes = client.get_stock_latest_quote(multisymbol_request_params)

gld_latest_ask_price = latest_multisymbol_quotes["AAPL"].ask_price
print(gld_latest_ask_price)

import requests

url = "https://data.alpaca.markets/v2/stocks/meta/exchanges"

headers = {"accept": "application/json"}

response = requests.get(url, headers=headers)

print(response.text)