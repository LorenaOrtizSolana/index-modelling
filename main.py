import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def fetch_prices(tickers, start_date, end_date):
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker')

    if len(tickers) == 1:
        prices = pd.DataFrame(data['Adj Close'])
        prices.columns = tickers
    else:
        prices = data['Adj Close']

    return prices

prices = fetch_prices(['AAPL', 'MSFT', 'GOOGL', 'AMZN'], '2020-01-01', '2024-12-31')

