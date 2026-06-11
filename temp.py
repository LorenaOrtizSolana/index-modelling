import pandas as pd
import numpy as np
import yfinance as yf
from Backtesting import Backtest
import Momentum
from Backtesting import Backtest
from Momentum import select_momentum, weight_equal_momentum
from MeanReverting import select_mean_reversion_etfs, weight_mean_reversion_scaled, run_mean_reversion_strategy
from RiskParity import select_risk_parity_assets, weight_risk_parity

def fetch_prices(tickers, start_date, end_date):
    flags = {"1":[],"2":[],"3":[], "4":[]}
    data = yf.download(tickers, start=start_date, end=end_date,
                       progress=False,
                       group_by='ticker')
    ##all adj close for all tickers
    #handle different levels with conditions

    ##remove tickers without a single data point

    return