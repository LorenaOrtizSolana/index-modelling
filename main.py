import pandas as pd
import numpy as np
import yfinance as yf


def fetch_prices(tickers, start_date, end_date, flag_single_missing=True):
    flags = {
        'holidays_removed': [],
        'delisted_stocks': [],
        'single_missing_dates': [],
        'forward_filled_gaps': 0,
        'stocks_with_no_data': []
    }

    print(f"Fetching data for {len(tickers)} tickers from {start_date} to {end_date}...")

    data = yf.download(tickers, start=start_date, end=end_date,
                       progress=False,
                       group_by='ticker')

    print(f"Raw data shape: {data.shape}")

    if isinstance(data.columns, pd.MultiIndex):
        if 'Adj Close' in data.columns.levels[0]:
            prices = data.xs('Adj Close', axis=1, level=0)
        elif 'Close' in data.columns.levels[0]:
            prices = data.xs('Close', axis=1, level=0)
        else:
            prices = data.xs(data.columns.levels[0][0], axis=1, level=0)
    else:
        prices = data

    valid_columns = []
    for col in prices.columns:
        if prices[col].notna().sum() > 0:
            valid_columns.append(col)
        else:
            flags['stocks_with_no_data'].append(col)

    prices = prices[valid_columns]

    if flags['stocks_with_no_data']:
        print(f"Removed {len(flags['stocks_with_no_data'])} tickers with no data: {flags['stocks_with_no_data']}")

    if len(prices.columns) == 0:
        print("ERROR: No valid tickers found")
        return pd.DataFrame(), flags

    print(f"Extracted prices: {len(prices)} days, {len(prices.columns)} stocks")

    missing_by_date = prices.isnull().sum(axis=1)
    holiday_mask = missing_by_date > (len(prices.columns) * 0.5)
    holiday_dates = prices.index[holiday_mask].tolist()
    flags['holidays_removed'] = holiday_dates
    prices = prices.loc[~holiday_mask]

    if holiday_dates:
        print(f"Removed {len(holiday_dates)} holiday dates with >50% missing")

    if flag_single_missing:
        for date in prices.index:
            missing_tickers = prices.columns[prices.loc[date].isnull()].tolist()
            if len(missing_tickers) == 1:
                flags['single_missing_dates'].append({
                    'date': date,
                    'ticker': missing_tickers[0]
                })

    for ticker in prices.columns:
        last_20_pct = int(len(prices) * 0.2)
        if last_20_pct < 5:
            last_20_pct = 5

        last_values = prices[ticker].iloc[-last_20_pct:]

        if last_values.isnull().sum() > len(last_values) * 0.5:
            flags['delisted_stocks'].append(ticker)
            prices = prices.drop(columns=ticker)

    if flags['delisted_stocks']:
        print(f"Removed {len(flags['delisted_stocks'])} delisted stocks")

    original_nulls = prices.isnull().sum().sum()
    prices = prices.ffill(limit=5)
    prices = prices.bfill(limit=2)
    filled_nulls = original_nulls - prices.isnull().sum().sum()
    flags['forward_filled_gaps'] = filled_nulls

    if filled_nulls > 0:
        print(f"Forward filled {filled_nulls} missing values")

    rows_before = len(prices)
    prices = prices.dropna()
    rows_dropped = rows_before - len(prices)

    if rows_dropped > 0:
        print(f"Dropped {rows_dropped} rows with unfixable missing data")

    print(f"Final data: {len(prices)} days, {len(prices.columns)} stocks")

    return prices, flags

"""
bt = Backtest(prices, rebalance_freq='M', transaction_cost_bps=10)

results = bt.run(
    lambda d, p: select_momentum(d, p, top_pct=0.3, lookback_days=252, skip_days=21),
    lambda s, p, d: weight_equal_momentum(s, p, d)
)
"""