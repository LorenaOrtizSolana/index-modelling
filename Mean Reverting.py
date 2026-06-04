import pandas as pd
import numpy as np
import yfinance as yf


def select_mean_reversion_etfs(rebalance_date, prices_as_of, full_price_history,
                               vix_prices=None,
                               min_volume=1_000_000,
                               max_etfs=10,
                               trend_window=200,
                               max_trend_distance=0.05,
                               max_vix=25,
                               exclude_volatility_etfs=True):
    historical = full_price_history[full_price_history.index <= rebalance_date]

    if len(historical) < trend_window:
        return []

    if vix_prices is not None:
        vix_historical = vix_prices[vix_prices.index <= rebalance_date]
        if len(vix_historical) > 0:
            current_vix = vix_historical.iloc[-1]
            if current_vix > max_vix:
                return []

    current_prices = historical.iloc[-1]

    ma = historical.rolling(window=trend_window).mean().iloc[-1]

    trend_distance = abs((current_prices - ma) / ma)

    range_bound_etfs = trend_distance[trend_distance <= max_trend_distance].index.tolist()

    if exclude_volatility_etfs:
        volatility_etfs = ['VXX', 'UVXY', 'SVXY', 'XIV', 'VIXY', 'VXZ', 'VIX']
        range_bound_etfs = [t for t in range_bound_etfs if t not in volatility_etfs]

    sector_etfs = ['XLF', 'XLE', 'XLV', 'XLI', 'XLB', 'XLU', 'XLK', 'XLP', 'XLY']
    broad_etfs = ['SPY', 'QQQ', 'IWM', 'DIA', 'EFA', 'EEM']
    bond_etfs = ['TLT', 'LQD', 'HYG', 'SHY', 'IEF']

    prioritized = []
    for etf in sector_etfs:
        if etf in range_bound_etfs:
            prioritized.append(etf)
    for etf in bond_etfs:
        if etf in range_bound_etfs and etf not in prioritized:
            prioritized.append(etf)
    for etf in broad_etfs:
        if etf in range_bound_etfs and etf not in prioritized:
            prioritized.append(etf)
    for etf in range_bound_etfs:
        if etf not in prioritized:
            prioritized.append(etf)

    select_mean_reversion_etfs.last_metadata = {
        'total_etfs_considered': len(prices_as_of),
        'range_bound_etfs': len(range_bound_etfs),
        'trend_distances': trend_distance,
        'current_vix': current_vix if vix_prices is not None else None,
        'rebalance_date': rebalance_date
    }

    return prioritized[:max_etfs]


def weight_equal_long_only(selected_tickers, prices_as_of, rebalance_date,
                           max_weight=0.2):
    if len(selected_tickers) == 0:
        return {}

    weight = min(1.0 / len(selected_tickers), max_weight)
    return {ticker: weight for ticker in selected_tickers}


def calculate_zscore_signals(prices, etf_list, window=20):
    etf_prices = prices[etf_list]

    rolling_mean = etf_prices.rolling(window=window).mean()
    rolling_std = etf_prices.rolling(window=window).std()

    zscore = (etf_prices - rolling_mean) / rolling_std

    signals = pd.DataFrame(0, index=zscore.index, columns=zscore.columns)
    signals[zscore < -2] = 1
    signals[zscore > 2] = -1

    return zscore, signals