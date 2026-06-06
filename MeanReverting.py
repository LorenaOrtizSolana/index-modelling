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
            if isinstance(current_vix, pd.Series):
                current_vix = current_vix.iloc[0]
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


def weight_mean_reversion_scaled(selected_tickers, prices_as_of, rebalance_date,
                                 prices, window=20, z_threshold=2.0,
                                 max_weight=0.20, target_vol=0.10):
    if len(selected_tickers) == 0:
        return {}

    historical = prices[prices.index <= rebalance_date]

    if len(historical) < window:
        return {ticker: 1.0 / len(selected_tickers) for ticker in selected_tickers}

    rolling_mean = historical[selected_tickers].rolling(window=window).mean()
    rolling_std = historical[selected_tickers].rolling(window=window).std()

    latest_mean = rolling_mean.iloc[-1]
    latest_std = rolling_std.iloc[-1]
    current_prices = historical.iloc[-1][selected_tickers]

    z_scores = (current_prices - latest_mean) / latest_std

    raw_weights = {}

    for ticker in selected_tickers:
        z = z_scores[ticker]

        if z < -z_threshold:
            strength = abs(z) - z_threshold
            strength = min(strength, 3.0)
            raw_weights[ticker] = 1.0 + strength

        elif z > z_threshold:
            strength = abs(z) - z_threshold
            strength = min(strength, 3.0)
            raw_weights[ticker] = -(1.0 + strength)

    if len(raw_weights) == 0:
        return {}

    capped_weights = {}
    for ticker, weight in raw_weights.items():
        if weight > 0:
            capped_weights[ticker] = min(weight, max_weight)
        elif weight < 0:
            capped_weights[ticker] = max(weight, -max_weight)

    total_positive = sum(w for w in capped_weights.values() if w > 0)
    total_negative = abs(sum(w for w in capped_weights.values() if w < 0))

    normalized_weights = {}

    if total_positive > 0:
        scale_long = 0.5 / total_positive
        for ticker, weight in capped_weights.items():
            if weight > 0:
                normalized_weights[ticker] = weight * scale_long

    if total_negative > 0:
        scale_short = 0.5 / total_negative
        for ticker, weight in capped_weights.items():
            if weight < 0:
                normalized_weights[ticker] = weight * scale_short

    try:
        returns = historical[selected_tickers].pct_change().dropna()
        vol = returns.std() * np.sqrt(252)

        for ticker in normalized_weights:
            if ticker in vol.index and vol[ticker] > 0:
                vol_scaler = min(1.0, target_vol / vol[ticker])
                normalized_weights[ticker] *= vol_scaler

        total = sum(abs(w) for w in normalized_weights.values())
        if total > 0:
            for ticker in normalized_weights:
                normalized_weights[ticker] = normalized_weights[ticker] / total
    except:
        pass

    return normalized_weights


def run_mean_reversion_strategy(backtest_instance, vix_prices=None):
    def selection_with_timing(rebalance_date, prices_as_of, full_history):
        selected = select_mean_reversion_etfs(
            rebalance_date, prices_as_of, full_history,
            vix_prices=vix_prices
        )
        return selected

    def weighting_with_timing(selected_tickers, prices_as_of, rebalance_date):
        return weight_mean_reversion_scaled(
            selected_tickers, prices_as_of, rebalance_date,
            prices=backtest_instance.prices,
            window=20, z_threshold=2.0, max_weight=0.20
        )

    results = backtest_instance.run(selection_with_timing, weighting_with_timing)
    return results