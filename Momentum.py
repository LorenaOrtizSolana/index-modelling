import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def select_momentum(rebalance_date, prices_as_of, full_price_history,
                    lookback_months=12, skip_months=1,
                    top_pct=0.2, bottom_pct=0.2,
                    winner_std_filter=2.0,
                    min_price=5.0):
    historical = full_price_history[full_price_history.index <= rebalance_date]

    if len(historical) < 252:
        return []

    monthly_prices = historical.resample('M').last()
    monthly_returns = monthly_prices.pct_change()

    if len(monthly_returns) < lookback_months + skip_months + 1:
        return []

    momentum_returns = monthly_returns.iloc[-(lookback_months + skip_months):-skip_months].sum()

    current_prices = historical.iloc[-1]
    price_filter = current_prices >= min_price
    valid_tickers = price_filter[price_filter].index.tolist()

    momentum_returns = momentum_returns[momentum_returns.index.isin(valid_tickers)]

    if len(momentum_returns) == 0:
        return []

    momentum_sorted = momentum_returns.sort_values(ascending=False)
    n_stocks = len(momentum_sorted)
    n_top = max(1, int(n_stocks * top_pct))
    n_bottom = max(1, int(n_stocks * bottom_pct))

    winners = momentum_sorted.head(n_top).index.tolist()
    losers = momentum_sorted.tail(n_bottom).index.tolist()

    if winner_std_filter > 0 and len(winners) > 0:
        rolling_mean = historical.rolling(window=252).mean()
        rolling_std = historical.rolling(window=252).std()

        latest_mean = rolling_mean.iloc[-1]
        latest_std = rolling_std.iloc[-1]

        z_score = (latest_mean - current_prices) / latest_std

        filtered_winners = []
        for ticker in winners:
            if ticker not in z_score.index:
                filtered_winners.append(ticker)
            elif pd.isna(z_score[ticker]):
                filtered_winners.append(ticker)
            elif z_score[ticker] <= winner_std_filter:
                filtered_winners.append(ticker)

        winners = filtered_winners

    if len(losers) > 3:
        losers = losers[:-1]

    select_momentum.last_metadata = {
        'n_winners': len(winners),
        'n_losers': len(losers),
        'total_considered': n_stocks,
        'momentum_returns': momentum_returns,
        'rebalance_date': rebalance_date
    }

    return winners + losers


def weight_equal_momentum(selected_tickers, prices_as_of, rebalance_date,
                          long_pct=0.5, short_pct=0.5, max_weight=0.1):
    if len(selected_tickers) == 0:
        return {}

    if hasattr(select_momentum, 'last_metadata'):
        n_winners = select_momentum.last_metadata.get('n_winners', len(selected_tickers) // 2)
    else:
        n_winners = len(selected_tickers) // 2

    winners = selected_tickers[:n_winners]
    losers = selected_tickers[n_winners:]

    weights = {}

    if len(winners) > 0:
        weight_per_winner = min(long_pct / len(winners), max_weight)
        for ticker in winners:
            weights[ticker] = weight_per_winner

    if len(losers) > 0:
        weight_per_loser = min(short_pct / len(losers), max_weight)
        for ticker in losers:
            weights[ticker] = -weight_per_loser

    return weights