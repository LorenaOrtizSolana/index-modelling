import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from datetime import timedelta

import pandas as pd
import numpy as np

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime


def select_momentum(rebalance_date, prices_as_of, full_price_history,
                    lookback_months=12, skip_months=1,
                    top_pct=0.2, bottom_pct=0.2,
                    winner_std_filter=2.0,
                    min_price=5.0,
                    min_holding_months=0,
                    entry_top_pct=None,
                    exit_bottom_pct=None):
    historical = full_price_history[full_price_history.index <= rebalance_date]

    if len(historical) < 252:
        return []

    lookback_days = (lookback_months + skip_months) * 21
    start_date = rebalance_date - timedelta(days=lookback_days)
    window = historical.loc[start_date:rebalance_date]

    if len(monthly_returns) < lookback_months + skip_months + 1:
        return []

    momentum_returns = (window.iloc[-1] / window.iloc[0]) - 1
    momentum_returns = momentum_returns.dropna()

    current_prices = historical.iloc[-1]
    price_filter = current_prices >= min_price
    valid_tickers = price_filter[price_filter].index.tolist()

    momentum_returns = momentum_returns[momentum_returns.index.isin(valid_tickers)]

    if len(momentum_returns) == 0:
        return []

    momentum_sorted = momentum_returns.sort_values(ascending=False)
    n_stocks = len(momentum_sorted)

    actual_top_pct = entry_top_pct if entry_top_pct is not None else top_pct
    actual_bottom_pct = exit_bottom_pct if exit_bottom_pct is not None else bottom_pct

    n_top = max(1, int(n_stocks * actual_top_pct))
    n_bottom = max(1, int(n_stocks * actual_bottom_pct))

    raw_winners = momentum_sorted.head(n_top).index.tolist()
    raw_losers = momentum_sorted.tail(n_bottom).index.tolist()

    if not hasattr(select_momentum, 'holding_months'):
        select_momentum.holding_months = {}
    if not hasattr(select_momentum, 'last_winners'):
        select_momentum.last_winners = []
    if not hasattr(select_momentum, 'last_losers'):
        select_momentum.last_losers = []

    for ticker in list(select_momentum.holding_months.keys()):
        select_momentum.holding_months[ticker] += 1

    for ticker in raw_winners + raw_losers:
        if ticker not in select_momentum.holding_months:
            select_momentum.holding_months[ticker] = 0

    if min_holding_months > 0:
        winners = []
        for ticker in raw_winners:
            holding = select_momentum.holding_months.get(ticker, 0)
            if holding <= min_holding_months:
                winners.append(ticker)
            elif ticker in select_momentum.last_winners:
                winners.append(ticker)

        losers = []
        for ticker in raw_losers:
            holding = select_momentum.holding_months.get(ticker, 0)
            if holding <= min_holding_months:
                losers.append(ticker)
            elif ticker in select_momentum.last_losers:
                losers.append(ticker)
    else:
        winners = raw_winners.copy()
        losers = raw_losers.copy()

    if entry_top_pct is not None and exit_bottom_pct is not None:
        prev_winners = select_momentum.last_winners
        prev_losers = select_momentum.last_losers

        final_winners = []
        for ticker in prev_winners:
            if ticker in momentum_sorted.index:
                rank = momentum_sorted.index.get_loc(ticker)
                pct_rank = rank / n_stocks
                if pct_rank <= exit_bottom_pct:
                    final_winners.append(ticker)

        for ticker in winners:
            if ticker not in final_winners and ticker in momentum_sorted.index:
                rank = momentum_sorted.index.get_loc(ticker)
                pct_rank = rank / n_stocks
                if pct_rank <= entry_top_pct:
                    final_winners.append(ticker)

        final_losers = []
        for ticker in prev_losers:
            if ticker in momentum_sorted.index:
                rank = momentum_sorted.index.get_loc(ticker)
                pct_rank = rank / n_stocks
                if pct_rank >= (1 - exit_bottom_pct):
                    final_losers.append(ticker)

        for ticker in losers:
            if ticker not in final_losers and ticker in momentum_sorted.index:
                rank = momentum_sorted.index.get_loc(ticker)
                pct_rank = rank / n_stocks
                if pct_rank >= (1 - entry_top_pct):
                    final_losers.append(ticker)

        winners = final_winners
        losers = final_losers

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

    select_momentum.last_winners = winners.copy()
    select_momentum.last_losers = losers.copy()

    current_held = set(winners + losers)
    for ticker in list(select_momentum.holding_months.keys()):
        if ticker not in current_held:
            del select_momentum.holding_months[ticker]

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
