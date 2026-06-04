import pandas as pd
import numpy as np
import yfinance as yf


def select_risk_parity_assets(prices, rebalance_date, full_price_history,
                              max_assets=5,
                              min_vol_difference=0.05,
                              max_correlation=0.5,
                              lookback_days=252):
    historical = full_price_history[full_price_history.index <= rebalance_date]

    if len(historical) < lookback_days:
        return []

    returns = historical.pct_change().dropna()
    if len(returns) < 60:
        return []

    volatility = returns.std() * np.sqrt(252)

    corr_matrix = returns.corr()

    vol_sorted = volatility.sort_values()

    selected = []

    for i in range(len(vol_sorted)):
        for j in range(i + 1, len(vol_sorted)):
            vol_diff = vol_sorted.iloc[j] - vol_sorted.iloc[i]

            if vol_diff >= min_vol_difference:
                candidate1 = vol_sorted.index[i]
                candidate2 = vol_sorted.index[j]

                if candidate1 in corr_matrix.columns and candidate2 in corr_matrix.columns:
                    corr = corr_matrix.loc[candidate1, candidate2]

                    if abs(corr) <= max_correlation:
                        if candidate1 not in selected:
                            selected.append(candidate1)
                        if candidate2 not in selected:
                            selected.append(candidate2)

        if len(selected) >= max_assets:
            break

    if len(selected) < 2:
        selected = ['SPY', 'TLT', 'GLD']

    select_risk_parity_assets.last_metadata = {
        'volatilities': volatility[selected],
        'correlations': corr_matrix.loc[selected, selected] if len(selected) > 0 else None,
        'rebalance_date': rebalance_date
    }

    return selected[:max_assets]


def weight_risk_parity(selected_tickers, prices_as_of, rebalance_date,
                       lookback_days=252, max_weight=0.4):
    if len(selected_tickers) == 0:
        return {}

    inv_vol = 1 / select_risk_parity_assets.last_metadata['volatilities']

    weights = inv_vol / inv_vol.sum()

    weights = weights.clip(upper=max_weight)
    weights = weights / weights.sum()

    return weights.to_dict()


def run_risk_parity(self):
    returns = self.prices.pct_change()
    volatility = returns.rolling(60).std() * np.sqrt(252)

    weights = 1 / volatility
    weights = weights.div(weights.sum(axis=1), axis=0)

    portfolio_returns = (weights.shift(1) * returns).sum(axis=1)

    return portfolio_returns