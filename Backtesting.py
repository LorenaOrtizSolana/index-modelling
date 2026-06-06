import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

class Backtest:

    def __init__(self, prices, rebalance_freq='M', transaction_cost_bps=10):
        self.prices = prices
        self.rebalance_freq = rebalance_freq
        self.transaction_cost_bps = transaction_cost_bps
        self.cost_decimal = transaction_cost_bps / 10000
        self.tickers = prices.columns.tolist()

    def run(self, selection_func, weighting_func):
        rebalance_dates = self._get_rebalance_dates()

        daily_returns = pd.Series(index=self.prices.index, dtype=float)
        all_weights = pd.DataFrame(0.0, index=self.prices.index, columns=self.tickers)
        turnover_series = pd.Series(index=self.prices.index, dtype=float)

        prev_weights = pd.Series(0, index=self.tickers)

        for i, rebalance_date in enumerate(rebalance_dates):
            prices_as_of = self.prices.loc[:rebalance_date].iloc[-1]

            selected = selection_func(rebalance_date, prices_as_of, self.prices)

            weights = weighting_func(selected, prices_as_of, rebalance_date)

            weights_series = pd.Series(weights, index=self.tickers).fillna(0)

            if i == 0:
                turnover = 0
            else:
                turnover = (weights_series - prev_weights).abs().sum() / 2
            turnover_series[rebalance_date] = turnover

            cost_impact = turnover * self.cost_decimal

            next_date = rebalance_dates[i + 1] if i + 1 < len(rebalance_dates) else self.prices.index[-1]
            mask = (self.prices.index >= rebalance_date) & (self.prices.index < next_date)
            all_weights.loc[mask] = weights_series.values

            prev_weights = weights_series.copy()

        all_weights = all_weights.ffill()

        daily_returns = self._calculate_returns(all_weights)

        daily_returns -= turnover_series * self.cost_decimal

        diagnostics = self._calculate_diagnostics(daily_returns, turnover_series)

        return {
            'returns': daily_returns,
            'weights': all_weights,
            'turnover': turnover_series,
            'diagnostics': diagnostics
        }

    def _get_rebalance_dates(self):
        dates = self.prices.index
        if self.rebalance_freq == 'M':
            return dates[dates.to_series().apply(lambda x: x.day <= 7).values]
        elif self.rebalance_freq == 'Q':
            return dates[dates.to_series().apply(lambda x: x.month in [1, 4, 7, 10] and x.day <= 7).values]
        elif self.rebalance_freq == 'W':
            return dates[dates.dayofweek == 0]
        else:
            return dates

    def _calculate_returns(self, weights):
        stock_returns = self.prices.pct_change()
        shifted_weights = weights.shift(1)
        portfolio_returns = (shifted_weights * stock_returns).sum(axis=1)
        return portfolio_returns.fillna(0)

    def _calculate_diagnostics(self, returns, turnover):
        clean_returns = returns.dropna()
        clean_turnover = turnover.dropna()

        cumulative = (1 + clean_returns).cumprod()

        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        ann_return = (cumulative.iloc[-1] ** (252 / len(clean_returns))) - 1
        ann_vol = clean_returns.std() * np.sqrt(252)

        return {
            'total_return': cumulative.iloc[-1] - 1,
            'annualized_return': ann_return,
            'annualized_volatility': ann_vol,
            'sharpe_ratio': (ann_return - 0.02) / ann_vol if ann_vol > 0 else 0,
            'max_drawdown': drawdown.min(),
            'max_drawdown_pct': drawdown.min(),
            'annual_turnover': clean_turnover.sum() * (252 / len(clean_returns)),
            'hit_rate': (clean_returns > 0).mean(),
            'avg_daily_return': clean_returns.mean(),
            'median_daily_return': clean_returns.median(),
            'worst_day': clean_returns.min(),
            'best_day': clean_returns.max()
        }