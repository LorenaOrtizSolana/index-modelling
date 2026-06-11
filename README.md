# index-modelling
# Index Backtesting Framework

A production-ready backtesting framework for systematic index strategies with full audit capabilities. Built for quantitative index product development.

## Features

- **Generic Backtest Engine** – handles rebalancing, transaction costs, turnover calculation
- **Momentum Strategy** – configurable lookback periods, entry/exit buffers, minimum holding periods
- **Excel Implementation** – price-weighted (with split divisor adjustment), market-cap-weighted, equal-weighted (with drift formula)
- **Data Pipeline** – automatic handling of delisted stocks, holidays, missing data
- **Risk Metrics** – Sharpe ratio (2% risk-free), max drawdown, annualized volatility, hit rate, turnover

## Quick Start

```python
from Backtesting import Backtest
from Momentum import select_momentum, weight_equal_momentum

# Fetch price data
prices, flags = fetch_prices(tickers, '2020-01-01', '2024-12-31')

# Initialize backtest (monthly rebalancing, 10bps costs)
bt = Backtest(prices, rebalance_freq='M', transaction_cost_bps=10)

# Run momentum strategy
results = bt.run(
    lambda d, p, hist: select_momentum(d, p, hist, top_pct=0.3, lookback_days=252),
    lambda s, p, d: weight_equal_momentum(s, p, d)
)

# View diagnostics
print(results['diagnostics'])

