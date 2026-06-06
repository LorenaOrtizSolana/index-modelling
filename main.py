import pandas as pd
import numpy as np
import yfinance as yf
from Backtesting import Backtest
import Momentum
from Backtesting import Backtest
from Momentum import select_momentum, weight_equal_momentum
from MeanReverting import select_mean_reversion_etfs, weight_mean_reversion_scaled, run_mean_reversion_strategy
from RiskParity import select_risk_parity_assets, weight_risk_parity

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


def run_complete_portfolio_comparison(tickers, start_date, end_date):
    prices, flags = fetch_prices(tickers, start_date, end_date)

    vix = yf.download('^VIX', start=start_date, end=end_date, progress=False)['Close']

    bt_momentum = Backtest(prices, rebalance_freq='M', transaction_cost_bps=10)
    bt_meanrev = Backtest(prices, rebalance_freq='W', transaction_cost_bps=5)
    bt_riskparity = Backtest(prices, rebalance_freq='Q', transaction_cost_bps=5)

    results_momentum = bt_momentum.run(
        lambda d, p, hist: select_momentum(d, p, hist),
        lambda s, p, d: weight_equal_momentum(s, p, d)
    )

    results_meanrev = run_mean_reversion_strategy(bt_meanrev, vix)

    results_riskparity = bt_riskparity.run(
        lambda d, p, hist: select_risk_parity_assets(p, d, hist),
        lambda s, p, d: weight_risk_parity(s, p, d, bt_riskparity.prices)
    )

    comparison = pd.DataFrame({
        'Momentum': results_momentum['diagnostics'],
        'Mean Reversion': results_meanrev['diagnostics'],
        'Risk Parity (Inv Vol)': results_riskparity['diagnostics']
    }).T

    print("=" * 60)
    print("STRATEGY COMPARISON")
    print("=" * 60)
    print(comparison.round(4))

    return {
        'momentum': results_momentum,
        'mean_reversion': results_meanrev,
        'risk_parity': results_riskparity,
        'comparison': comparison
    }


if __name__ == "__main__":
    all_tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLV', 'TLT', 'GLD',
                   'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'JPM', 'V']

    results = run_complete_portfolio_comparison(
        tickers=all_tickers,
        start_date='2020-01-01',
        end_date='2024-12-31'
    )

    print("\n" + "=" * 60)
    print("DETAILED DIAGNOSTICS")
    print("=" * 60)

    for strategy, data in results.items():
        if strategy != 'comparison':
            print(f"\n{strategy.upper()}:")
            diag = data['diagnostics']
            print(f"  Total Return: {diag['total_return']:.2%}")
            print(f"  Sharpe Ratio: {diag['sharpe_ratio']:.2f}")
            print(f"  Max Drawdown: {diag['max_drawdown_pct']:.2%}")
            print(f"  Hit Rate: {diag['hit_rate']:.2%}")