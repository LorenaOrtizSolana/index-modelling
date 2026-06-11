import pandas as pd
import numpy as np
import yfinance as yf
from Backtesting import Backtest
from Momentum import select_momentum, weight_equal_momentum


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
                       progress=False, auto_adjust=True)

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
        print(f"Removed {len(flags['stocks_with_no_data'])} tickers with no data")

    if len(prices.columns) == 0:
        print("ERROR: No valid tickers found")
        return pd.DataFrame(), flags

    print(f"Extracted prices: {len(prices)} days, {len(prices.columns)} stocks")

    missing_by_date = prices.isnull().sum(axis=1)
    holiday_mask = missing_by_date > (len(prices.columns) * 0.7)
    holiday_dates = prices.index[holiday_mask].tolist()
    flags['holidays_removed'] = holiday_dates
    prices = prices.loc[~holiday_mask]

    if holiday_dates:
        print(f"Removed {len(holiday_dates)} holiday dates")

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

        if last_values.isnull().sum() > len(last_values) * 0.7:
            flags['delisted_stocks'].append(ticker)
            prices = prices.drop(columns=ticker)

    if flags['delisted_stocks']:
        print(f"Removed {len(flags['delisted_stocks'])} delisted stocks")

    original_nulls = prices.isnull().sum().sum()
    prices = prices.ffill(limit=30)
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


def run_simple_parameter_sweep(prices):
    results = []

    rebalance_freqs = ['M', 'Q']
    min_holdings = [0, 3, 6]
    entry_buffers = [None, 0.25, 0.30]

    total = len(rebalance_freqs) * len(min_holdings) * len(entry_buffers)
    print(f"\nTesting {total} combinations...")

    combo = 0
    for rebalance_freq in rebalance_freqs:
        for min_hold in min_holdings:
            for entry_buffer in entry_buffers:
                combo += 1

                exit_buffer = entry_buffer + 0.10 if entry_buffer is not None else None

                bt = Backtest(prices, rebalance_freq=rebalance_freq, transaction_cost_bps=10)

                try:
                    res = bt.run(
                        lambda d, p, hist: select_momentum(
                            d, p, hist,
                            lookback_months=12,
                            top_pct=0.2,
                            min_holding_months=min_hold,
                            entry_top_pct=entry_buffer,
                            exit_bottom_pct=exit_buffer
                        ),
                        lambda s, p, d: weight_equal_momentum(s, p, d)
                    )

                    diag = res['diagnostics']
                    results.append({
                        'rebalance_freq': rebalance_freq,
                        'min_holding_months': min_hold,
                        'entry_buffer': entry_buffer if entry_buffer else 'none',
                        'exit_buffer': exit_buffer if exit_buffer else 'none',
                        'annual_return': diag['annualized_return'],
                        'annual_vol': diag['annualized_volatility'],
                        'sharpe': diag['sharpe_ratio'],
                        'max_drawdown': diag['max_drawdown_pct'],
                        'annual_turnover': diag['annual_turnover'],
                        'hit_rate': diag['hit_rate'],
                        'total_return': diag['total_return'],
                        'avg_daily_return': diag['avg_daily_return'],
                        'worst_day': diag['worst_day'],
                        'best_day': diag['best_day']
                    })

                    print(f"  {combo}/{total}: freq={rebalance_freq}, min_hold={min_hold}, "
                          f"entry={entry_buffer} -> Sharpe={diag['sharpe_ratio']:.2f}, "
                          f"Turnover={diag['annual_turnover']:.1%}")

                except Exception as e:
                    print(f"Error: {e}")
                    continue

    df = pd.DataFrame(results)

    print("\n" + "=" * 60)
    print("PARAMETER SWEEP RESULTS")
    print("=" * 60)

    best = df.loc[df['sharpe'].idxmax()]
    print(f"\nBEST SHARPE: {best['sharpe']:.2f}")
    print(f"   Rebalance: {best['rebalance_freq']}, Min Hold: {best['min_holding_months']}m, "
          f"Entry Buffer: {best['entry_buffer']}")
    print(f"   Turnover: {best['annual_turnover']:.1%}, Return: {best['annual_return']:.1%}, "
          f"Drawdown: {best['max_drawdown']:.1%}")

    lowest = df.loc[df['annual_turnover'].idxmin()]
    print(f"\nLOWEST TURNOVER: {lowest['annual_turnover']:.1%}")
    print(f"   Sharpe: {lowest['sharpe']:.2f}, Min Hold: {lowest['min_holding_months']}m")

    print("\nIMPACT OF MIN HOLDING PERIOD:")
    print(df.groupby('min_holding_months')[['annual_turnover', 'sharpe', 'annual_return']].mean().round(4))

    print("\nIMPACT OF ENTRY BUFFER:")
    print(df.groupby('entry_buffer')[['annual_turnover', 'sharpe']].mean().round(4))

    return df


if __name__ == "__main__":
    RUN_SWEEP = True

    tickers = ['SPY', 'QQQ', 'IWM', 'XLF', 'XLE', 'XLV', 'TLT',
               'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA']

    prices, flags = fetch_prices(tickers, '2020-01-01', '2024-12-31')

    if RUN_SWEEP:
        results_df = run_simple_parameter_sweep(prices)
        results_df.to_csv('momentum_simple_sweep.csv', index=False)

        print("\n" + "=" * 60)
        print("FULL DIAGNOSTICS FOR BEST COMBINATION")
        print("=" * 60)

        best = results_df.loc[results_df['sharpe'].idxmax()]
        print(f"\n  Rebalance Frequency:   {best['rebalance_freq']}")
        print(f"  Min Holding Months:    {best['min_holding_months']}")
        print(f"  Entry Buffer:          {best['entry_buffer']}")
        print(f"  Exit Buffer:           {best['exit_buffer']}")
        print(f"  Total Return:          {best['total_return']:.2%}")
        print(f"  Annualized Return:     {best['annual_return']:.2%}")
        print(f"  Annualized Volatility: {best['annual_vol']:.2%}")
        print(f"  Sharpe Ratio:          {best['sharpe']:.3f}")
        print(f"  Max Drawdown:          {best['max_drawdown']:.2%}")
        print(f"  Annual Turnover:       {best['annual_turnover']:.2%}")
        print(f"  Hit Rate:              {best['hit_rate']:.2%}")
        print(f"  Avg Daily Return:      {best['avg_daily_return']:.4%}")
        print(f"  Worst Day:             {best['worst_day']:.4%}")
        print(f"  Best Day:              {best['best_day']:.4%}")

        print("\nSaved to momentum_simple_sweep.csv")

    else:
        print("Running original comparison (requires MeanReverting, RiskParity modules)")