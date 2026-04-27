import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter

def run_backtest(ticker="EURUSD=X"):
    """
    Performance Backtest:
    Uses the Particle Filter signals to trade on high-frequency FX data.
    """
    print(f"=== Starting Backtest for {ticker} ===")

    # 1. Fetch Fresh High-Frequency Data (Last 5 days, 5-minute intervals)
    # Note: 1m data is only available for the last 7 days in yfinance
    raw_data = fetch_fx_data(ticker=ticker, period="5d", interval="5m")
    df = preprocess_data(raw_data)

    # 2. Run Particle Filter to extract the trend
    returns = df['Log_Returns'].values
    print("Extracting signals via Particle Filter...")
    trend = run_particle_filter(returns)
    df['Signal_Trend'] = trend

    # 3. Generate Trading Strategy
    # Signal = 1 if trend is positive, -1 if trend is negative
    df['Position'] = np.where(df['Signal_Trend'] > 0, 1, -1)

    # IMPORTANT: Shift signals by 1 to avoid 'Look-ahead bias'
    # We trade at time T+1 based on the signal generated at time T
    df['Strategy_Returns'] = df['Position'].shift(1) * df['Log_Returns']

    # 4. Calculate Cumulative Performance
    df['Cum_Market_Returns'] = df['Log_Returns'].cumsum().apply(np.exp)
    df['Cum_Strategy_Returns'] = df['Strategy_Returns'].cumsum().apply(np.exp)

    # 5. Visualization
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df['Cum_Market_Returns'], label='Market (Buy & Hold)', color='black', alpha=0.5)
    plt.plot(df.index, df['Cum_Strategy_Returns'], label='PF Strategy', color='green', linewidth=2)
    
    plt.title(f'Backtest Results: Particle Filter Strategy on {ticker}')
    plt.xlabel('Time')
    plt.ylabel('Cumulative Return (Growth of $1)')
    plt.legend()
    plt.grid(True)
    
    plot_path = "/Users/kaying/Desktop/FX/backtest_results.png"
    plt.savefig(plot_path)
    
    # Calculate Final Metrics
    total_return = df['Cum_Strategy_Returns'].iloc[-1] - 1
    market_return = df['Cum_Market_Returns'].iloc[-1] - 1
    
    print(f"\n--- Backtest Metrics (5-Day Period) ---")
    print(f"Strategy Total Return: {total_return:.2%}")
    print(f"Market Total Return: {market_return:.2%}")
    print(f"Outperformance: {(total_return - market_return):.2%}")
    print(f"\nVisualization saved to: {plot_path}")

if __name__ == "__main__":
    try:
        run_backtest()
    except Exception as e:
        print(f"Backtest Error: {e}")
