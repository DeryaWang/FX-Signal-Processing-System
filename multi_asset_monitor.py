import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter

def monitor_multiple_currencies():
    """
    Multi-Asset Monitoring System:
    Compares the extracted trends across EUR, GBP, JPY, and CNY.
    """
    # Mapping of symbols to friendly names
    tickers = {
        "EURUSD=X": "Euro",
        "GBPUSD=X": "British Pound",
        "USDJPY=X": "Japanese Yen",
        "USDCNY=X": "Chinese Yuan"
    }

    results = {}

    print("=== Multi-Asset FX Monitoring System ===")
    
    for ticker, name in tickers.items():
        print(f"\nProcessing {name} ({ticker})...")
        try:
            # 1. Fetch and Preprocess
            raw_data = fetch_fx_data(ticker=ticker, period="1mo", interval="1h")
            df = preprocess_data(raw_data)
            
            # 2. Extract Trend via Particle Filter
            returns = df['Log_Returns'].values
            trend = run_particle_filter(returns)
            
            # 3. Store the last 100 points for comparison
            results[name] = trend[-100:]
            print(f"Signal for {name}: {'UP' if trend[-1] > 0 else 'DOWN'} (Strength: {abs(trend[-1]):.6f})")
            
        except Exception as e:
            print(f"Error processing {name}: {e}")

    # 4. Comparative Visualization
    if results:
        print("\nGenerating Comparative Chart...")
        plt.figure(figsize=(14, 10))
        
        for i, (name, trend_data) in enumerate(results.items()):
            plt.subplot(len(results), 1, i+1)
            # Plot the zero line
            plt.axhline(0, color='black', linestyle='--', alpha=0.3)
            # Plot the trend
            color = 'green' if trend_data[-1] > 0 else 'red'
            plt.plot(trend_data, label=f'{name} Trend Signal', color=color, linewidth=1.5)
            plt.title(f'Extracted Trend Signal: {name}')
            plt.legend(loc='upper left')
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plot_path = "/Users/kaying/Desktop/FX/multi_asset_comparison.png"
        plt.savefig(plot_path)
        print(f"\nSUCCESS: Multi-asset comparison saved to {plot_path}")

if __name__ == "__main__":
    monitor_multiple_currencies()
