import matplotlib.pyplot as plt
import pandas as pd
from data_processor import fetch_fx_data, preprocess_data, analyze_distribution
from particle_filter import run_particle_filter
from statistical_analysis import perform_adf_test, evaluate_model_complexity

def main():
    """
    FX Signal Processing System - Integration Orchestrator
    This script runs the full pipeline from data acquisition to visualization.
    """
    print("=== FX Signal Processing System Initialized ===\n")

    # 1. Data Acquisition
    raw_df = fetch_fx_data(ticker="EURUSD=X", period="1mo", interval="1h")
    
    # 2. Preprocessing & Distribution Analysis
    processed_df = preprocess_data(raw_df)
    analyze_distribution(processed_df)

    # 3. Particle Filtering (Trend Extraction)
    returns = processed_df['Log_Returns'].values
    trend = run_particle_filter(returns)
    processed_df['Filtered_Trend'] = trend

    # 4. Statistical Validation
    perform_adf_test(processed_df['Log_Returns'], name="Raw Returns")
    perform_adf_test(processed_df['Filtered_Trend'], name="Extracted Trend")
    evaluate_model_complexity(processed_df['Filtered_Trend'])

    # 5. Visualization
    print("\nGenerating visualization...")
    plt.figure(figsize=(12, 8))

    # Plot 1: Log Returns vs Filtered Trend
    plt.subplot(2, 1, 1)
    plt.plot(processed_df.index, processed_df['Log_Returns'], 
             label='Observed Log Returns (Noisy)', color='gray', alpha=0.4)
    plt.plot(processed_df.index, processed_df['Filtered_Trend'], 
             label='PF Extracted Trend (Signal)', color='blue', linewidth=2)
    plt.title('EUR/USD: Raw Market Noise vs. Particle Filter Trend')
    plt.legend()
    plt.grid(True)

    # Plot 2: Cumulative Signal (To see the 'Price' path drift)
    plt.subplot(2, 1, 2)
    plt.plot(processed_df.index, processed_df['Log_Returns'].cumsum(), 
             label='Cumulative Market Return', color='black', alpha=0.5)
    plt.plot(processed_df.index, processed_df['Filtered_Trend'].cumsum(), 
             label='Cumulative Filtered Trend', color='red', linewidth=2)
    plt.title('Cumulative Performance: Market vs. Extracted Signal')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    
    # Save the plot to the Desktop folder
    plot_path = "/Users/kaying/Desktop/FX/signal_analysis_plot.png"
    plt.savefig(plot_path)
    print(f"\nSUCCESS: Plot saved to {plot_path}")
    print("\nYou can open 'signal_analysis_plot.png' on your desktop to see the results.")

if __name__ == "__main__":
    main()
