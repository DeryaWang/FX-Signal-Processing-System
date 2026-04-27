import yfinance as yf
import pandas as pd
import numpy as np
from scipy.stats import kurtosis, skew

def fetch_fx_data(ticker="EURUSD=X", period="1mo", interval="1h"):
    """
    Step 1: Fetch real-world FX data from Yahoo Finance.
    """
    print(f"Fetching data for {ticker}...")
    data = yf.download(ticker, period=period, interval=interval)
    if data.empty:
        raise ValueError("No data found. Please check your ticker or connection.")
    
    # Flatten multi-index if necessary
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
        
    return data

def preprocess_data(df):
    """
    Step 2: Advanced Preprocessing with 4-Hour Resampling.
    """
    # Create a clean copy of only the Close column
    df_clean = pd.DataFrame(index=df.index)
    df_clean['Close'] = df['Close'].values
    
    # Ensure no zero or negative prices
    df_clean = df_clean[df_clean['Close'] > 0].copy()
    
    # Forward fill missing prices before resampling
    df_clean['Close'] = df_clean['Close'].ffill()
    
    # --- NEW: Resample to 4-Hour Frequency ---
    # 4H bars provide the best balance between trend capture and cost control.
    df_clean = df_clean.resample('4h').last().ffill()
    
    # Calculate Log Returns
    df_clean['Log_Returns'] = np.log(df_clean['Close'] / df_clean['Close'].shift(1))
    
    # Drop the first NaN
    df_clean.dropna(subset=['Log_Returns'], inplace=True)
    
    # Outlier Removal (5-sigma)
    std_returns = df_clean['Log_Returns'].std()
    mean_returns = df_clean['Log_Returns'].mean()
    outlier_cutoff = 5 * std_returns
    
    df_clean = df_clean[(df_clean['Log_Returns'] < mean_returns + outlier_cutoff) & 
                        (df_clean['Log_Returns'] > mean_returns - outlier_cutoff)]
    
    print(f"Preprocessing complete (4H Resampling). Remaining samples: {len(df_clean)}")
    return df_clean

def analyze_distribution(df):
    """
    Step 3: Statistical Distribution Analysis.
    """
    returns = df['Log_Returns']
    kurt = kurtosis(returns, fisher=False) 
    sk = skew(returns)
    
    print("\n--- Statistical Analysis ---")
    print(f"Mean Return: {returns.mean():.6f}")
    print(f"Volatility (Std): {returns.std():.6f}")
    print(f"Skewness: {sk:.4f}")
    print(f"Kurtosis: {kurt:.4f}")
    
    if kurt > 3.5:
        print("Conclusion: Data is Heavy-Tailed (Non-Gaussian).")
    return kurt, sk
