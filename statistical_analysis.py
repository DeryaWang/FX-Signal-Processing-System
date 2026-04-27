import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.arima.model import ARIMA

def perform_adf_test(series, name="Series"):
    """
    Augmented Dickey-Fuller (ADF) Test.
    Null Hypothesis (H0): The series has a unit root (is non-stationary).
    Alternate Hypothesis (H1): The series is stationary.
    """
    print(f"\n--- ADF Stationarity Test: {name} ---")
    result = adfuller(series.dropna())
    
    print(f'ADF Statistic: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    
    if result[1] < 0.05:
        print(f"Conclusion: {name} is STATIONARY (p < 0.05)")
    else:
        print(f"Conclusion: {name} is NON-STATIONARY (p >= 0.05)")
    return result[1]

def evaluate_model_complexity(series):
    """
    Uses AIC (Akaike Information Criterion) and BIC (Bayesian Information Criterion)
    to find the optimal ARMA model order. 
    This prevents overfitting by penalizing unnecessary parameters.
    """
    print("\n--- Model Selection (AIC/BIC) ---")
    best_aic = float("inf")
    best_order = None
    best_bic = float("inf")

    # We test a few ARMA(p, q) combinations
    for p in range(3):
        for q in range(3):
            if p == 0 and q == 0: continue
            try:
                # Fit an ARIMA(p, 0, q) model which is equivalent to ARMA(p, q)
                model = ARIMA(series, order=(p, 0, q))
                results = model.fit()
                
                print(f"Order ({p}, {q}): AIC={results.aic:.2f}, BIC={results.bic:.2f}")
                
                if results.aic < best_aic:
                    best_aic = results.aic
                    best_order = (p, q)
                if results.bic < best_bic:
                    best_bic = results.bic
            except:
                continue
    
    print(f"\nOptimal Model Order by AIC: {best_order}")
    return best_order

from arch import arch_model

def fit_garch_model(series):
    """
    Fits a GARCH(1,1) model to the return series.
    Returns: (Alpha, Beta, Conditional Volatility Series, Next-day Forecast)
    """
    # Multiply by 100 for numerical stability in GARCH estimation
    scaled_series = series * 100
    
    model = arch_model(scaled_series, vol='Garch', p=1, q=1, dist='normal', rescale=False)
    results = model.fit(disp='off')
    
    alpha = results.params['alpha[1]']
    beta = results.params['beta[1]']
    
    # Conditional volatility (standard deviation)
    cond_vol = results.conditional_volatility / 100.0 # Scale back
    
    # Forecast next period
    forecast = results.forecast(horizon=1)
    next_vol = np.sqrt(forecast.variance.values[-1, 0]) / 100.0
    
    return alpha, beta, cond_vol, next_vol

if __name__ == "__main__":
    try:
        # Load the data filtered in the previous step
        data = pd.read_csv("~/Desktop/FX/filtered_data.csv")
        
        # 1. Test stationarity of the raw returns vs. the filtered trend
        perform_adf_test(data['Log_Returns'], name="Raw Log Returns")
        perform_adf_test(data['Filtered_Trend'], name="PF Extracted Trend")
        
        # 2. Use Information Criteria to select the best ARMA model for the trend
        # This validates the 'generalizability' mentioned in your resume.
        evaluate_model_complexity(data['Filtered_Trend'])
        
    except Exception as e:
        print(f"Error: {e}. Ensure particle_filter.py has run successfully.")
