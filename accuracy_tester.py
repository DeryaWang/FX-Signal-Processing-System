import pandas as pd
import numpy as np
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter

def test_prediction_accuracy(ticker="EURUSD=X"):
    print(f"\n=== Testing Prediction Accuracy for {ticker} ===")
    
    # 1. Get Data (1 Year of 4H data for a large sample size)
    raw_data = fetch_fx_data(ticker=ticker, period="1y", interval="1h")
    df = preprocess_data(raw_data)
    
    # 2. Extract Particle Filter Signal
    returns = df['Log_Returns'].values
    print("Running Particle Filter...")
    trend_signal = run_particle_filter(returns)
    df['Signal'] = trend_signal
    
    # 3. Define the Target: Direction of the NEXT period
    # If Log_Returns at T+1 > 0, the direction is Up (1), else Down (0)
    df['Actual_Direction'] = np.where(df['Log_Returns'].shift(-1) > 0, 1, 0)
    
    # 4. Define the Prediction: Signal at T
    # If Signal at T > 0, we predict Up (1), else Down (0)
    df['Predicted_Direction'] = np.where(df['Signal'] > 0, 1, 0)
    
    # 5. Calculate Accuracy
    # Drop the last row because we don't know the future of the last point
    df.dropna(inplace=True)
    
    correct_predictions = (df['Actual_Direction'] == df['Predicted_Direction']).sum()
    total_predictions = len(df)
    accuracy = correct_predictions / total_predictions
    
    # 6. Confidence-Weighted Accuracy
    # Test accuracy only when the signal is "Strong" (Top 25% of absolute signal strength)
    threshold = df['Signal'].abs().quantile(0.75)
    strong_df = df[df['Signal'].abs() > threshold]
    strong_accuracy = (strong_df['Actual_Direction'] == strong_df['Predicted_Direction']).sum() / len(strong_df)

    print(f"\n--- Results for {ticker} ---")
    print(f"Total Samples Tested: {total_predictions}")
    print(f"Overall Prediction Accuracy: {accuracy:.2%}")
    print(f"Strong Signal Accuracy (Top 25% confidence): {strong_accuracy:.2%}")
    
    edge = accuracy - 0.50
    print(f"Edge over Random Guess: {edge:.2%}")
    
    if accuracy > 0.52:
        print("Conclusion: Your model has a STATISTICAL EDGE. It can predict market direction better than chance.")
    else:
        print("Conclusion: Your model is close to a random walk, which is common in highly efficient FX markets.")

if __name__ == "__main__":
    # Test on the two most liquid pairs
    for pair in ["EURUSD=X", "USDJPY=X"]:
        try:
            test_prediction_accuracy(pair)
        except Exception as e:
            print(f"Error testing {pair}: {e}")
