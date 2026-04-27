import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter
from lstm_engine import LSTMPredictor

def test_lstm_accuracy(ticker="EURUSD=X"):
    print(f"\n--- Deep Learning Accuracy Test: {ticker} ---")
    
    # 1. Fetch 1 Year of data
    raw = fetch_fx_data(ticker=ticker, period="1y", interval="1h")
    df = preprocess_data(raw)
    
    # 2. Engineering Features for LSTM
    print("Extracting features and PF signals...")
    df['PF_Signal'] = run_particle_filter(df['Log_Returns'].values)
    df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
    df['MA'] = df['Close'].rolling(window=10).mean()
    df['Z_Score'] = (df['Close'] - df['MA']) / (df['Close'].rolling(window=10).std() + 1e-9)
    df['Sentiment'] = np.random.normal(0, 0.05, len(df))
    df.dropna(inplace=True)

    # 3. Train/Test Split (80% Train, 20% Test)
    split = int(len(df) * 0.8)
    train_df = df.iloc[:split].copy()
    test_df = df.iloc[split:].copy()
    
    # 4. Training
    predictor = LSTMPredictor(seq_length=24)
    X_train, y_train = predictor.prepare_data(train_df)
    predictor.train_model(X_train, y_train, epochs=20)
    
    # 5. Testing (Out-of-Sample)
    print(f"Testing on {len(test_df)} unseen periods...")
    correct = 0
    total = 0
    
    features = ['Log_Returns', 'PF_Signal', 'Sentiment', 'Rolling_Vol', 'Z_Score']
    test_data = test_df[features].values
    
    for i in range(len(test_data) - 25):
        # Current sequence
        seq = test_data[i : i + 24]
        # Actual direction of the NEXT bar
        actual_return = test_df['Log_Returns'].iloc[i + 24]
        actual_dir = 1 if actual_return > 0 else 0
        
        # LSTM Prediction
        prob = predictor.predict_next(seq)
        pred_dir = 1 if prob > 0.5 else 0
        
        if pred_dir == actual_dir:
            correct += 1
        total += 1
        
    accuracy = correct / total if total > 0 else 0
    return accuracy

if __name__ == "__main__":
    results = {}
    for pair in ["EURUSD=X", "USDJPY=X"]:
        try:
            acc = test_lstm_accuracy(pair)
            results[pair] = acc
        except Exception as e:
            print(f"Error: {e}")
            
    print("\n" + "="*40)
    print(f"{'CURRENCY':<15} | {'LSTM OUT-OF-SAMPLE ACCURACY'}")
    print("-" * 40)
    for pair, acc in results.items():
        print(f"{pair:<15} | {acc:.2%}")
    print("="*40)
    print("Note: Random walk baseline is 50.00%")
