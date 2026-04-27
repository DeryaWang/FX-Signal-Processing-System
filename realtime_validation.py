import pandas as pd
import numpy as np
import torch
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter
from lstm_engine import LSTMPredictor

def validate_with_lstm():
    print("🧠 === HYBRID PF-LSTM REAL-TIME PREDICTION === 🧠")
    
    tickers = ["EURUSD=X", "USDJPY=X", "GBPUSD=X"]
    results = []

    for ticker in tickers:
        print(f"\nProcessing {ticker}...")
        try:
            # 1. Data Prep
            raw = fetch_fx_data(ticker=ticker, period="3mo", interval="1h")
            df = preprocess_data(raw)
            
            # 2. Add Base Signal (PF) and Indicators
            df['PF_Signal'] = run_particle_filter(df['Log_Returns'].values)
            df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
            df['MA'] = df['Close'].rolling(window=10).mean()
            df['Z_Score'] = (df['Close'] - df['MA']) / (df['Close'].rolling(window=10).std() + 1e-9)
            df['Sentiment'] = np.random.normal(0, 0.1, len(df)) # Simulated for speed
            
            df.dropna(inplace=True)

            # 3. LSTM Training
            predictor = LSTMPredictor(seq_length=24)
            X, y = predictor.prepare_data(df)
            predictor.train_model(X, y, epochs=15)
            
            # 4. Predict Next
            last_features = ['Log_Returns', 'PF_Signal', 'Sentiment', 'Rolling_Vol', 'Z_Score']
            last_seq = df[last_features].tail(24).values
            up_prob = predictor.predict_next(last_seq)
            
            # Comparison with raw PF signal
            pf_pred = "UP" if df['PF_Signal'].iloc[-1] > 0 else "DOWN"
            lstm_pred = "UP" if up_prob > 0.5 else "DOWN"
            
            results.append({
                "Pair": ticker,
                "PF_Signal": pf_pred,
                "LSTM_Prob": f"{up_prob:.1%}",
                "FINAL_PRED": lstm_pred if abs(up_prob - 0.5) > 0.05 else "NEUTRAL"
            })
            
        except Exception as e:
            print(f"Error: {e}")

    # Display results
    print("\n" + "="*65)
    print(f"{'CURRENCY':<10} | {'PF SIGNAL':<10} | {'LSTM UP PROB':<15} | {'HYBRID PREDICTION'}")
    print("-" * 65)
    for r in results:
        print(f"{r['Pair']:<10} | {r['PF_Signal']:<10} | {r['LSTM_Prob']:<15} | {r['FINAL_PRED']}")
    print("="*65)

if __name__ == "__main__":
    validate_with_lstm()
