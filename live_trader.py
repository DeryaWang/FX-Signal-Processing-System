import pandas as pd
import numpy as np
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter
from multi_factor_engine import MultiFactorEngine
from sentiment_analyzer import FinancialSentimentAnalyzer

def generate_live_signals():
    print("🚀 === FX LIVE TRADING DECISION SYSTEM === 🚀")
    print("Time: Current Market Spot\n")

    # Initialize Engines
    mf_engine = MultiFactorEngine()
    # Using your Guardian API Key
    sentiment_analyzer = FinancialSentimentAnalyzer(api_key="afc9cfbe-8349-4829-b1f7-44cf83498942")
    
    tickers = {
        "EURUSD=X": "Euro",
        "GBPUSD=X": "Pound",
        "USDJPY=X": "Yen",
        "AUDUSD=X": "Aussie"
    }

    recommendations = []

    for ticker, name in tickers.items():
        print(f"\n--- Analyzing {name} ({ticker}) ---")
        try:
            # 1. Fetch the most recent data (last 7 days to seed the filter)
            raw_data = fetch_fx_data(ticker=ticker, period="7d", interval="1h")
            df = preprocess_data(raw_data)
            
            # 2. Get Current Price & Trend
            pf_signals = run_particle_filter(df['Log_Returns'].values)
            current_trend = pf_signals[-1]
            
            # 3. Get Real-time Sentiment
            current_sentiment = sentiment_analyzer.get_sentiment_score(name)
            
            # 4. Get Macro/Carry Factor
            base, quote = ticker[:3], ticker[3:6]
            carry = mf_engine.get_carry_factor(base, quote)
            
            # 5. Final Decision Logic
            decision = "WAIT (Neutral)"
            confidence = "Low"
            
            # Thresholds
            trend_threshold = 0.00005 
            
            if current_trend > trend_threshold and current_sentiment > 0.05:
                decision = "🚀 BUY (Long)"
                confidence = "High" if carry > 0 else "Medium"
            elif current_trend < -trend_threshold and current_sentiment < -0.05:
                decision = "📉 SELL (Short)"
                confidence = "High" if carry < 0 else "Medium"
            
            recommendations.append({
                "Asset": name,
                "Price": df['Close'].iloc[-1],
                "Trend": "UP" if current_trend > 0 else "DOWN",
                "Sentiment": f"{current_sentiment:.2f}",
                "Carry": f"{carry:.4f}",
                "DECISION": decision,
                "Conf.": confidence
            })
            
        except Exception as e:
            print(f"Error analyzing {name}: {e}")

    # Display Executive Summary
    print("\n" + "="*75)
    print(f"{'ASSET':<10} | {'PRICE':<10} | {'TREND':<6} | {'SENT':<6} | {'DECISION':<18} | {'CONF'}")
    print("-" * 75)
    for r in recommendations:
        print(f"{r['Asset']:<10} | {r['Price']:<10.4f} | {r['Trend']:<6} | {r['Sentiment']:<6} | {r['DECISION']:<18} | {r['Conf.']}")
    print("="*75)
    print("\nNote: Real-time signals generated via Particle Filter & Guardian NLP.")

if __name__ == "__main__":
    generate_live_signals()
