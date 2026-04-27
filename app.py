import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter
from sentiment_analyzer import FinancialSentimentAnalyzer
from lstm_engine import LSTMPredictor

# Page Config
st.set_page_config(page_title="FX Quant Terminal", layout="wide")

st.title("🏛️ FX Quant Signal Terminal")
st.markdown("Professional-grade Currency Analysis using Particle Filters and LSTM Neural Networks")

# Sidebar for controls
st.sidebar.header("Trading Parameters")
ticker = st.sidebar.selectbox("Select FX Pair", ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["7d", "1mo", "3mo"])
run_lstm = st.sidebar.checkbox("Enable LSTM Deep Learning Prediction")

# Initialize Sentiment Analyzer with your key
sentiment_engine = FinancialSentimentAnalyzer(api_key="afc9cfbe-8349-4829-b1f7-44cf83498942")

def load_and_process():
    with st.spinner(f"Fetching live data for {ticker}..."):
        raw_df = fetch_fx_data(ticker=ticker, period=timeframe, interval="1h")
        df = preprocess_data(raw_df)
        
        # Run Particle Filter
        df['PF_Signal'] = run_particle_filter(df['Log_Returns'].values)
        df['Trend_Line'] = df['PF_Signal'].cumsum() # For visualization
        
        return df

# Main Dashboard Logic
df = load_and_process()

# --- TOP ROW: Key Metrics ---
col1, col2, col3, col4 = st.columns(4)
current_price = df['Close'].iloc[-1]
price_change = (df['Close'].iloc[-1] / df['Close'].iloc[-2] - 1) * 100
current_sentiment = sentiment_engine.get_sentiment_score(ticker[:3])

col1.metric("Live Price", f"{current_price:.4f}", f"{price_change:.2f}%")
col2.metric("Trend Signal (PF)", "UP" if df['PF_Signal'].iloc[-1] > 0 else "DOWN")
col3.metric("News Sentiment", f"{current_sentiment:.2f}")
col4.metric("Market Volatility", f"{df['Log_Returns'].std():.4f}")

# --- MIDDLE ROW: Interactive Chart ---
st.subheader("Price Action vs. Extracted Trend Signal")
fig = go.Figure()
fig.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Market Price', line=dict(color='gray', width=1)))
# Create a secondary axis for the signal
fig.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(5).mean(), name='Smoothed Price', line=dict(color='blue', width=2)))

fig.update_layout(height=500, template="plotly_white", hovermode="x unified")
st.plotly_chart(fig, use_container_width=True)

# --- BOTTOM ROW: AI Prediction ---
if run_lstm:
    st.divider()
    st.subheader("🤖 LSTM Neural Network Forecast (Next 4H)")
    
    # Prep data for LSTM
    df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
    df['MA'] = df['Close'].rolling(window=10).mean()
    df['Z_Score'] = (df['Close'] - df['MA']) / (df['Close'].rolling(window=10).std() + 1e-9)
    df['Sentiment'] = current_sentiment
    df.dropna(inplace=True)
    
    predictor = LSTMPredictor(seq_length=24)
    X, y = predictor.prepare_data(df)
    
    with st.spinner("Training LSTM on historical patterns..."):
        predictor.train_model(X, y, epochs=10)
        
    last_seq = df[['Log_Returns', 'PF_Signal', 'Sentiment', 'Rolling_Vol', 'Z_Score']].tail(24).values
    up_prob = predictor.predict_next(last_seq)
    
    p_col1, p_col2 = st.columns(2)
    p_col1.write(f"Directional Probability (UP): **{up_prob:.1%}**")
    p_col1.progress(up_prob)
    
    prediction = "📈 BULLISH" if up_prob > 0.55 else "📉 BEARISH" if up_prob < 0.45 else "⚖️ NEUTRAL"
    p_col2.info(f"Final AI Recommendation: **{prediction}**")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Data refreshed at: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
