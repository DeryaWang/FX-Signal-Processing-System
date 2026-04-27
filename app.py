import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter
from sentiment_analyzer import FinancialSentimentAnalyzer
from lstm_engine import LSTMPredictor
from statistical_analysis import fit_garch_model

# Page Config
st.set_page_config(page_title="FX Quant Terminal", layout="wide")

st.title("🏛️ FX Quant Signal Terminal")
st.markdown("Professional-grade Currency Analysis using Particle Filters, GARCH(1,1), and LSTM")

# Sidebar
st.sidebar.header("Trading Parameters")
ticker = st.sidebar.selectbox("Select FX Pair", ["EURUSD=X", "USDJPY=X", "GBPUSD=X", "AUDUSD=X"])
timeframe = st.sidebar.selectbox("Select Timeframe", ["1mo", "3mo", "6mo"])
run_lstm = st.sidebar.checkbox("Enable LSTM AI Prediction")

sentiment_engine = FinancialSentimentAnalyzer(api_key="afc9cfbe-8349-4829-b1f7-44cf83498942")

def load_data():
    raw_df = fetch_fx_data(ticker=ticker, period=timeframe, interval="1h")
    df = preprocess_data(raw_df)
    df['PF_Signal'] = run_particle_filter(df['Log_Returns'].values)
    return df

df = load_data()

# --- METRICS ---
col1, col2, col3, col4 = st.columns(4)
alpha, beta, cond_vol, next_vol = fit_garch_model(df['Log_Returns'])

col1.metric("Live Price", f"{df['Close'].iloc[-1]:.4f}")
col2.metric("GARCH Next Vol", f"{next_vol:.4f}")
col3.metric("Alpha (Impact)", f"{alpha:.2f}")
col4.metric("Beta (Persistence)", f"{beta:.2f}")

# --- CHARTS ---
st.subheader("Volatility Analysis: GARCH(1,1) Estimated Risk")
fig_vol = go.Figure()
fig_vol.add_trace(go.Scatter(x=df.index, y=cond_vol, name='Conditional Volatility (GARCH)', line=dict(color='orange', width=2)))
fig_vol.update_layout(height=400, template="plotly_white", margin=dict(l=0, r=0, t=0, b=0))
st.plotly_chart(fig_vol, use_container_width=True)

st.subheader("Price Action vs. Particle Filter Signal")
fig_price = go.Figure()
fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'], name='Market Price', line=dict(color='gray', alpha=0.5)))
fig_price.add_trace(go.Scatter(x=df.index, y=df['Close'].rolling(5).mean(), name='Filtered Trend', line=dict(color='blue', width=2)))
fig_price.update_layout(height=400, template="plotly_white")
st.plotly_chart(fig_price, use_container_width=True)

# --- LSTM SECTION ---
if run_lstm:
    st.divider()
    st.subheader("🤖 LSTM Neural Network Forecast (Next 4H)")
    df['Rolling_Vol'] = df['Log_Returns'].rolling(window=10).std()
    df['MA'] = df['Close'].rolling(window=10).mean()
    df['Z_Score'] = (df['Close'] - df['MA']) / (df['Close'].rolling(window=10).std() + 1e-9)
    df['Sentiment'] = sentiment_engine.get_sentiment_score(ticker[:3])
    df.dropna(inplace=True)
    
    predictor = LSTMPredictor(seq_length=24)
    X, y = predictor.prepare_data(df)
    with st.spinner("Training LSTM..."):
        predictor.train_model(X, y, epochs=10)
    
    last_seq = df[['Log_Returns', 'PF_Signal', 'Sentiment', 'Rolling_Vol', 'Z_Score']].tail(24).values
    up_prob = predictor.predict_next(last_seq)
    
    p1, p2 = st.columns(2)
    p1.write(f"Directional Probability (UP): **{up_prob:.1%}**")
    p1.progress(up_prob)
    p2.info(f"AI Signal: **{'BULLISH' if up_prob > 0.55 else 'BEARISH' if up_prob < 0.45 else 'NEUTRAL'}**")

st.sidebar.markdown("---")
st.sidebar.markdown("**GARCH Params Explained:**")
st.sidebar.caption(f"Alpha ({alpha:.2f}): How quickly the model reacts to new market shocks.")
st.sidebar.caption(f"Beta ({beta:.2f}): How long market volatility persists after a shock.")
