# FX Signal Processing and Trading System

## Project Overview
This repository contains a modular quantitative trading system developed to extract underlying price trends from non-Gaussian high-frequency foreign exchange (FX) data. The system integrates advanced statistical signal processing with deep learning architectures to improve directional prediction accuracy and risk management.

## Key Components

### 1. Data Acquisition and Preprocessing (data_processor.py)
- Fetches real-world historical FX data from Yahoo Finance.
- Implements advanced cleaning: forward-filling time-series gaps and 5-sigma outlier removal.
- Performs log-return transformations and validates heavy-tailed (non-Gaussian) distributions via kurtosis analysis.

### 2. Signal Extraction Engine (particle_filter.py)
- Implements a Sequential Monte Carlo (Particle Filter) algorithm to estimate unobserved state variables.
- Utilizes the log-likelihood trick and adaptive resampling to maintain numerical stability in high-volatility environments.
- Extracts smoothed trend signals from noisy market observations.

### 3. Multi-Factor Fusion (multi_factor_engine.py)
- Integrates diverse alpha sources including price momentum, interest rate differentials (Carry), and simulated order flow imbalance.
- Dynamically fetches real-market interest rate proxies via bond yields using the yfinance API.

### 4. NLP Sentiment Analysis (sentiment_analyzer.py)
- Connects to The Guardian Open Platform API to retrieve real-time financial headlines.
- Employs the VADER (Valence Aware Dictionary and sEntiment Reasoner) engine to calculate sentiment polarity scores for fundamental filtering.

### 5. Deep Learning Engine (lstm_engine.py)
- Features a dual-layer Long Short-Term Memory (LSTM) neural network.
- Performs non-linear pattern recognition by training on fused features (PF signals, volatility, Z-scores, and sentiment).
- Achieved an out-of-sample directional accuracy of approximately 64% on EUR/USD in backtesting.

### 6. Industrial Backtesting Framework (industrial_portfolio_system.py)
- Simulates realistic institutional trading environments including spread costs and execution latency.
- Implements complex risk controls: ATR-based volatility scaling, strict stop-losses, and time-session filtering.

## Methodology and Validation
- Stationarity Validation: Augmented Dickey-Fuller (ADF) tests.
- Model Selection: Information Criteria (AIC/BIC) to ensure generalizability and prevent overfitting.
- Performance Metrics: Directional Hit Rate (Win Rate) and Net PnL after transaction costs.

## Technical Stack
- Python 3.13
- PyTorch (Deep Learning)
- Scikit-learn (Preprocessing)
- Statsmodels (Econometrics)
- Matplotlib (Visualization)
- Yfinance & Guardian API (Data Ingestion)
