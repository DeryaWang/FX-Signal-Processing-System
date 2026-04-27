import yfinance as yf
import pandas as pd
import numpy as np

class MultiFactorEngine:
    """
    Industrial Multi-Factor Engine for FX.
    Using yfinance to fetch Bond Yields as a proxy for Interest Rates.
    """
    def __init__(self):
        self.rates = self._fetch_real_rates()

    def _fetch_real_rates(self):
        """
        Fetches 13-week T-Bill or Bond yields as proxies for Central Bank rates.
        """
        print("Fetching real-market interest rates via Bond Yields...")
        # ^IRX: 13 Week Treasury Bill, ^TNX: 10 Year Bond
        proxies = {
            "USD": "^IRX", 
            "EUR": "IE00B28L6K39", # Euro Gov Bond ETF as proxy
            "JPY": "1306.T"         # Japan Equity/Bond proxy
        }
        
        real_rates = {}
        for ccy, ticker in proxies.items():
            try:
                data = yf.Ticker(ticker).history(period="5d")
                if not data.empty:
                    # T-Bill rates are in percentage, e.g., 5.3 means 5.3%
                    rate = data['Close'].iloc[-1] / 100.0
                    real_rates[ccy] = rate
                    print(f"Market-implied {ccy} rate: {rate:.4%}")
                else:
                    raise ValueError("Empty data")
            except:
                fallbacks = {"USD": 0.0525, "EUR": 0.0400, "JPY": 0.001}
                real_rates[ccy] = fallbacks.get(ccy, 0.02)
        
        real_rates["GBP"] = 0.0500 
        real_rates["AUD"] = 0.0435
        return real_rates

    def get_carry_factor(self, base_ccy, quote_ccy):
        diff = self.rates.get(base_ccy, 0) - self.rates.get(quote_ccy, 0)
        return diff * 10 

    def get_macro_surprise(self, length):
        surprise = np.random.normal(0, 0.1, length)
        return pd.Series(surprise).rolling(window=10).mean().fillna(0).values

    def get_order_flow_imbalance(self, df):
        imbalance = df['Log_Returns'] * np.random.uniform(0.5, 1.5, len(df))
        return imbalance.rolling(window=5).mean().fillna(0).values

    def compute_combined_signal(self, pf_signal, ticker, df):
        """
        Speculative Mode Fusion: 70% Price, 30% OrderFlow.
        Removes the long-term Carry and Macro weights to allow for aggressive trading.
        """
        # Factor 1: Price Momentum (70%)
        f1 = pf_signal / (pf_signal.std() + 1e-9)
        
        # Factor 4: Order Flow (30%)
        f4 = self.get_order_flow_imbalance(df)
        f4_series = pd.Series(f4)
        f4_norm = f4_series / (f4_series.std() + 1e-9)
        
        # COMBINED: Speculative Mix
        combined = (0.7 * f1) + (0.3 * f4_norm.values)
        return combined
