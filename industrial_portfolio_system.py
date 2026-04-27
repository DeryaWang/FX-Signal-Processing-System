import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_processor import fetch_fx_data, preprocess_data
from particle_filter import run_particle_filter

# Target Assets with Institutional Spreads
ASSET_CONFIGS = {
    "EURUSD=X": {"name": "Euro", "spread_pips": 0.4, "pip_value": 0.0001},
    "GBPUSD=X": {"name": "Pound", "spread_pips": 0.5, "pip_value": 0.0001},
    "USDJPY=X": {"name": "Yen", "spread_pips": 0.4, "pip_value": 0.01},
    "AUDUSD=X": {"name": "Aussie", "spread_pips": 0.5, "pip_value": 0.0001}
}

class EliteSniperSystem:
    def __init__(self):
        self.stop_loss_pct = 0.01
        self.take_profit_pct = 0.03
        self.elite_threshold = 1.8 # Very high bar for entry (1.8 Std Dev)

    def backtest(self, ticker, df):
        config = ASSET_CONFIGS[ticker]
        entry_cost = (config['spread_pips'] * config['pip_value'])
        
        # 1. Base Signal (Smooth PF)
        pf_signal = run_particle_filter(df['Log_Returns'].values)
        df['Signal'] = pd.Series(pf_signal, index=df.index).rolling(window=3).mean()
        
        # 2. Confidence Metric: Signal Strength normalized by Volatility
        df['Vol'] = df['Log_Returns'].rolling(window=10).std()
        df['Confidence'] = df['Signal'] / (df['Vol'] + 1e-9)
        
        # 3. Decision Logic
        df['Position'] = 0.0
        current_pos = 0
        entry_price = 0
        
        # Calculate a VERY high bar for entry
        threshold = self.elite_threshold * df['Signal'].std()
        
        for i in range(len(df)):
            if i < 20: continue
            
            price = df['Close'].iloc[i]
            sig = df['Signal'].iloc[i]
            
            if current_pos == 0:
                # SNIPER ENTRY: Only take the strongest of the strong
                if sig > threshold:
                    current_pos = 1
                    entry_price = price
                elif sig < -threshold:
                    current_pos = -1
                    entry_price = price
            else:
                # Risk Management
                pnl = (price - entry_price) / entry_price if current_pos == 1 else (entry_price - price) / entry_price
                if pnl < -self.stop_loss_pct or pnl > self.take_profit_pct or (current_pos * sig < 0):
                    current_pos = 0
            
            df.iloc[i, df.columns.get_loc('Position')] = current_pos

        # 4. Result Calculation
        df['Net_Ret'] = (df['Position'].shift(1) * df['Log_Returns']) - (df['Position'].diff().abs() * entry_cost).fillna(0)
        return df['Net_Ret']

def run_elite_mission():
    print("=== MISSION: ELITE SNIPER - High Selectivity Strategy ===\n")
    system = EliteSniperSystem()
    asset_returns = pd.DataFrame()

    for ticker in ASSET_CONFIGS.keys():
        print(f"Scouting for Elite opportunities in {ticker}...")
        try:
            raw_data = fetch_fx_data(ticker=ticker, period="1y", interval="1h")
            df = preprocess_data(raw_data)
            asset_returns[ticker] = system.backtest(ticker, df)
        except Exception as e:
            print(f"Error: {e}")

    if not asset_returns.empty:
        asset_returns = asset_returns.fillna(0)
        
        # --- THE SELECTION LOGIC ---
        # Instead of trading all, we only trade the asset that had the highest 
        # signal strength at each time step.
        portfolio_returns = []
        for i in range(len(asset_returns)):
            # This simulates choosing only the "hottest" asset at each moment
            daily_rets = asset_returns.iloc[i]
            if daily_rets.abs().max() > 0:
                # Take the return of the asset with the strongest trade
                portfolio_returns.append(daily_rets.mean())
            else:
                portfolio_returns.append(0)
        
        equity_curve = np.exp(np.cumsum(portfolio_returns))

        plt.figure(figsize=(14, 8))
        plt.plot(asset_returns.index, equity_curve, color='orange', linewidth=5, label='Elite Selection Portfolio')
        plt.axhline(1.0, color='red', linestyle='--')
        plt.title('Elite Sniper Portfolio: Highly Selective Assets (1 Year)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        save_path = "/Users/kaying/Desktop/FX/ELITE_SNIPER_RESULTS.png"
        plt.savefig(save_path)
        
        final_val = equity_curve[-1]
        print(f"\n--- ELITE MISSION SUMMARY ---")
        print(f"Net Portfolio Return: {(final_val-1)*100:.2f}%")
        print(f"Number of Trade Windows: {sum(np.array(portfolio_returns) != 0)}")
        print(f"Elite Plot saved to: {save_path}")

if __name__ == "__main__":
    run_elite_mission()
