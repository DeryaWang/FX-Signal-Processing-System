import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class FXLSTMModel(nn.Module):
    """
    LSTM Architecture for FX Prediction.
    Takes multi-modal features and predicts next-bar direction.
    """
    def __init__(self, input_size=5, hidden_size=64, num_layers=2):
        super(FXLSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        # Take the output of the last time step
        out = self.fc(out[:, -1, :])
        return self.sigmoid(out)

class LSTMPredictor:
    def __init__(self, seq_length=24):
        self.seq_length = seq_length
        self.model = FXLSTMModel()
        self.scaler = MinMaxScaler()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.criterion = nn.BCELoss()

    def prepare_data(self, df):
        """
        Feature Engineering:
        Combines Log_Returns, PF_Signal, Sentiment, Volatility, and Carry.
        """
        # Ensure all columns exist, fill if missing
        features = ['Log_Returns', 'PF_Signal', 'Sentiment', 'Rolling_Vol', 'Z_Score']
        for f in features:
            if f not in df.columns: df[f] = 0
            
        data = df[features].values
        scaled_data = self.scaler.fit_transform(data)
        
        X, y = [], []
        for i in range(len(scaled_data) - self.seq_length):
            X.append(scaled_data[i : i + self.seq_length])
            # Target: 1 if next Log_Return > 0, else 0
            y.append(1 if df['Log_Returns'].iloc[i + self.seq_length] > 0 else 0)
            
        return torch.FloatTensor(np.array(X)), torch.FloatTensor(np.array(y)).view(-1, 1)

    def train_model(self, X, y, epochs=10):
        print(f"Training LSTM on {len(X)} sequences...")
        self.model.train()
        for epoch in range(epochs):
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.criterion(outputs, y)
            loss.backward()
            self.optimizer.step()
            if (epoch+1) % 5 == 0:
                print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    def predict_next(self, last_sequence):
        self.model.eval()
        with torch.no_grad():
            scaled_seq = self.scaler.transform(last_sequence)
            tensor_seq = torch.FloatTensor(scaled_seq).unsqueeze(0)
            prob = self.model(tensor_seq).item()
            return prob # Probability of price going UP

if __name__ == "__main__":
    # Quick sanity check
    test_predictor = LSTMPredictor()
    print("LSTM Engine Initialized.")
