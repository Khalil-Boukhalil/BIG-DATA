import torch
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import torch.nn as nn

class LSTMModel(nn.Module):
    def __init__(self, hidden_size=64):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.dense = nn.Linear(hidden_size, 1)

    def forward(self, x):
        lstm_out, (hn, cn) = self.lstm(x)
        out = self.dense(lstm_out[:, -1, :])
        return out

model = LSTMModel(hidden_size=64)
model.load_state_dict(torch.load('lstm_model.pth'))
model.eval()

df = pd.read_csv("parking_data.csv")
scaler = MinMaxScaler()
df['free_spaces'] = scaler.fit_transform(df[['free_spaces']])

values = df['free_spaces'].values[-10:].reshape(1, 10, 1)

values_tensor = torch.tensor(values, dtype=torch.float32)

with torch.no_grad():
    prediction = model(values_tensor)

predicted_spaces = scaler.inverse_transform(prediction.numpy())[0][0]

print(f"📊 Predicted Free Spaces: {predicted_spaces}")
