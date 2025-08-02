import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

def train_forecaster(data: np.ndarray, seq_len: int, lr: float, epochs: int, save_path: str):
    """
    data: numpy array shape (T, features), first feature is target.
    seq_len: length of input sequences.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(data[i + seq_len, 0])
    X = torch.tensor(np.stack(X), dtype=torch.float32)
    y = torch.tensor(y, dtype=torch.float32).unsqueeze(-1)
    ds = TensorDataset(X, y)
    dl = DataLoader(ds, batch_size=32, shuffle=True)
    model = LSTMForecaster(input_size=data.shape[1])
    opt = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        total, count = 0.0, 0
        for xb, yb in dl:
            pred = model(xb)
            loss = loss_fn(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item(); count += 1
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total/count:.4f}")
    torch.save(model.state_dict(), save_path)
    print(f"Saved forecaster to {save_path}")
