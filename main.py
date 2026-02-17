import torch
import torch.nn as nn

#batch-počet sekvencí, seq_len - počet framů jedné sekvence, feature - dimenze?, která popisuje obraz
batch_size = 3
seq_len = 10
feature_dim = 5

#co si model pamatuje?
hidden_dim = 16


x = torch.randn(batch_size, seq_len, feature_dim)

#LSTM model
class SimpleLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_dim, feature_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out)
        return out

model = SimpleLSTM()

#inicializace hidden a cell state
h_t = torch.zeros(1, batch_size, hidden_dim)
c_t = torch.zeros(1, batch_size, hidden_dim)

print("Vstup:", x.shape)

#tisk sekvence krok po kroku
for t in range(seq_len):
    print(f"\n--- Krok {t} ---")

    x_t = x[:, t, :].unsqueeze(1)

    out, (h_t, c_t) = model.lstm(x_t, (h_t, c_t))

    print("Výstup tvar:", out.shape)
    print("Hodnoty:", h_t[0, 0, :5])


