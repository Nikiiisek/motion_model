import torch
import torch.nn as nn
from ConvLSTM_cell import ConvLSTMCell


torch.manual_seed(42)
batch = 1
seq_len = 3
channels = 1
height = 2
width = 2
hidden_channels = 2

x = torch.randn(batch, seq_len, channels, height, width)
print("Celý vstup")
print(x)

#vrstva
class ConvLSTM(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

#tisk průchodu časem - krok po kroku
    def forward(self, x):
        batch, seq_len, channels, height, width = x.shape

        h = torch.zeros(batch, self.hidden_channels, height, width)
        c = torch.zeros(batch, self.hidden_channels, height, width)
        outputs = []

        for t in range(seq_len):
            print(f"\n--- Krok {t} ---")
            print("Vstup frame:", x[:, t].shape)

            h, c = self.cell(x[:, t], h, c)

            print("Hidden hodnoty:")
            print(h)
            print("Cell state:", c.shape)

            outputs.append(h)

        outputs = torch.stack(outputs, dim=1)
        print("\nFinal stacked output:", outputs.shape)
        return outputs

model = ConvLSTM(in_channels=channels,
                 hidden_channels=hidden_channels,
                 kernel_size=3)

output = model(x)

print("Výstup", output.shape)

