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
num_classes = 3

x = torch.randn(batch, seq_len, channels, height, width)
print("Vstup")
print(x)
print("Tvar vstupu", x.shape)


class ConvLSTMManyToOne(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size, height, width, num_classes):
        super().__init__()

        self.hidden_channels = hidden_channels
        self.height = height
        self.width = width

        self.cell = ConvLSTMCell(in_channels, hidden_channels, kernel_size)

        self.classifier = nn.Linear(hidden_channels * height * width, num_classes)

    def forward(self, x):
        batch, seq_len, channels, height, width = x.shape

        h = torch.zeros(batch, self.hidden_channels, height, width)
        c = torch.zeros(batch, self.hidden_channels, height, width)

        for t in range(seq_len):
            print(f"\n--- Krok {t} ---")
            print("Frame vstup x[:, t] (tvar):", x[:, t].shape)
            print(x[:, t])

            h, c = self.cell(x[:, t], h, c)

            print("Hidden tvar:", h.shape)
            print("Hidden hodnoty:")
            print(h)

        print("\nPoslední hidden state (h):")
        print(h)

        flat = h.reshape(batch, -1)
        print("Flatten (tvar):", flat.shape)
        print(flat)

        out = self.classifier(flat)
        print("Logits (tvar):", out.shape)
        print(out)

        return out


model = ConvLSTMManyToOne(
    in_channels=channels,
    hidden_channels=hidden_channels,
    kernel_size=3,
    height=height,
    width=width,
    num_classes=num_classes
)

output = model(x)

print("Výstup", output.shape)