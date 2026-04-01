import torch
import torch.nn as nn
from ConvLSTM_cell import ConvLSTMCell


class ConvLSTMManyToOne(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=32, num_classes=2):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),            # 112 -> 56
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.cell = ConvLSTMCell(32, hidden_channels, kernel_size=3)

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(hidden_channels, num_classes)

    def forward(self, x):
        batch, seq_len, channels, height, width = x.shape
        device = x.device

        first = self.stem(x[:, 0])
        _, stem_channels, stem_h, stem_w = first.shape

        h = torch.zeros(batch, self.cell.hidden_channels, stem_h, stem_w, device=device)
        c = torch.zeros(batch, self.cell.hidden_channels, stem_h, stem_w, device=device)

        h, c = self.cell(first, h, c)

        for t in range(1, seq_len):
            feat = self.stem(x[:, t])
            h, c = self.cell(feat, h, c)

        out = self.pool(h)
        out = out.flatten(1)
        out = self.classifier(out)

        return out