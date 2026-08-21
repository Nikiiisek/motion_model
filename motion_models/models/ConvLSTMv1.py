import torch
import torch.nn as nn

from .ConvLSTM_cell import ConvLSTMCell


class ConvLSTM(nn.Module):
    def __init__(
        self,
        in_channels=3,
        hidden_channels=16,
        kernel_size=3,
        num_classes=2
    ):
        super().__init__()

        self.hidden_channels = hidden_channels

        self.cell = ConvLSTMCell(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            kernel_size=kernel_size
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(
            hidden_channels,
            num_classes
        )

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """

        batch, seq_len, channels, height, width = x.shape

        h = torch.zeros(
            batch,
            self.hidden_channels,
            height,
            width,
            device=x.device,
            dtype=x.dtype
        )

        c = torch.zeros(
            batch,
            self.hidden_channels,
            height,
            width,
            device=x.device,
            dtype=x.dtype
        )

        for t in range(seq_len):
            h, c = self.cell(x[:, t], h, c)

        # klasifikace z posledního hidden state
        x = self.pool(h)
        x = x.flatten(1)

        logits = self.classifier(x)

        return logits