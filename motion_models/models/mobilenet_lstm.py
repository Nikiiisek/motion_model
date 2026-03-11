import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3SmallLSTM(nn.Module):
    def __init__(
        self,
        num_classes: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.0,
        pretrained: bool = True,
    ):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.feature_dim = 576

        self.lstm = nn.LSTM(
            input_size=self.feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        x: (B, T, C, H, W)
        """
        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)
        x = self.feature_extractor(x)
        x = self.pool(x)
        x = x.flatten(1)

        x = x.view(b, t, self.feature_dim)

        lstm_out, (h_n, c_n) = self.lstm(x)

        #poslední hidden stav poslední LSTM vrstvy
        x = h_n[-1]

        logits = self.classifier(x)
        return logits