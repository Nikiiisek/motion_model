import torch
import torch.nn as nn
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights


class MobileNetV3SmallBaseline(nn.Module):

    def __init__(self, num_classes: int, pretrained: bool = True):
        super().__init__()

        weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        backbone = mobilenet_v3_small(weights=weights)

        self.feature_extractor = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        feature_dim = 576
        self.classifier = nn.Linear(feature_dim, num_classes)

    def forward(self, x):

        b, t, c, h, w = x.shape

        x = x.view(b * t, c, h, w)          #(B*T, C, H, W)
        x = self.feature_extractor(x)       #(B*T, 576, H', W')
        x = self.pool(x)                    #(B*T, 576, 1, 1)
        x = x.flatten(1)                    #(B*T, 576)

        x = x.view(b, t, -1)                #(B, T, 576)
        x = x.mean(dim=1)                   #temporal pooling

        logits = self.classifier(x)         #(B, num_classes)

        return logits