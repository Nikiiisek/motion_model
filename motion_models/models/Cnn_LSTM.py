import torch
import torch.nn as nn
from torchvision import models

class CnnLstmClassifier(nn.Module):
    def __init__(self, num_classes, hidden_size=128, freeze_cnn=True):
        super().__init__()

        cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        feat_dim = cnn.fc.in_features
        cnn.fc = nn.Identity()

        if freeze_cnn:
            for p in cnn.parameters():
                p.requires_grad = False

        self.cnn = cnn
        self.lstm = nn.LSTM(
            input_size=feat_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True
        )
        self.head = nn.Linear(hidden_size, num_classes)

    def forward(self, x, lengths):
        # x: B x T x C x H x W
        B, T, C, H, W = x.shape

        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)              # (B*T) x F
        feats = feats.view(B, T, -1)     # B x T x F

        packed = nn.utils.rnn.pack_padded_sequence(
            feats, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        _, (h_n, _) = self.lstm(packed)
        last = h_n[-1]                   # B x hidden

        logits = self.head(last)         # B x num_classes
        return logits
