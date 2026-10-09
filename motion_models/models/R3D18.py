import torch.nn as nn
from torchvision.models.video import r3d_18, R3D_18_Weights


class R3D18Classifier(nn.Module):

    def __init__(self, num_classes, pretrained=True):
        super().__init__()

        if pretrained:
            weights = R3D_18_Weights.KINETICS400_V1
        else:
            weights = None

        self.model = r3d_18(weights=weights)

        in_features = self.model.fc.in_features

        self.model.fc = nn.Linear(
            in_features,
            num_classes
        )

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        x = x.contiguous()
        return self.model(x)