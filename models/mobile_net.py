import torch.nn.functional as F

from torch import nn

from .modules import Bottleneck, UpsampleConv


class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()

        # Conv Layers
        modules = [
            nn.Conv2d(3, 32, kernel_size=7, stride=1, padding=3, bias=False),
            nn.InstanceNorm2d(32, affine=True),
            nn.ReLU(inplace=True),
            Bottleneck(32, 64, kernel_size=3, stride=2),
            Bottleneck(64, 128, kernel_size=3, stride=2),
        ]

        # Residual layers
        for _ in range(5):
            modules.append(Bottleneck(128, 128,  3, 1))

        # Upsampling Layers
        modules += [
            UpsampleConv(128, 64, kernel_size=3, stride=1),
            UpsampleConv(64, 32, kernel_size=3, stride=1),
            nn.Conv2d(32, 3, kernel_size=7, stride=1, padding=3, bias=False)
        ]

        self.model = nn.Sequential(*modules)

    def forward(self, x):
        return self.model(x)
