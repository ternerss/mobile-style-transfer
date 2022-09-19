import torch.nn.functional as F

from torch import nn

from .modules import Bottleneck, UpsampleConv

class TransformerMobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        # Conv Layers
        self.reflection_pad = nn.ReflectionPad2d(9//2)
        self.conv1 = nn.Conv2d(3, 32, kernel_size=9, stride=1, bias=False)
        self.in1 = nn.InstanceNorm2d(32, affine=True)
        self.conv2 = Bottleneck(32, 64, kernel_size=3, stride=2)
        self.conv3 = Bottleneck(64, 128, kernel_size=3, stride=2)

        # Residual Layers
        self.res1 = Bottleneck(128, 128,  3, 1)
        self.res2 = Bottleneck(128, 128,  3, 1)
        self.res3 = Bottleneck(128, 128,  3, 1)
        self.res4 = Bottleneck(128, 128,  3, 1)
        self.res5 = Bottleneck(128, 128,  3, 1)

        # Upsampling Layers
        self.upconv1 = UpsampleConv(128, 64, kernel_size=3, stride=1)
        self.upconv2 = UpsampleConv(64, 32, kernel_size=3, stride=1)
        self.conv4 = nn.Conv2d(32, 3, kernel_size=9, stride=1, bias=False)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = F.relu(self.in1(self.conv1(out)))
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.res1(out)
        out = self.res2(out)
        out = self.res3(out)
        out = self.res4(out)
        out = self.res5(out)
        out = self.upconv1(out)
        out = self.upconv2(out)
        out = self.conv4(self.reflection_pad(out))
        return out
