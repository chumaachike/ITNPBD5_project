import torch.nn as nn
from Building_Blocks import DilatedConvBlock, ConvBNReLU, DepthwiseConvBlock, BottleneckBlock, GDConvLayer

class MobileFaceNetHybrid(nn.Module):
    def __init__(self, embedding_size=128):
        super(MobileFaceNetHybrid, self).__init__()
        self.conv1 = ConvBNReLU(3, 64, kernel_size=3, stride=2, padding=1)
        self.dwconv2 = DepthwiseConvBlock(64, stride=1)

        self.blocks = nn.Sequential(
            BottleneckBlock(64, 64, expansion_factor=2, stride=2),  # Repeat 1 time
            DilatedConvBlock(64, 64),
            *[BottleneckBlock(64, 64, expansion_factor=2, stride=1) for _ in range(3)],  # Repeat 4 times
            BottleneckBlock(64, 128, expansion_factor=4, stride=2),
            *[BottleneckBlock(128, 128, expansion_factor=2, stride=1) for _ in range(6)],  # Repeat 6 times
            BottleneckBlock(128, 128, expansion_factor=4, stride=2),
            BottleneckBlock(128, 128, expansion_factor=2, stride=1),
            DilatedConvBlock(128, 128),
            ConvBNReLU(128, 512, kernel_size=1, stride=1, padding=0)
        )

        self.gdconv = GDConvLayer(512)
        self.fc = nn.Conv2d(512, embedding_size, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn = nn.BatchNorm2d(embedding_size)

    def forward(self, x):
        x = self.conv1(x)
        x = self.dwconv2(x)
        x = self.blocks(x)
        x = self.gdconv(x)
        x = self.fc(x)
        x = self.bn(x)
        x = x.view(x.size(0), -1)
        return x