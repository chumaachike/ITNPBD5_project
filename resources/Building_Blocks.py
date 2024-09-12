import torch
import torch.nn as nn
import torch.nn.functional as F

class DilatedConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DilatedConvBlock, self).__init__()
        base_channels = out_channels // 2
        self.dilated_conv_1 = nn.Conv2d(in_channels, base_channels, kernel_size=3, stride=1, padding=2, dilation=2, bias=False)
        self.dilated_conv_2 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=1, padding=9, dilation=3, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        out1 = self.dilated_conv_1(x)
        out2 = self.dilated_conv_2(x)
        x = torch.cat([out1, out2], dim=1)
        x = self.bn(x)
        return self.relu(x)

class ConvBNReLU(nn.Sequential):
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=1, groups=1):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_planes, out_planes, kernel_size, stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU6(inplace=True)
        )

class DepthwiseConvBlock(nn.Module):
    def __init__(self, in_channels, stride=1):
        super(DepthwiseConvBlock, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride):
        super(BottleneckBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expansion_factor != 1:
            layers.append(ConvBNReLU(in_channels, hidden_dim, kernel_size=1, stride=1, padding=0))
        layers.extend([
            DepthwiseConvBlock(hidden_dim, stride=stride),
            nn.Conv2d(hidden_dim, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class GDConvLayer(nn.Module):
    def __init__(self, in_channels):
        super(GDConvLayer, self).__init__()
        self.gdconv = nn.Conv2d(in_channels, in_channels, kernel_size=7, stride=1, padding=0, groups=in_channels, bias=False)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.gdconv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

