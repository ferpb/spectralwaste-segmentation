import torch.nn as nn
import torch.nn.functional as F


# Basic modules

class DepthwiseSeparableConv2d(nn.Module):
    """
    Depthwise separable convolution

    1. It performs a spatial convolution independently for each input channel
    2. It performs a pointwise (1x1) convolution onto the output channels
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias)
        self.bn = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        x = self.pointwise(x)
        return x

class MultiDilationDepthwiseSeparableConv2d(nn.Module):
    """
    Multi-dilation depthwise separable convolution

    1. It performs two parallel depthwise convolutions, one with dilation rate 1 and another
       with dilation rate >= 1
    2. Their respective outputs are added
    3. A pointwise convolution is applied to the result
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=True):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding,       1 , in_channels, bias)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, in_channels, bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias)
        self.bn1 = nn.BatchNorm2d(in_channels, eps=1e-3)
        self.bn2 = nn.BatchNorm2d(in_channels, eps=1e-3)

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = F.relu(x1)
        x2 = self.conv2(x)
        x2 = self.bn2(x2)
        x2 = F.relu(x2)
        output = x1 + x2
        output = self.pointwise(output)
        return output

class Conv2dBatchNorm(nn.Module):
    """
    Combination of a convolutional layer and batch normalization
    """
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, dilation=1, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, 1, dilation, 1, bias)
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = F.relu(x)
        return x

# MiniNet modules

class MiniNetv2Downsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = Conv2dBatchNorm(in_channels, out_channels, 3, stride=2, dilation=1)

    def forward(self, x):
        output = self.conv(x)
        return output

class MiniNetv2Module(nn.Module):
    def __init__(self, in_channels, out_channels, dilation):
        super().__init__()
        self.conv = MultiDilationDepthwiseSeparableConv2d(in_channels, out_channels, 3, stride=1, padding='same', dilation=dilation)

    def forward(self, x):
        output = self.conv(x)
        return output

class MiniNetv2Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # self.conv = nn.ConvTranspose2d(in_channels, out_channels, 3, stride=2, padding=1, output_padding=1)
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2, padding=0)

    def forward(self, x):
        output = self.conv(x)
        return output

# MiniNet-v2 model

class MiniNetv2(nn.Module):
    """
    MiniNet-v2 network

    Note: Input width and height should be multiples of 8
    """
    def __init__(self, in_channels, out_channels, interpolate=True):
        super().__init__()

        self.interpolate = interpolate

        # 1. Downsample block
        self.d1 = MiniNetv2Downsample(in_channels, 16)
        self.d2 = MiniNetv2Downsample(16, 64)
        self.m_downsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(10)])
        self.d3 = MiniNetv2Downsample(64, 128)

        # 2. Feature extractor block
        rates = [1, 2, 1, 4, 1, 8, 1, 16, 1, 1, 1, 2, 1, 4, 1, 8]
        self.m_feature = nn.ModuleList([MiniNetv2Module(128, 128, rate) for rate in rates])

        # 3. Refinement block
        self.d4 = MiniNetv2Downsample(in_channels, 16)
        self.d5 = MiniNetv2Downsample(16, 64)

        # 4. Upsample block
        self.up1 = MiniNetv2Upsample(128, 64)
        self.m_upsample = nn.ModuleList([MiniNetv2Module(64, 64, 1) for i in range(4)])
        self.output = MiniNetv2Upsample(64, out_channels)

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        m_downsample = d2
        for m in self.m_downsample:
            m_downsample = m(m_downsample) # m1-m10
        d3 = self.d3(m_downsample)

        m_feature = d3
        for m in self.m_feature:
            m_feature = m(m_feature) # m10-m25

        d4 = self.d4(x)
        d5 = self.d5(d4)

        up1 = self.up1(m_feature)

        m_upsample = up1 + d5
        for m in self.m_upsample:
            m_upsample = m(m_upsample) # m26-m29

        output = self.output(m_upsample)

        if self.interpolate:
            output = F.interpolate(output, size=x.shape[2:], mode='bilinear', align_corners=True)

        return output
