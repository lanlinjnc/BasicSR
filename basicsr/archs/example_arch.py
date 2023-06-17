import math
import torch
import argparse
import torch.nn as nn
from torch.nn import functional as F

from basicsr.archs.arch_util import default_init_weights
from basicsr.utils.registry import ARCH_REGISTRY


@ARCH_REGISTRY.register()
class ExampleArch(nn.Module):
    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upscale=4):
        super(ExampleArch, self).__init__()

        self.upscale = upscale
        self.block1 = nn.Sequential(
            nn.Conv2d(num_in_ch, num_feat, kernel_size=3, padding=1, padding_mode='replicate'),
            nn.LeakyReLU(0.2)
        )
        self.resblock1 = ResidualBlock(num_feat)
        self.resblock2 = ResidualBlock(num_feat)
        self.resblock3 = ResidualBlock(num_feat)
        self.resblock4 = ResidualBlock(num_feat)
        self.resblock5 = ResidualBlock(num_feat)
        self.block7 = nn.Sequential(
            nn.Conv2d(num_feat, num_feat, kernel_size=3, padding=1, padding_mode='replicate'),
            # nn.BatchNorm2d(num_feat)
        )
        self.SA = SpatialAttention(kernel_size=3)
        self.block8 = nn.Conv2d(num_feat, num_out_ch, kernel_size=3, padding=1, padding_mode='replicate')

    def forward(self, x):
        # print(x.shape)
        x = F.interpolate(x, scale_factor=self.upscale, mode='bicubic', align_corners=False)
        # print(x.shape)
        block1 = self.block1(x)
        block2 = self.resblock1(block1)
        block3 = self.resblock2(block2)
        block4 = self.resblock3(block3)
        block5 = self.resblock4(block4)
        block6 = self.resblock5(block5)
        block7 = self.block7(block6)
        block7 = block7 * self.SA(block7)  # 将这个权值乘上原输入特征层
        block8 = self.block8(block1 + block7)

        return (torch.tanh(block8) + 1) / 2


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn1 = nn.BatchNorm2d(channels)
        self.lrelu = nn.LeakyReLU(0.2)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, padding_mode='replicate')
        self.bn2 = nn.BatchNorm2d(channels)
        self.SELayer = SELayer(channels, reduction=16)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.lrelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)
        residual = self.SELayer(residual)

        return x + residual


# 通道注意力机制，经典SE模块
class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):  # 传入输入通道数，缩放比例
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化高宽为1
        self.fc1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),  # 降维
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),  # 升维
            nn.Sigmoid())
        self.fc2 = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        # b,c,h.w
        b, c, _, _ = x.size()  # batch \channel\ high\ weight
        # b,c,1,1----> b,c
        y = self.avg_pool(x).view(b, c)  # 调整维度、去掉最后两个维度
        # b,c- ----> b,c/16 ---- >b,c ----> b,c,1,1
        y1 = self.fc1(y).view(b, c, 1, 1)  # 添加上h,w维度

        # b,c,1,1----> b,c
        z = self.avg_pool(x)  # 平均欧化
        # b,c- ----> b,c/16 ---- >b,c
        y2 = self.fc2(z)  # 降维、升维

        return x * y1.expand_as(x)  # 来扩展张量中某维数据的尺寸，将输入tensor的维度扩展为与指定tensor相同的size


# 空间注意力机制
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        return self.sigmoid(x)  # sigmoid激活操作


'''
class ExampleArch(nn.Module):
    """Example architecture.

    Args:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_out_ch (int): Channel number of outputs. Default: 3.
        num_feat (int): Channel number of intermediate features. Default: 64.
        upscale (int): Upsampling factor. Default: 4.
    """

    def __init__(self, num_in_ch=3, num_out_ch=3, num_feat=64, upscale=4):
        super(ExampleArch, self).__init__()
        self.upscale = upscale

        self.conv1 = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        self.upconv1 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.upconv2 = nn.Conv2d(num_feat, num_feat * 4, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)

        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)

        # activation function
        self.lrelu = nn.LeakyReLU(negative_slope=0.1, inplace=True)

        # initialization
        default_init_weights(
            [self.conv1, self.conv2, self.conv3, self.upconv1, self.upconv2, self.conv_hr, self.conv_last], 0.1)

    def forward(self, x):
        feat = self.lrelu(self.conv1(x))
        feat = self.lrelu(self.conv2(feat))
        feat = self.lrelu(self.conv3(feat))

        out = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        out = self.lrelu(self.pixel_shuffle(self.upconv2(out)))

        out = self.conv_last(self.lrelu(self.conv_hr(out)))
        base = F.interpolate(x, scale_factor=self.upscale, mode='bilinear', align_corners=False)
        out += base
        return out
'''
