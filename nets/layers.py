# -*- coding: utf-8 -*-
# @Time    : 2020/6/8 下午2:11
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : layers.py

import torch
import torch.nn.functional as F
import torch.nn as nn


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True, bn=False):
    conv = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                     padding=padding)

    out = [conv]

    if bn:
        out.append(nn.BatchNorm2d(out_channels, affine=False))
    if relu:
        out.append(nn.ReLU())

    return nn.Sequential(*out)


class SPPS(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, relu=True, bn=True):
        super(SPPS, self).__init__()

        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=2, padding=1,
                          relu=relu, bn=bn)
        self.conv2 = conv(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, stride=4, padding=1,
                          relu=relu, bn=bn)
        self.conv3 = conv(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, stride=8, padding=1,
                          relu=relu, bn=bn)

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)

        x2_upsample = F.interpolate(x2, scale_factor=2., mode="bilinear", align_corners=False)
        x3_upsample = F.interpolate(x3, scale_factor=4., mode="bilinear", align_corners=False)

        # print("x1: ", x1.shape)
        # print("x2: ", x2.shape)
        # print("x3: ", x3.shape)
        # print("x2_s: ", x2_upsample.shape)
        # print("x3_s: ", x3_upsample.shape)

        return torch.cat([x1, x2_upsample, x3_upsample], dim=1)


class SPP(nn.Module):
    def __init__(self, in_channels, out_channels, use_se=False, relu=True, bn=True):
        super(SPP, self).__init__()

        self.max1 = nn.MaxPool2d(kernel_size=2)
        self.max2 = nn.MaxPool2d(kernel_size=4)
        self.max3 = nn.MaxPool2d(kernel_size=8)

        self.conv1 = conv(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=3, stride=1, padding=1,
                          relu=relu, bn=bn)
        self.conv2 = conv(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, stride=1, padding=1,
                          relu=relu, bn=bn)
        self.conv3 = conv(in_channels=in_channels, out_channels=out_channels // 4, kernel_size=3, stride=1, padding=1,
                          relu=relu, bn=bn)

    def forward(self, x):
        x1 = self.conv1(self.max1(x))
        x2 = self.conv2(self.max2(x))
        x3 = self.conv3(self.max3(x))

        h, w = x1.size(2), x1.size(3)
        x2_upsample = F.interpolate(x2, size=(h, w), mode="bilinear", align_corners=False)
        x3_upsample = F.interpolate(x3, size=(h, w), mode="bilinear", align_corners=False)

        # print("x1: ", x1.shape)
        # print("x2: ", x2.shape)
        # print("x3: ", x3.shape)
        # print("x2_s: ", x2_upsample.shape)
        # print("x3_s: ", x3_upsample.shape)

        return torch.cat([x1, x2_upsample, x3_upsample], dim=1)


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, bias=True):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size,
                               stride=stride, padding=padding, dilation=dilation, groups=in_channels, bias=bias)

        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)

        return x


class Block(nn.Module):
    """
    Base block for XceptionA and DFANet.
    inputChannel: channels of inputs of the base block.
    outputChannel:channnels of outputs of the base block.
    stride: stride
    BatchNorm:
    """

    def __init__(self, inputChannel, outputChannel, stride=1, BatchNorm=nn.BatchNorm2d):
        super(Block, self).__init__()

        self.conv1 = nn.Sequential(SeparableConv2d(inputChannel, outputChannel // 4, stride=stride, ),
                                   BatchNorm(outputChannel // 4),
                                   nn.ReLU())
        self.conv2 = nn.Sequential(SeparableConv2d(outputChannel // 4, outputChannel // 4),
                                   BatchNorm(outputChannel // 4),
                                   nn.ReLU())
        self.conv3 = nn.Sequential(SeparableConv2d(outputChannel // 4, outputChannel),
                                   BatchNorm(outputChannel),
                                   nn.ReLU())
        self.projection = nn.Conv2d(inputChannel, outputChannel, 1, stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        identity = self.projection(x)
        return out + identity


class SEModule(nn.Module):
    """
    self attention model.

    """

    def __init__(self, in_channels, out_channels):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 1000, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1000, out_channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.ones((4, 16, 480, 640))
    net = SPP(in_channels=16, out_channels=32, relu=True, bn=True)
    print("net: ", net)

    y = net(x)
    print(y.size())
