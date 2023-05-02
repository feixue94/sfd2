# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   sfd2 -> train
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   09/03/2023 16:11
=================================================='''

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class ResBlock(nn.Module):
    def __init__(self, inplanes, outplanes, stride=1, groups=32, dilation=1, norm_layer=None):
        super(ResBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self.conv1 = conv1x1(inplanes, outplanes)
        self.bn1 = norm_layer(outplanes)
        self.conv2 = conv3x3(outplanes, outplanes, stride, groups, dilation)
        self.bn2 = norm_layer(outplanes)
        self.conv3 = conv1x1(outplanes, outplanes)
        self.bn3 = norm_layer(outplanes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


def batch_normalization(channels, relu=False):
    if relu:
        return nn.Sequential(
            nn.BatchNorm2d(channels, affine=False, track_running_stats=True),
            nn.ReLU(), )
    else:
        return nn.Sequential(
            nn.BatchNorm2d(channels, affine=False, track_running_stats=True), )


def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, relu=True, use_bn=True, dilation=1):
    if not use_bn:
        if relu:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation)
            )
    else:
        if relu:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation),
                nn.BatchNorm2d(out_channels, affine=False),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                          padding=padding, dilation=dilation),
                nn.BatchNorm2d(out_channels, affine=False),
                # nn.ReLU(),
            )


class ResSegNet(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False, ms_detector=True):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability
        self.ms_detector = ms_detector

        d1, d2, d3, d4, d5, d6 = 64, 128, 256, 256, 256, 256
        self.conv1a = conv(in_channels=3, out_channels=d1, kernel_size=3, relu=True, use_bn=True)
        self.conv1b = conv(in_channels=d1, out_channels=d1, kernel_size=3, stride=2, relu=False, use_bn=False)
        self.bn1b = batch_normalization(channels=d1, relu=True)

        self.conv2a = conv(in_channels=d1, out_channels=d2, kernel_size=3, relu=True, use_bn=True)
        self.conv2b = conv(in_channels=d2, out_channels=d2, kernel_size=3, stride=2, relu=False, use_bn=False)
        self.bn2b = batch_normalization(channels=d2, relu=True)

        self.conv3a = conv(in_channels=d2, out_channels=d3, kernel_size=3, relu=True, use_bn=True)
        self.conv3b = conv(in_channels=d3, out_channels=d3, kernel_size=3, relu=False, use_bn=False)
        self.bn3b = batch_normalization(channels=d3, relu=True)

        self.conv4 = nn.Sequential(
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
        )

        self.convPa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.convDa = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)

        if self.require_stability:
            self.ConvSta = nn.Conv2d(256, 1, kernel_size=1)

    def det(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(out1a)
        out1c = self.bn1b(out1b)

        out2a = self.conv2a(out1c)
        out2b = self.conv2b(out2a)
        out2c = self.bn2b(out2b)

        out3a = self.conv3a(out2c)
        out3b = self.conv3b(out3a)
        out3c = self.bn3b(out3b)

        out4 = self.conv4(out3c)

        cPa = self.convPa(out4)
        semi = self.convPb(cPa)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(out4)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            # out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
            stability = torch.sigmoid(self.ConvSta(out4))
            stability = F.interpolate(stability, size=(x.shape[2], x.shape[3]), mode='bilinear')
        else:
            stability = None

        if self.require_feature and self.training:
            return score, stability, desc, (out1c, out2c, out3c)
        else:
            return score, stability, desc

    def det_train(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(out1a)
        out1c = self.bn1b(out1b)

        out2a = self.conv2a(out1c)
        out2b = self.conv2b(out2a)
        out2c = self.bn2b(out2b)

        out3a = self.conv3a(out2c)
        out3b = self.conv3b(out3a)
        out3c = self.bn3b(out3b)

        out4 = self.conv4(out3c)

        cPa = self.convPa(out4)
        semi = self.convPb(cPa)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(out4)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            # out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
            stability = torch.sigmoid(self.ConvSta(out4))
            stability = F.interpolate(stability, size=(x.shape[2], x.shape[3]), mode='bilinear')
        else:
            stability = None

        if self.require_feature and self.training:
            return score, semi_norm, stability, desc, (out2c, out3c)
        else:
            return score, semi_norm, stability, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, semi, stability, desc, seg_feats = self.det_train(x)
            return {
                "reliability": score,
                "score": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, semi, stability, desc = self.det_train(x)
            return {
                "reliability": score,
                "score": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
            }


class ResSegNetV2(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False, ms_detector=True):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability
        self.ms_detector = ms_detector

        d1, d2, d3, d4, d5, d6 = 64, 128, 256, 256, 256, 256
        self.conv1a = conv(in_channels=3, out_channels=d1, kernel_size=3, relu=True, use_bn=True)
        self.conv1b = conv(in_channels=d1, out_channels=d1, kernel_size=3, stride=2, relu=False, use_bn=False)
        self.bn1b = batch_normalization(channels=d1, relu=True)

        self.conv2a = conv(in_channels=d1, out_channels=d2, kernel_size=3, relu=True, use_bn=True)
        self.conv2b = conv(in_channels=d2, out_channels=d2, kernel_size=3, stride=2, relu=False, use_bn=False)
        self.bn2b = batch_normalization(channels=d2, relu=True)

        self.conv3a = conv(in_channels=d2, out_channels=d3, kernel_size=3, relu=True, use_bn=True)
        self.conv3b = conv(in_channels=d3, out_channels=d3, kernel_size=3, relu=False, use_bn=False)
        self.bn3b = batch_normalization(channels=d3, relu=True)

        self.conv4 = nn.Sequential(
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
            ResBlock(inplanes=256, outplanes=256, groups=32),
        )

        self.convPa = nn.Sequential(
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
        )
        self.convDa = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        )

        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)

        if self.require_stability:
            self.ConvSta = nn.Conv2d(256, 3, kernel_size=1)

    def cls_to_value(self, x):
        # 0 - 0.1, 1-0.5, 2-1.0
        cls = torch.max(x, dim=1, keepdim=True)[1]
        stab = torch.ones_like(cls).float()
        stab[cls == 0] = 0.1
        stab[cls == 1] = 0.5
        return stab

    def det(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(out1a)
        out1c = self.bn1b(out1b)

        out2a = self.conv2a(out1c)
        out2b = self.conv2b(out2a)
        out2c = self.bn2b(out2b)

        out3a = self.conv3a(out2c)
        out3b = self.conv3b(out3a)
        out3c = self.bn3b(out3b)

        out4 = self.conv4(out3c)

        cPa = self.convPa(out4)
        semi = self.convPb(cPa)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(out4)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            stability = self.ConvSta(out4)
            stability = F.interpolate(stability, size=(x.shape[2], x.shape[3]), mode='bilinear')
            stability = self.cls_to_value(x=stability)
        else:
            stability = None

        if self.require_feature and self.training:
            return score, stability, desc, (out1c, out2c, out3c)
        else:
            return score, stability, desc

    def det_train(self, x):
        out1a = self.conv1a(x)
        out1b = self.conv1b(out1a)
        out1c = self.bn1b(out1b)

        out2a = self.conv2a(out1c)
        out2b = self.conv2b(out2a)
        out2c = self.bn2b(out2b)

        out3a = self.conv3a(out2c)
        out3b = self.conv3b(out3a)
        out3c = self.bn3b(out3b)

        out4 = self.conv4(out3c)

        cPa = self.convPa(out4)
        semi = self.convPb(cPa)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(out4)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            # out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
            stability = self.ConvSta(out4)
            stability = F.interpolate(stability, size=(x.shape[2], x.shape[3]), mode='bilinear')
            score = score * self.cls_to_value(x=stability)

            stability = torch.softmax(stability, dim=1)
        else:
            stability = None

        if self.require_feature and self.training:
            return score, semi_norm, stability, desc, (out2c, out3c)
        else:
            return score, semi_norm, stability, desc

    def forward(self, batch):
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, semi, stability, desc, seg_feats = self.det_train(x)
            return {
                "reliability": score,
                "score": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, semi, stability, desc = self.det_train(x)
            return {
                "reliability": score,
                "score": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
            }
