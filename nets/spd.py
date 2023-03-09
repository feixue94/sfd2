# -*- coding: utf-8 -*-
# @Time    : 2020/5/12 下午5:00
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from nets.superpoint import SuperPointNet
from segmentation_models_pytorch.encoders import get_encoder

import datetime
import time


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


class SoftDetector(nn.Module):
    def __init__(self, N=8):
        super().__init__()
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)
        # self.maxpool = nn.MaxPool2d(N + 1, stride=1, padding=N // 2)
        # self.avgpool = nn.AvgPool2d(N + 1, stride=1, padding=N // 2)

    def forward(self, x):
        # x = self.preproc(x)
        # x = F.relu(x)
        alpha = F.softplus(x - self.preproc(x))  # spatial average
        alpha = alpha / (1 + alpha)
        beta = F.softplus(x - torch.mean(x, dim=1, keepdim=True))  # channel average
        beta = beta / (1 + beta)
        return torch.max(alpha * beta, dim=1, keepdim=True)[0]


class MultiScaleSoftDetector(nn.Module):
    def __init__(self, N=8):
        super().__init__()
        self.preproc = nn.AvgPool2d(3, stride=1, padding=1)

    def forward_one(self, x):
        alpha = F.softplus(x - self.preproc(x))  # spatial average
        alpha = alpha / (1 + alpha)
        beta = F.softplus(x - torch.mean(x, dim=1, keepdim=True))  # channel average
        beta = beta / (1 + beta)
        return torch.max(alpha * beta, dim=1, keepdim=True)[0]

    def forward(self, x: list, weights=[3. / 6, 2. / 6, 1. / 6]):
        '''
        if weights is not None, sum(weights) should be 1
        '''
        score = None
        for idx, v in enumerate(x):
            if score is None:
                score = self.forward_one(x=v) * weights[idx]
            else:
                score = score + self.forward_one(x=v) * weights[idx]
        return score


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


class L2Net(nn.Module):
    def __init__(self, outdim=128):
        super(L2Net, self).__init__()
        self.outdim = outdim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),

        )

        self.convD = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        self.convS = nn.Conv2d(128, 1, kernel_size=1, padding=0)

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        x = self.conv1(x)
        score = self.convS(x).squeeze()
        desc = self.convD(x)

        desc = F.normalize(desc, dim=1)

        desc1 = desc[:b, :, :, :]
        desc2 = desc[b:, :, :, :]

        score1 = score[:b, :, :]
        score2 = score[b:, :, :]

        return {
            'dense_features1': desc1,
            'scores1': score1,
            'dense_features2': desc2,
            'scores2': score2,
        }


class SPDL2Net(nn.Module):
    def __init__(self, outdim=128, use_bn=True):
        super(SPDL2Net, self).__init__()
        # c1, c2, c3, c4, c5 = 32, 64, 128, 128, 128
        c1, c2, c3, c4, c5 = 24, 48, 96, 128, 128

        self.conv = conv

        self.conv1a = self.conv(3, c1, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv1b = self.conv(c1, c1, kernel_size=3, stride=1, padding=1, use_bn=use_bn)

        self.conv2a = self.conv(c1, c2, kernel_size=3, stride=1, padding=1, use_bn=use_bn)
        self.conv2b = self.conv(c2, c2, kernel_size=3, stride=1, padding=2, dilation=2, use_bn=use_bn)

        self.conv3a = self.conv(c2, c3, kernel_size=3, stride=1, padding=2, dilation=2, use_bn=use_bn)
        self.conv3b = self.conv(c3, c3, kernel_size=3, stride=1, padding=4, dilation=4, use_bn=use_bn)

        self.conv4a = self.conv(c3, c4, kernel_size=2, stride=1, padding=2, dilation=4, use_bn=use_bn)
        self.conv4b = self.conv(c4, c4, kernel_size=2, stride=1, padding=4, dilation=8, use_bn=use_bn)
        # self.conv4c = nn.Conv2d(c4, c4, kernel_size=2, stride=1, padding=8, dilation=16)

        self.convDb = nn.Conv2d(c4, out_channels=outdim, kernel_size=1, stride=1, padding=0, dilation=1)

        self.convPb = nn.Conv2d(c4, 1, kernel_size=1, stride=1, padding=0, dilation=1)

    def det(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)

        x = self.conv2a(x)
        x = self.conv2b(x)

        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.conv4a(x)
        x = self.conv4b(x)
        # x = self.conv4c(x)

        score = self.convPb(x)
        desc = self.convDb(x)

        desc = F.normalize(desc, dim=1)

        # print(score.shape, desc.shape)

        return score, desc

    def forward(self, x):
        score, desc = self.det(x)
        return {
            "score": score,
            "desc": desc
        }


class SPD2L2Net(nn.Module):
    def __init__(self, outdim=128):
        super(SPD2L2Net, self).__init__()
        self.outdim = outdim
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),

        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        self.convPb = nn.Conv2d(128, 1, kernel_size=1, padding=0)

    def det(self, x):
        x = self.conv1(x)
        score = self.convPb(x).squeeze()
        score = torch.sigmoid(score)
        desc = self.convDb(x)
        desc = F.normalize(desc, dim=1)

        # print("desc: ", torch.norm(desc, dim=1))

        return score, desc

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        score, desc = self.det(x)

        return {
            "score": score,
            "desc": desc
        }

        desc1 = desc[:b, :, :, :]
        desc2 = desc[b:, :, :, :]

        score1 = score[:b, :, :]
        score2 = score[b:, :, :]

        return {
            'dense_features1': desc1,
            'scores1': score1,
            'dense_features2': desc2,
            'scores2': score2,
        }


class L2SegNetNB(nn.Module):
    def __init__(self, outdim=128, require_feature=False):
        super(L2SegNetNB, self).__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
        #               dilation=1),
        #     nn.BatchNorm2d(128, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
        #     nn.BatchNorm2d(128, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True), )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        self.convPb = nn.Conv2d(128, 1, kernel_size=1, padding=0)

    def det(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        score = self.convPb(out6).squeeze()
        score = torch.sigmoid(score)
        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_feature:
            return score, desc, (out1, out2, out3)
        else:
            return score, None, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, desc, seg_feats = self.det(x)
            return {
                "score": score,
                "desc": desc,
                "pred_feats": seg_feats,
            }

        else:
            score, desc = self.det(x)
            return {
                "score": score,
                "desc": desc,
            }

        desc1 = desc[:b, :, :, :]
        desc2 = desc[b:, :, :, :]

        score1 = score[:b, :, :]
        score2 = score[b:, :, :]

        return {
            'dense_features1': desc1,
            'scores1': score1,
            'dense_features2': desc2,
            'scores2': score2,
        }


class L2SegNetNB2(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        self.convPb = nn.Conv2d(128, 2, kernel_size=1, padding=0)

        if self.require_stability:
            self.ConvSta = nn.Conv2d(128, 1, kernel_size=1)

    def det(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        semi = self.convPb(out6)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + + .00001)
        score = semi_norm[:, :-1, :, :]

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            stability = self.convSta(out6)
            stability = torch.sigmoid(stability)

        if self.require_feature and self.training:
            return score, stability, desc, (out1, out2, out3)
        else:
            return score, None, desc

    def det_train(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        semi = self.convPb(out6)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + + .00001)
        score = semi_norm

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            stability = self.convSta(out6)
            stability = torch.sigmoid(stability)

        if self.require_feature and self.training:
            return score, semi_norm, stability, desc, (out1, out2, out3)
        else:
            return score, semi_norm, None, desc

    def forward(self, batch):
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, semi, stability, desc, seg_feats = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, semi, stability, desc = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
            }


class L2SegNetNB3(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True))
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(24, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(24, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True), )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)

        self.convPa = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1),
        )
        self.convPb = nn.Conv2d(128, 65, kernel_size=1, padding=0)

        if self.require_stability:
            self.convSta = nn.Conv2d(128, 1, kernel_size=1)

    def det(self, x):
        # out1 = self.conv1(x)
        # out2 = self.conv2(out1)
        out3 = self.conv3(x)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        semi = self.convPa(out5)
        semi = self.convPb(semi)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            stability = self.convSta(out6)
            stability = torch.sigmoid(stability)

        if self.require_feature and self.training:
            return score, stability, desc, (out3, out4)
        else:
            return score, None, desc

    def det_train(self, x):
        # out1 = self.conv1(x)
        # out2 = self.conv2(out1)
        out3 = self.conv3(x)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        semi = self.convPa(out5)
        semi = self.convPb(semi)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            stability = self.convSta(out6)
            stability = torch.sigmoid(stability)

        if self.require_feature and self.training:
            return score, semi_norm, stability, desc, (out3, out4)
        else:
            return score, semi_norm, None, desc

    def forward(self, batch):
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, semi, stability, desc, seg_feats = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, semi, stability, desc = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
            }


class L2SegNetNB4(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability

        # self.conv1 = nn.Sequential(
        #     nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(16, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True))
        #
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(24, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
        #     nn.BatchNorm2d(24, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True), )
        #
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=96, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm2d(96, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(96, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)

        self.convPa = nn.Sequential(
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(96, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=96, out_channels=96, kernel_size=3, stride=2, padding=1),
        )
        self.convPb = nn.Conv2d(96, 65, kernel_size=1, padding=0)

        if self.require_stability:
            self.convSta = nn.Conv2d(128, 1, kernel_size=1)

    def det(self, x):
        # out1 = self.conv1(x)
        # out2 = self.conv2(out1)
        out3 = self.conv3(x)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        semi = self.convPa(out5)
        semi = self.convPb(semi)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            stability = self.convSta(out6)
            stability = torch.sigmoid(stability)

        if self.require_feature and self.training:
            return score, stability, desc, (out3, out4)
        else:
            return score, None, desc

    def det_train(self, x):
        # out1 = self.conv1(x)
        # out2 = self.conv2(out1)
        out3 = self.conv3(x)
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        semi = self.convPa(out5)
        semi = self.convPb(semi)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_stability:
            stability = self.convSta(out6)
            stability = torch.sigmoid(stability)

        if self.require_feature and self.training:
            return score, semi_norm, stability, desc, (out3, out4)
        else:
            return score, semi_norm, None, desc

    def forward(self, batch):
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, semi, stability, desc, seg_feats = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, semi, stability, desc = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
            }


class L2SegNetNBD(nn.Module):
    def __init__(self, outdim=128, require_feature=False):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
        #               dilation=1),
        #     nn.BatchNorm2d(128, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
        #     nn.BatchNorm2d(128, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True), )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        self.convPb = nn.Conv2d(128, 1, kernel_size=1, padding=0)

    def det(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        score = self.convPb(out6).squeeze()
        score = torch.sigmoid(score)
        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1)

        if self.require_feature:
            return score, desc, (out1, out2, out3)
        else:
            return score, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, desc, seg_feats = self.det(x)
            return {
                "score": score,
                "desc": desc,
                "pred_feats": seg_feats,
            }

        else:
            score, desc = self.det(x)
            return {
                "score": score,
                "desc": desc,
            }

        desc1 = desc[:b, :, :, :]
        desc2 = desc[b:, :, :, :]

        score1 = score[:b, :, :]
        score2 = score[b:, :, :]

        return {
            'dense_features1': desc1,
            'scores1': score1,
            'dense_features2': desc2,
            'scores2': score2,
        }


class L2SegNetGNB(nn.Module):
    def __init__(self, outdim=128, require_feature=False):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        # self.conv5 = nn.Sequential(
        #     nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
        #               dilation=1),
        #     nn.BatchNorm2d(128, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True),
        #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
        #     nn.BatchNorm2d(128, affine=False, track_running_stats=True),
        #     nn.ReLU(inplace=True), )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=2,
                      dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=4, dilation=4, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        # self.convPb = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.clf = nn.Conv2d(128, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(128, 1, kernel_size=1)

        self.sta = nn.Conv2d(128, 1, kernel_size=1)

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors=F.normalize(x, p=2, dim=1),
                    repeatability=self.softmax(urepeatability),
                    reliability=self.softmax(ureliability))

    def det(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1, p=2)

        urel = self.softmax(self.clf(out6 ** 2))
        usta = torch.sigmoid(self.sta(out6 ** 2))
        urep = self.softmax(self.sal(out6 ** 2))

        if self.require_feature:
            return urel, usta, urep, desc, (out1, out2, out3)
        else:
            return urel, usta, urep, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            urel, usta, urep, desc, seg_feats = self.det(x)
            return {
                "reliability": urel,
                "repeatability": urep,
                "stability": usta,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            urel, usta, urep, desc = self.det(x)
            return {
                "reliability": urel,
                "repeatability": urep,
                "stability": usta,
                "desc": desc,
            }


class L2SegNetF(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                      dilation=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        # self.convPb = nn.Conv2d(128, 2, kernel_size=1, padding=0)
        self.convPb = SoftDetector()

        if self.require_stability:
            self.sta = nn.Conv2d(128, 1, kernel_size=1)

    def det(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        # out6 = F.upsample_bilinear(out6, size=(x.shape[2], x.shape[3]))

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1, p=2)

        score = self.convPb(out6)
        # print('score: ', torch.min(score), torch.median(score), torch.max(score))

        if self.require_stability:
            stability = torch.sigmoid(self.sta(out6))
        else:
            stability = None

        if self.require_feature and self.training:
            return score, stability, desc, (out1, out2, out3)
        else:
            return score, stability, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, stability, desc, seg_feats = self.det(x)
            return {
                "reliability": score,
                # "repeatability": ,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, stability, desc = self.det(x)
            return {
                "reliability": score,
                # "repeatability": urep,
                "stability": stability,
                "desc": desc,
            }


class L2SegNet(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True))

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=16, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(24, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv4 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            # nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1, dilation=1, groups=1),
            nn.BatchNorm2d(64, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )

        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1,
                      dilation=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=2, dilation=2, groups=1),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True), )

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        # self.convPb = nn.Conv2d(128, 2, kernel_size=1, padding=0)
        self.convPb = SoftDetector()

        if self.require_stability:
            self.sta = nn.Conv2d(128, 1, kernel_size=1)

    def det(self, x):
        out1 = self.conv1(x)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)

        out6 = F.upsample_bilinear(out6, size=(x.shape[2], x.shape[3]))

        desc = self.convDb(out6)
        desc = F.normalize(desc, dim=1, p=2)

        score = self.convPb(out6)
        # print('score: ', torch.min(score), torch.median(score), torch.max(score))

        if self.require_stability:
            stability = torch.sigmoid(self.sta(out6))
        else:
            stability = None

        if self.require_feature and self.training:
            return score, stability, desc, (out1, out2, out3)
        else:
            return score, stability, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, stability, desc, seg_feats = self.det(x)
            return {
                "reliability": score,
                # "repeatability": ,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, stability, desc = self.det(x)
            return {
                "reliability": score,
                # "repeatability": urep,
                "stability": stability,
                "desc": desc,
            }


class L2SegNetS(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False, ms_detector=True):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability
        self.ms_detector = ms_detector

        d1, d2, d3, d4, d5, d6 = 32, 64, 128, 128, 256, 256
        self.conv1a = conv(in_channels=3, out_channels=d1, kernel_size=3, relu=True, use_bn=True)
        self.conv1b = conv(in_channels=d1, out_channels=d1, kernel_size=3, relu=False, use_bn=False)
        self.bn1b = batch_normalization(channels=d1, relu=True)

        self.conv2a = conv(in_channels=d1, out_channels=d2, kernel_size=3, stride=2, relu=True, use_bn=True)
        self.conv2b = conv(in_channels=d2, out_channels=d2, kernel_size=3, relu=False, use_bn=False)
        self.bn2b = batch_normalization(channels=d2, relu=True)

        self.conv3a = conv(in_channels=d2, out_channels=d3, kernel_size=3, stride=2, relu=True, use_bn=True)
        self.conv3b = conv(in_channels=d3, out_channels=d3, kernel_size=3, relu=False, use_bn=False)
        self.bn3b = batch_normalization(channels=d3, relu=True)

        self.conv4a = conv(in_channels=d3, out_channels=d4, kernel_size=3, relu=True, use_bn=True)
        self.conv4b = conv(in_channels=d4, out_channels=d4, kernel_size=3, relu=True, use_bn=True)
        self.conv4c = conv(in_channels=d4, out_channels=d4, kernel_size=3, relu=False, use_bn=False)

        # if not self.ms_detector:
        #     self.convPb = SoftDetector()
        # else:
        #     self.convPb = MultiScaleSoftDetector()
        self.convDb = conv(in_channels=d4, out_channels=outdim, kernel_size=1, relu=False, use_bn=False)
        self.convPb = conv(in_channels=d4, out_channels=4 * 4 + 1, kernel_size=1, relu=False, use_bn=False)

        if self.require_stability:
            self.ConvSta = nn.Conv2d(d1 + d2 + d4, 1, kernel_size=1)

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

        out4a = self.conv4a(out3c)
        out4b = self.conv4b(out4a)
        out4c = self.conv4c(out4b)

        desc = self.convDb(out4c)
        desc = F.normalize(desc, dim=1, p=2)

        semi = self.convPb(out4c)
        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 4, 4)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 4, Wc * 4)
        '''
        if not self.ms_detector:
            score = self.convPb(out4c)
            if self.require_stability:
                out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

                stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
            else:
                stability = None
        else:
            out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # weights = [3. / 6, 2. / 6, 1. / 6]
            # score = self.convPb([out1b, out2b_up, out4c_up], weights=weights)

            if self.require_stability:
                stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
            else:
                stability = None
        '''

        # weights = [3. / 6, 2. / 6, 1. / 6]
        # score = self.convPb([out1b, out2b_up, out4c_up], weights=weights)
        if self.require_stability:
            out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
        else:
            stability = None

        if self.require_feature and self.training:
            return score, stability, desc, (out1c, out2c, out3c)
        else:
            return score, stability, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, stability, desc, seg_feats = self.det(x)
            return {
                "reliability": score,
                # "repeatability": ,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, stability, desc = self.det(x)
            return {
                "reliability": score,
                # "repeatability": urep,
                "stability": stability,
                "desc": desc,
            }


class L2SegNetT(nn.Module):
    def __init__(self, outdim=128, require_feature=False, require_stability=False, ms_detector=True):
        super().__init__()
        self.outdim = outdim
        self.require_feature = require_feature
        self.require_stability = require_stability
        self.ms_detector = ms_detector

        d1, d2, d3, d4, d5, d6 = 32, 64, 128, 128, 256, 256
        self.conv1a = conv(in_channels=3, out_channels=d1, kernel_size=3, relu=True, use_bn=True)
        self.conv1b = conv(in_channels=d1, out_channels=d1, kernel_size=3, relu=False, use_bn=False)
        self.bn1b = batch_normalization(channels=d1, relu=True)

        self.conv2a = conv(in_channels=d1, out_channels=d2, kernel_size=3, stride=2, relu=True, use_bn=True)
        self.conv2b = conv(in_channels=d2, out_channels=d2, kernel_size=3, relu=False, use_bn=False)
        self.bn2b = batch_normalization(channels=d2, relu=True)

        self.conv3a = conv(in_channels=d2, out_channels=d3, kernel_size=3, stride=2, relu=True, use_bn=True)
        self.conv3b = conv(in_channels=d3, out_channels=d3, kernel_size=3, relu=False, use_bn=False)
        self.bn3b = batch_normalization(channels=d3, relu=True)

        self.conv4a = conv(in_channels=d3, out_channels=d4, kernel_size=3, relu=True, use_bn=True)
        self.conv4b = conv(in_channels=d4, out_channels=d4, kernel_size=3, relu=True, use_bn=True)
        self.conv4c = conv(in_channels=d4, out_channels=d4, kernel_size=3, relu=False, use_bn=False)

        # if not self.ms_detector:
        #     self.convPb = SoftDetector()
        # else:
        #     self.convPb = MultiScaleSoftDetector()
        self.convDb = conv(in_channels=d4, out_channels=outdim, kernel_size=1, relu=False, use_bn=False)
        self.convPb = nn.Sequential(
            conv(in_channels=d3, out_channels=d3, kernel_size=3, stride=2, relu=True, use_bn=True),
            conv(in_channels=d3, out_channels=d3, kernel_size=3, relu=True, use_bn=True),
            conv(in_channels=d3, out_channels=65, kernel_size=1, padding=0, relu=False, use_bn=False),
        )

        if self.require_stability:
            self.ConvSta = nn.Conv2d(d1 + d2 + d4, 1, kernel_size=1)

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

        out4a = self.conv4a(out3c)
        out4b = self.conv4b(out4a)
        out4c = self.conv4c(out4b)

        desc = self.convDb(out4c)
        desc = F.normalize(desc, dim=1, p=2)

        semi = self.convPb(out3c)

        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)
        '''
        if not self.ms_detector:
            score = self.convPb(out4c)
            if self.require_stability:
                out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
                out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)

                stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
            else:
                stability = None
        else:
            out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            # weights = [3. / 6, 2. / 6, 1. / 6]
            # score = self.convPb([out1b, out2b_up, out4c_up], weights=weights)

            if self.require_stability:
                stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
            else:
                stability = None
        '''

        # weights = [3. / 6, 2. / 6, 1. / 6]
        # score = self.convPb([out1b, out2b_up, out4c_up], weights=weights)
        if self.require_stability:
            out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
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

        out4a = self.conv4a(out3c)
        out4b = self.conv4b(out4a)
        out4c = self.conv4c(out4b)

        desc = self.convDb(out4c)
        desc = F.normalize(desc, dim=1, p=2)

        semi = self.convPb(out3c)
        # print(x.shape, desc.shape, semi.shape)

        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), 1, Hc * 8, Wc * 8)
        # weights = [3. / 6, 2. / 6, 1. / 6]
        # score = self.convPb([out1b, out2b_up, out4c_up], weights=weights)
        if self.require_stability:
            out2b_up = F.interpolate(out2b, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            out4c_up = F.interpolate(out4c, size=(x.shape[2], x.shape[3]), mode='bilinear', align_corners=False)
            stability = torch.sigmoid(self.ConvSta(torch.cat([out1b, out2b_up, out4c_up], dim=1)))
        else:
            stability = None

        if self.require_feature and self.training:
            return score, semi_norm, stability, desc, (out1c, out2c, out3c)
        else:
            return score, semi_norm, stability, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        if self.require_feature:
            score, semi, stability, desc, seg_feats = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
                "pred_feats": seg_feats,
            }
        else:
            score, semi, stability, desc = self.det_train(x)
            return {
                "reliability": score,
                "semi": semi,
                "stability": stability,
                "desc": desc,
            }


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


class ResNetXN(nn.Module):
    def __init__(self, encoder_name='resnext101_32x4d', encoder_depth=2, encoder_weights='ssl', outdim=128,
                 freeze_encoder=False):
        super(ResNetXN, self).__init__()

        encoder = get_encoder(name=encoder_name,
                              in_channels=3,
                              depth=encoder_depth,
                              weights=encoder_weights)

        if encoder_depth == 3:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,  # 4x ds
                encoder.layer2,  # 8x ds
            )
            c = 512
            self.ds = 8

            self.conv = nn.Sequential(
                nn.Conv2d(c, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )

        elif encoder_depth == 2:
            self.encoder = nn.Sequential(
                encoder.conv1,
                encoder.bn1,  # 2x ds
                encoder.relu,
                encoder.maxpool,
                encoder.layer1,  # 4x ds
            )
            c = 256
            self.ds = 4

            self.conv = nn.Sequential(
                ResBlock(inplanes=256, outplanes=256, groups=32),
                ResBlock(inplanes=256, outplanes=256, groups=32),
                ResBlock(inplanes=256, outplanes=256, groups=32),
            )

            self.convPa = nn.Sequential(
                torch.nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                torch.nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(256),
                # nn.ReLU(inplace=True),
            )
            self.convDa = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
                # nn.BatchNorm2d(256),
                # nn.ReLU(inplace=True),
            )

        self.convPb = torch.nn.Conv2d(256, 65, kernel_size=1, stride=1, padding=0)
        self.convDb = torch.nn.Conv2d(256, outdim, kernel_size=1, stride=1, padding=0)

    def det(self, x):
        x = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(len(features))

        # pass through several CNN layers first
        x = self.conv(x)

        # Detector Head.
        cPa = self.convPa(x)
        semi = self.convPb(cPa)
        # print('semi: ', torch.min(semi), torch.median(semi), torch.max(semi))

        semi = torch.exp(semi)
        semi_norm = semi / (torch.sum(semi, dim=1, keepdim=True) + .00001)
        # print('semi: ', torch.min(semi), torch.median(semi), torch.max(semi))
        # print('semi_norm: ', torch.min(semi_norm), torch.median(semi_norm), torch.max(semi_norm))
        score = semi_norm[:, :-1, :, :]
        Hc, Wc = score.size(2), score.size(3)
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

        # Descriptor Head
        cDa = self.convDa(x)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        return score, None, desc

    def det_train(self, x):
        x = self.encoder(x)
        # print(features[0].shape, features[1].shape, features[2].shape)
        # print(features[3].shape)
        # print(features[4].shape)
        # print(len(features))

        # pass through several CNN layers first
        x = self.conv(x)

        # Detector Head.
        cPa = self.convPa(x)
        semi = self.convPb(cPa)
        # semi = torch.sigmoid(semi)

        Hc, Wc = semi.size(2), semi.size(3)

        semi_norm = torch.softmax(semi, dim=1)
        score = semi_norm[:, :-1, :, :]

        # recover resolution
        score = score.permute([0, 2, 3, 1])
        score = score.view(score.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8).unsqueeze(1)

        # Descriptor Head
        cDa = self.convDa(x)
        desc = self.convDb(cDa)
        desc = F.normalize(desc, dim=1)

        return score, desc, semi_norm

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        score, desc, semi = self.det_train(x)

        return {
            "score": score,
            "desc": desc,
            "semi": semi,
        }


if __name__ == '__main__':
    img = torch.ones((1, 1, 1024, 1024)).cuda()
    total_time = 0
    # net = SPD2NetSPP(outdim=128, use_bn=True).cuda()
    # net = SPD2Net32V3(outdim=32, use_bn=True).cuda()
    # net = SPDL2Net(outdim=32, use_bn=True).cuda()
    # net = L2SegNet(outdim=128).cuda().eval()
    # net = L2SegNetF(outdim=128).cuda().eval()
    # net = L2SegNetS(outdim=128, require_stability=True).cuda().eval()
    # net = L2SegNetT(outdim=128, require_stability=True).cuda().eval()
    # net = ResSegNet(outdim=128, require_stability=True).cuda().eval()
    # net = L2SegNetNB3(outdim=128, require_stability=True).cuda().eval()
    # net = ResSegNet(outdim=128, require_stability=True).cuda().eval()
    net = SuperPointNet().cuda().eval()

    # torch.save(net.state_dict(), "a.pkl")
    # exit(0)
    # net = SPD2Net32V2(outdim=128, use_bn=True).cuda()
    with torch.no_grad():
        torch.cuda.synchronize()
        start_time = time.time()
        for i in range(1000):
            out = net.det(img)
        total_time = time.time() - start_time
        torch.cuda.synchronize()

    print("mean time: ", total_time / 1000)
    print('Peak memory(MB): ', torch.cuda.max_memory_allocated() / 1e6)
    # net = L2Net(outdim=128).cuda()
    #
    # print("net: ", net)
    #
    # out = net()
    #
    # print(out.keys())
