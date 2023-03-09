# -*- coding: utf-8 -*-
# @Time    : 2020/6/15 上午11:01
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : dfanet.py

import torch
import torch.nn as nn
import time

from nets.layers import SeparableConv2d, Block, SEModule, conv


class DFANet(nn.Module):
    def __init__(self, outdim, use_bn=True):
        super(DFANet, self).__init__()

        c1, c2, c3, c4, c5 = 16, 32, 64, 96, 128
        self.conv1a = conv(1, c1, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.conv1b = Block(inputChannel=c1, outputChannel=c2, stride=2)

        self.conv2a = conv(c2, c2, kernel_size=3, padding=1, relu=True, bn=True)
        self.conv2b = Block(inputChannel=c2, outputChannel=c3, stride=2)

        self.conv3a = conv(c3, c3, kernel_size=3, padding=1, relu=True, bn=True)
        self.conv3b = Block(inputChannel=c3, outputChannel=c4, stride=2)

        self.conv4a = conv(c4, c5, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.conv4b = conv(c5, c5, kernel_size=3, stride=1, padding=1, relu=True, bn=True)

        # Detector Head.
        self.convPa = conv(c5, 64, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.convPb = torch.nn.Conv2d(64, 64, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

        # Descriptor Head.
        self.convDa = conv(c5, 128, kernel_size=3, stride=1, padding=1, relu=True, bn=True)
        self.convDb = torch.nn.Conv2d(128, out_channels=outdim, kernel_size=1, stride=1, padding=0)

    def det(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)

        x = self.conv2a(x)
        x = self.conv2b(x)

        x = self.conv3a(x)
        x = self.conv3b(x)

        x = self.conv4a(x)
        x = self.conv4b(x)

        # Detector Head.
        cPa = self.convPa(x)
        semi = self.convPb(cPa)
        semi = self.sigmoid(semi)

        # Descriptor Head
        cDa = self.convDa(x)
        desc = self.convDb(cDa)

        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        # print("semi:", semi.shape)
        # print("desc:", desc.shape)

        return semi, desc

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        semi, desc = self.det(x)

        Hc, Wc = semi.size(2), semi.size(3)
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

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


if __name__ == '__main__':
    # img = torch.ones((1, 1, 512, 512)).cuda()
    img = torch.ones((1, 1, 480, 640)).cuda()
    # net = CSPD2Net32(outdim=128, use_bn=True).cuda()

    total_time = 0
    net = DFANet(outdim=128, use_bn=True).cuda()
    for i in range(1000):
        start_time = time.time()
        out = net.det(img)
        total_time = total_time + time.time() - start_time

    print("mean time: ", total_time / 1000)
    print("")
