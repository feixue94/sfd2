# -*- coding: utf-8 -*-
# @Time    : 2020/6/16 下午4:15
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : sppnet.py

import torch
import torch.nn as nn


class SPPNet(nn.Module):
    def __init__(self, outdim=128, use_bn=False):
        super(SPPNet, self).__init__()
        self.relu = torch.nn.ReLU(inplace=True)
        # self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        # c1, c2, c3, c4, c5 = 32, 64, 128, 128, 128
        c1, c2, c3, c4, c5 = 64, 64, 128, 128, 256

        self.outdim = outdim
        #
        self.use_bn = use_bn
        # Shared Encoder.
        self.conv1a = torch.nn.Conv2d(1, c1, kernel_size=3, stride=1, padding=1)
        self.conv1b = torch.nn.Conv2d(c1, c1, kernel_size=3, stride=2, padding=1)

        self.conv2a = torch.nn.Conv2d(c1, c2, kernel_size=3, stride=1, padding=1)
        self.conv2b = torch.nn.Conv2d(c2, c2, kernel_size=3, stride=2, padding=1)

        self.conv3a = torch.nn.Conv2d(c2, c3, kernel_size=3, stride=1, padding=1)
        self.conv3b = torch.nn.Conv2d(c3, c3, kernel_size=3, stride=2, padding=1)

        self.conv4a = torch.nn.Conv2d(c3, c4, kernel_size=3, stride=1, padding=1)
        self.conv4b = torch.nn.Conv2d(c4, c4, kernel_size=3, stride=1, padding=1)

        # Detector Head.
        self.convPa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convPb = torch.nn.Conv2d(c5, 64, kernel_size=1, stride=1, padding=0)
        self.sigmoid = torch.nn.Sigmoid()

        # Descriptor Head.
        self.convDa = torch.nn.Conv2d(c4, c5, kernel_size=3, stride=1, padding=1)
        self.convDb = torch.nn.Conv2d(c5, out_channels=outdim, kernel_size=1, stride=1, padding=0)

        if self.use_bn:
            self.bn1a = nn.BatchNorm2d(c1, affine=False)
            self.bn1b = nn.BatchNorm2d(c1, affine=False)

            self.bn2a = nn.BatchNorm2d(c2, affine=False)
            self.bn2b = nn.BatchNorm2d(c2, affine=False)

            self.bn3a = nn.BatchNorm2d(c3, affine=False)
            self.bn3b = nn.BatchNorm2d(c3, affine=False)

            self.bn4a = nn.BatchNorm2d(c4, affine=False)
            self.bn4b = nn.BatchNorm2d(c4, affine=False)

            self.bnPa = nn.BatchNorm2d(c5, affine=False)
            self.bnDa = nn.BatchNorm2d(c5, affine=False)

        # initialize the weight
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)

    def det(self, x):
        if self.use_bn:
            x = self.relu(self.bn1a(self.conv1a(x)))
            x = self.relu(self.bn1b(self.conv1b(x)))

            x = self.relu(self.bn2a(self.conv2a(x)))
            x = self.relu(self.bn2b(self.conv2b(x)))

            x = self.relu(self.bn3a(self.conv3a(x)))
            x = self.relu(self.bn3b(self.conv3b(x)))

            x = self.relu(self.bn4a(self.conv4a(x)))
            x = self.relu(self.bn4b(self.conv4b(x)))

            # Detector Head.
            cPa = self.relu(self.bnPa(self.convPa(x)))
            semi = self.convPb(cPa)
            semi = self.sigmoid(semi)

            # Descriptor Head

            cDa = self.relu(self.bnDa(self.convDa(x)))
            desc = self.convDb(cDa)
        else:
            x = self.relu(self.conv1a(x))
            x = self.relu(self.conv1b(x))
            # x = self.pool(x)
            x = self.relu(self.conv2a(x))
            x = self.relu(self.conv2b(x))
            # x = self.pool(x)
            x = self.relu(self.conv3a(x))
            x = self.relu(self.conv3b(x))
            # x = self.pool(x)
            x = self.relu(self.conv4a(x))
            x = self.relu(self.conv4b(x))

            # Detector Head.
            cPa = self.relu(self.convPa(x))
            semi = self.convPb(cPa)
            semi = self.sigmoid(semi)

            # Descriptor Head
            cDa = self.relu(self.convDa(x))
            desc = self.convDb(cDa)
        dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        return semi, desc

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        semi, desc = self.det(x)
        Hc, Wc = semi.size(2), semi.size(3)
        # recover resolution
        semi = semi.permute([0, 2, 3, 1])
        score = semi.view(semi.size(0), Hc, Wc, 8, 8)
        score = score.permute([0, 1, 3, 2, 4])
        score = score.contiguous().view(score.size(0), Hc * 8, Wc * 8)

        # dn = torch.norm(desc, p=2, dim=1)  # Compute the norm.
        # desc = desc.div(torch.unsqueeze(dn, 1))  # Divide by norm to normalize.

        desc1 = desc[:b, :, :, :]
        desc2 = desc[b:, :, :, :]

        score1 = score[:b, :, :]
        score2 = score[b:, :, :]

        # print("desc1: ", desc1.size())
        # print("desc2: ", desc2.size())

        return {
            'dense_features1': desc1,
            'scores1': score1,
            'dense_features2': desc2,
            'scores2': score2,
        }


if __name__ == '__main__':
    import time

    img = torch.ones((1, 1, 480, 640)).cuda()

    total_time = 0
    # net = SPD2NetSPP(outdim=128, use_bn=True).cuda()
    net = SPPNet(outdim=128, use_bn=True).cuda()

    for i in range(1000):
        start_time = time.time()
        out = net.det(img)

        total_time = total_time + time.time() - start_time

    print("mean time: ", total_time / 1000)