# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/1 上午11:08
@Auth ： Fei Xue
@File ： sgd2.py
@Email： xuefei@sensetime.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


def nms_fast(in_corners, H, W, dist_thresh):
    """
    Run a faster approximate Non-Max-Suppression on numpy corners shaped:
      3xN [x_i,y_i,conf_i]^T

    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.

    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).

    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.

    Inputs
      in_corners - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    grid = np.zeros((H, W)).astype(int)  # Track NMS data.
    inds = np.zeros((H, W)).astype(int)  # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2, :])
    corners = in_corners[:, inds1]
    # rcorners = corners[:2, :].floor().astype(int)  # Rounded corners.
    rcorners = corners[:2, :].round().astype(int)  # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
        return np.zeros((3, 0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
        out = np.vstack((rcorners, in_corners[2])).reshape(3, 1)
        return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    # print(np.max(rcorners[0, :]), np.max(corners[0, :]), H, W)
    # print(np.max(rcorners[1, :]), np.max(corners[1, :]), H, W)
    for i, rc in enumerate(rcorners.T):
        # print("i: ", i)
        grid[rcorners[1, i], rcorners[0, i]] = 1
        inds[rcorners[1, i], rcorners[0, i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad, pad), (pad, pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
        # Account for top and left padding.
        pt = (rc[0] + pad, rc[1] + pad)
        if grid[pt[1], pt[0]] == 1:  # If not yet suppressed.
            grid[pt[1] - pad:pt[1] + pad + 1, pt[0] - pad:pt[0] + pad + 1] = 0
            grid[pt[1], pt[0]] = -1
            count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid == -1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    return out, out_inds


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


"""
class SPD2L2Net(nn.Module):
    def __init__(self, outdim=128, require_feature=False):
        super(SPD2L2Net, self).__init__()
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

        # print("desc: ", torch.norm(desc, dim=1))
        # print(seg_out1.shape, seg_out2.shape, seg_out3.shape)

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
"""

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


def process_multiscale(model, img, conf_th=0.10, scale_f=2 ** 0.25,
                       min_scale=0.5, max_scale=1.0,
                       min_size=256, max_size=9999,
                       scales=[1.0, 0.84, 0.71, 0.59, 0.5, 0.42],
                       # scales=[1.0, 0.86, 0.72],
                       ):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    B, one, H, W = img.shape
    # assert B == 1 and one == 1

    # assert max_scale <= 1
    # s = 1.0

    # X, Y, S, C, Q, D = [], [], [], [], [], []
    all_pts, all_scores, all_descs = [], [], []
    all_pts_list = []

    # while s + 0.001 >= max(min_scale, min_size / max(H, W)):
    #     # print("hhh")
    #     if s - 0.001 <= min(max_scale, max_size / max(H, W)):

    for s in scales:
        # print("s: ", s)
        # nh, nw = img.shape[2:]
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

        with torch.no_grad():
            heatmap, coarse_desc = model.det(img)

            # print("heatmap: ", heatmap.shape)
            # exit(0)

        if len(heatmap.size()) == 3:
            heatmap = heatmap.unsqueeze(1)
        if len(heatmap.size()) == 2:
            heatmap = heatmap.unsqueeze(0)
            heatmap = heatmap.unsqueeze(1)
        # print(heatmap.shape)
        if heatmap.size(2) != nh or heatmap.size(3) != nw:
            heatmap = F.interpolate(heatmap, size=[nh, nw], mode='bilinear', align_corners=False)

        heatmap = heatmap.data.cpu().numpy().squeeze()

        conf_thresh = conf_th
        nms_dist = 3
        border_remove = 3
        xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.

        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]

        valid_idex = heatmap > conf_thresh
        valid_score = heatmap[valid_idex]

        # """
        # --- Process descriptor.
        # coarse_desc = coarse_desc.data.cpu().numpy().squeeze()
        D = coarse_desc.size(1)
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            if coarse_desc.size(2) == nh and coarse_desc.size(3) == nw:
                desc = coarse_desc[:, :, pts[1, :], pts[0, :]]
                desc = desc.data.cpu().numpy().reshape(D, -1)
            else:
                # Interpolate into descriptor map using 2D point locations.
                samp_pts = torch.from_numpy(pts[:2, :].copy())
                samp_pts[0, :] = (samp_pts[0, :] / (float(nw) / 2.)) - 1.
                samp_pts[1, :] = (samp_pts[1, :] / (float(nh) / 2.)) - 1.
                samp_pts = samp_pts.transpose(0, 1).contiguous()
                samp_pts = samp_pts.view(1, 1, -1, 2)
                samp_pts = samp_pts.float()
                samp_pts = samp_pts.cuda()
                desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                desc = desc.data.cpu().numpy().reshape(D, -1)
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        pts[0, :] = pts[0, :] * W / nw
        pts[1, :] = pts[1, :] * H / nh

        pts = np.transpose(pts, [1, 0])
        all_pts.append(pts)
        all_scores.append(pts[:, 2])
        all_descs.append(np.transpose(desc, [1, 0]))

        all_pts_list.append(pts)

        # print(pts.shape)
        # print(valid_score.shape)
        # print(desc.shape)

        # s /= scale_f
        # down-scale the image for next iteration

    torch.backends.cudnn.benchmark = old_bm
    all_pts = np.vstack(all_pts)
    all_scores = all_pts[:, 2]
    all_descs = np.vstack(all_descs)
    # print("extract {:d} features from multiple scales".format(all_pts.shape[0]))

    return all_pts[:, 0:2], all_descs, all_scores
    # return pts, valid_score, desc


def process_multiscale_ori(model, img, conf_th=0.10, scale_f=2 ** 0.25,
                           min_scale=0.3, max_scale=1.0, min_size=256, max_size=2048):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    B, one, H, W = img.shape
    # assert B == 1 and one == 1

    assert max_scale <= 1
    s = 1.0

    print(img.shape)

    # X, Y, S, C, Q, D = [], [], [], [], [], []
    all_pts, all_scores, all_descs = [], [], []
    all_pts_list = []

    # print(min_size / max(H, W))
    # print(max_size / max(H, W))

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        # print("hhh")
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):

            # print("s: ", s)
            nh, nw = img.shape[2:]

            with torch.no_grad():
                heatmap, coarse_desc = model.det(img)

                # print("heatmap: ", heatmap.shape)
                # exit(0)

            if len(heatmap.size()) == 3:
                heatmap = heatmap.unsqueeze(1)
            if len(heatmap.size()) == 2:
                heatmap = heatmap.unsqueeze(0)
                heatmap = heatmap.unsqueeze(1)
            # print(heatmap.shape)
            if heatmap.size(2) != nh or heatmap.size(3) != nw:
                heatmap = F.interpolate(heatmap, size=[nh, nw], mode='bilinear', align_corners=False)

            heatmap = heatmap.data.cpu().numpy().squeeze()

            conf_thresh = conf_th
            nms_dist = 4
            border_remove = 4
            xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.

            pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]
            pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.
            # Remove points along border.
            bord = border_remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]

            valid_idex = heatmap > conf_thresh
            valid_score = heatmap[valid_idex]

            # """
            # --- Process descriptor.
            # coarse_desc = coarse_desc.data.cpu().numpy().squeeze()
            D = coarse_desc.size(1)
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                if coarse_desc.size(2) == nh and coarse_desc.size(3) == nw:
                    desc = coarse_desc[:, :, pts[1, :], pts[0, :]]
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                else:
                    # Interpolate into descriptor map using 2D point locations.
                    samp_pts = torch.from_numpy(pts[:2, :].copy())
                    samp_pts[0, :] = (samp_pts[0, :] / (float(nw) / 2.)) - 1.
                    samp_pts[1, :] = (samp_pts[1, :] / (float(nh) / 2.)) - 1.
                    samp_pts = samp_pts.transpose(0, 1).contiguous()
                    samp_pts = samp_pts.view(1, 1, -1, 2)
                    samp_pts = samp_pts.float()
                    samp_pts = samp_pts.cuda()
                    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

            pts[0, :] = pts[0, :] * W / nw
            pts[1, :] = pts[1, :] * H / nh

            pts = np.transpose(pts, [1, 0])
            all_pts.append(pts)
            all_scores.append(pts[:, 2])
            all_descs.append(np.transpose(desc, [1, 0]))

            all_pts_list.append(pts)

            # print(pts.shape)
            # print(valid_score.shape)
            # print(desc.shape)

            s /= scale_f
            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
        else:
            s /= scale_f
            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    torch.backends.cudnn.benchmark = old_bm

    ns = len(all_pts)
    all_pts = np.vstack(all_pts)
    all_scores = all_pts[:, 2]
    all_descs = np.vstack(all_descs)
    print("extract {:d} features from {:d} scales".format(all_pts.shape[0], ns))

    return all_pts, all_descs, all_scores, all_pts_list


def extract_feats_multiscale(model, img, conf_th=0.10, scale_f=2 ** 0.25,
                             min_scale=0.3, max_scale=1.0, min_size=256, max_size=2048):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    B, one, H, W = img.shape
    # assert B == 1 and one == 1

    assert max_scale <= 1
    s = 1.0

    print(img.shape)

    # X, Y, S, C, Q, D = [], [], [], [], [], []
    all_pts, all_scores, all_descs = [], [], []
    all_pts_list = []

    print(min_size / max(H, W))
    print(max_size / max(H, W))

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        # print("hhh")
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):

            # print("s: ", s)
            nh, nw = img.shape[2:]

            with torch.no_grad():
                heatmap, coarse_desc = model.det(img)

                # print("heatmap: ", heatmap.shape)
                # exit(0)

            if len(heatmap.size()) == 3:
                heatmap = heatmap.unsqueeze(1)
            if len(heatmap.size()) == 2:
                heatmap = heatmap.unsqueeze(0)
                heatmap = heatmap.unsqueeze(1)
            # print(heatmap.shape)
            if heatmap.size(2) != nh or heatmap.size(3) != nw:
                heatmap = F.interpolate(heatmap, size=[nh, nw], mode='bilinear', align_corners=False)

            heatmap = heatmap.data.cpu().numpy().squeeze()

            conf_thresh = conf_th
            nms_dist = 4
            border_remove = 4
            xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.

            pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
            pts[0, :] = ys
            pts[1, :] = xs
            pts[2, :] = heatmap[xs, ys]
            pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.
            # Remove points along border.
            bord = border_remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]

            valid_idex = heatmap > conf_thresh
            valid_score = heatmap[valid_idex]

            # """
            # --- Process descriptor.
            # coarse_desc = coarse_desc.data.cpu().numpy().squeeze()
            D = coarse_desc.size(1)
            if pts.shape[1] == 0:
                desc = np.zeros((D, 0))
            else:
                if coarse_desc.size(2) == nh and coarse_desc.size(3) == nw:
                    desc = coarse_desc[:, :, pts[1, :], pts[0, :]]
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                else:
                    # Interpolate into descriptor map using 2D point locations.
                    samp_pts = torch.from_numpy(pts[:2, :].copy())
                    samp_pts[0, :] = (samp_pts[0, :] / (float(nw) / 2.)) - 1.
                    samp_pts[1, :] = (samp_pts[1, :] / (float(nh) / 2.)) - 1.
                    samp_pts = samp_pts.transpose(0, 1).contiguous()
                    samp_pts = samp_pts.view(1, 1, -1, 2)
                    samp_pts = samp_pts.float()
                    samp_pts = samp_pts.cuda()
                    desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                    desc = desc.data.cpu().numpy().reshape(D, -1)
                    desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

            pts[0, :] = pts[0, :] * W / nw
            pts[1, :] = pts[1, :] * H / nh

            pts = np.transpose(pts, [1, 0])
            all_pts.append(pts)
            all_scores.append(pts[:, 2])
            all_descs.append(np.transpose(desc, [1, 0]))

            all_pts_list.append(pts)

            # print(pts.shape)
            # print(valid_score.shape)
            # print(desc.shape)

            s /= scale_f
            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
        else:
            s /= scale_f
            # down-scale the image for next iteration
            nh, nw = round(H * s), round(W * s)
            img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
    torch.backends.cudnn.benchmark = old_bm

    ns = len(all_pts)
    all_pts = np.vstack(all_pts)
    all_scores = all_pts[:, 2]
    all_descs = np.vstack(all_descs)
    # print("extract {:d} features from {:d} scales".format(all_pts.shape[1], ns))

    return all_pts, all_descs, all_scores


def process_singlescale(model, img, conf_th=0.10):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    B, one, H, W = img.shape

    with torch.no_grad():
        heatmap, coarse_desc = model.det(img)

        if len(heatmap.size()) == 3:
            heatmap = heatmap.unsqueeze(1)
        if len(heatmap.size()) == 2:
            heatmap = heatmap.unsqueeze(0)
            heatmap = heatmap.unsqueeze(1)
        # print(heatmap.shape)
        if heatmap.size(2) != H or heatmap.size(3) != W:
            heatmap = F.interpolate(heatmap, size=[H, W], mode='bilinear', align_corners=False)

        heatmap = heatmap.data.cpu().numpy().squeeze()

        conf_thresh = conf_th
        nms_dist = 8
        border_remove = 8
        xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.

        pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
        pts[0, :] = ys
        pts[1, :] = xs
        pts[2, :] = heatmap[xs, ys]
        pts, _ = nms_fast(pts, H, W, dist_thresh=nms_dist)  # Apply NMS.
        inds = np.argsort(pts[2, :])
        pts = pts[:, inds[::-1]]  # Sort by confidence.
        # Remove points along border.
        bord = border_remove
        toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
        toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
        toremove = np.logical_or(toremoveW, toremoveH)
        pts = pts[:, ~toremove]

        valid_idex = heatmap > conf_thresh
        valid_score = heatmap[valid_idex]

        # """
        # --- Process descriptor.
        # coarse_desc = coarse_desc.data.cpu().numpy().squeeze()
        D = coarse_desc.size(1)
        if pts.shape[1] == 0:
            desc = np.zeros((D, 0))
        else:
            if coarse_desc.size(2) == H and coarse_desc.size(3) == W:
                desc = coarse_desc[:, :, pts[1, :], pts[0, :]]
                desc = desc.data.cpu().numpy().reshape(D, -1)
            else:
                # Interpolate into descriptor map using 2D point locations.
                samp_pts = torch.from_numpy(pts[:2, :].copy())
                samp_pts[0, :] = (samp_pts[0, :] / (float(W) / 2.)) - 1.
                samp_pts[1, :] = (samp_pts[1, :] / (float(H) / 2.)) - 1.
                samp_pts = samp_pts.transpose(0, 1).contiguous()
                samp_pts = samp_pts.view(1, 1, -1, 2)
                samp_pts = samp_pts.float()
                samp_pts = samp_pts.cuda()
                desc = torch.nn.functional.grid_sample(coarse_desc, samp_pts)
                desc = desc.data.cpu().numpy().reshape(D, -1)
                desc /= np.linalg.norm(desc, axis=0)[np.newaxis, :]

        pts = np.transpose(pts, [1, 0])
        scores = pts[:, 2]
        desc = np.transpose(desc, [1, 0])

        return pts[:, 0:2], desc, scores


class Sgd2(BaseModel):
    default_conf = {
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    required_inputs = ['image']

    def _init(self, conf):
        # self.net = SPD2L2Net(outdim=128).eval().cuda()
        # self.net = SPD2L2Net(outdim=64).eval().cuda()
        self.net = L2SegNetNB(outdim=128).eval().cuda()
        self.net.load_state_dict(torch.load(conf["model_fn"])["model"])

    def _forward(self, data):
        RGB_mean = [0.485, 0.456, 0.406]
        RGB_std = [0.229, 0.224, 0.225]

        norm_RGB = tvf.Compose([tvf.Normalize(mean=RGB_mean, std=RGB_std)])
        image = data['image']
        image = norm_RGB(image)
        if self.conf['multiscale']:
            keypoints, descriptors, scores = process_multiscale(model=self.net,
                                                                conf_th=self.conf["conf_th"],
                                                                img=image)
        else:
            keypoints, descriptors, scores = process_singlescale(model=self.net, img=image,
                                                                 conf_th=self.conf["conf_th"])
        # print("feats1: ", keypoints.shape, descriptors.shape, scores.shape)

        if self.conf["max_keypoints"]:
            topK = self.conf["max_keypoints"]
            if topK < keypoints.shape[0]:
                idxs = (-scores).argsort()[:topK]
                keypoints = keypoints[idxs]
                descriptors = descriptors[idxs]
                scores = scores[idxs]
        # print("feats: ", keypoints.shape, descriptors.shape, scores.shape)
        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }
