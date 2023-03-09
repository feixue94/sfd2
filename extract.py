# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   sfd2 -> extract
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   09/03/2023 16:27
=================================================='''
import torch
import torch.nn.functional as F

import os
from PIL import Image
import numpy as np
from tools.dataloader import norm_RGB


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


def extrat_spp_feats_multiscale(model, img, conf_th=0.0050, scale_f=2 ** 0.25,
                                min_scale=0.05, max_scale=1.0, min_size=256, max_size=2048):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    B, one, H, W = img.shape

    assert max_scale <= 1
    s = 1.0

    # print(img.shape)

    # X, Y, S, C, Q, D = [], [], [], [], [], []
    all_pts, all_scores, all_descs = [], [], []
    all_pts_list = []

    # print(min_size / max(H, W))
    # print(max_size / max(H, W))

    while s + 0.001 >= max(min_scale, min_size / max(H, W)):
        # print("hhh")
        if s - 0.001 <= min(max_scale, max_size / max(H, W)):
            nh, nw = img.shape[2:]

            with torch.no_grad():
                heatmap, stability, coarse_desc = model.det(img)

                # print("nh, nw, heatmap: ", nh, nw, heatmap.shape, coarse_desc.shape)

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

            # print("heatmap: ", heatmap.shape, pts.shape)
            # print(valid_score.shape)
            # print(desc.shape)

        s /= scale_f
        # down-scale the image for next iteration
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)

    torch.backends.cudnn.benchmark = old_bm

    ns = len(all_pts)
    if ns == 0:
        return None, None, None
    all_pts = np.vstack(all_pts)
    all_scores = all_pts[:, 2]
    all_descs = np.vstack(all_descs)
    # print("extract {:d} features from {:d} scales".format(all_pts.shape[1], ns))

    return all_pts, all_descs, all_scores
    # return pts, valid_score, desc


def extract_spp_feats_singlescale(model, img, conf_th=0.10):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    B, one, H, W = img.shape

    with torch.no_grad():
        heatmap, stability, coarse_desc = model.det(img)

        if len(heatmap.size()) == 3:
            heatmap = heatmap.unsqueeze(1)
        if len(heatmap.size()) == 2:
            heatmap = heatmap.unsqueeze(0)
            heatmap = heatmap.unsqueeze(1)
        # print(heatmap.shape)
        if heatmap.size(2) != H or heatmap.size(3) != W:
            heatmap = F.interpolate(heatmap, size=[H, W], mode='bilinear', align_corners=False)

        if stability is not None:
            heatmap = heatmap * stability

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

        return pts, desc, scores, coarse_desc.squeeze().cpu().numpy(), heatmap


def extract_spp_return(sgd2, img_path, conf_th=0.1, need_nms=False, multi_scale=False, min_size=256, max_size=9999):
    if type(img_path) is str:
        img = Image.open(img_path).convert('RGB')
        H, W = img.size
        img = norm_RGB(img)
        img = img[None]
        img = img.cuda()
    else:
        img = img_path

    # print('img: ', img.shape, conf_th)

    # conf_th = 0.001
    min_size = min_size
    max_size = max_size
    if multi_scale:
        xys, desc, scores = extrat_spp_feats_multiscale(model=sgd2, img=img, conf_th=conf_th, scale_f=1.2,
                                                        min_size=min_size, max_size=max_size)  # spp mode

        return xys, desc, scores
    else:
        xys, desc, scores, desc_full, heat_map = extract_spp_feats_singlescale(model=sgd2, img=img, conf_th=conf_th)
        return xys, desc, scores, desc_full, heat_map
