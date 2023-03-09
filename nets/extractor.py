# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   r2d2 -> extractor
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   08/07/2022 00:42
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision.transforms as tvf

RGB_mean = [0.485, 0.456, 0.406]
RGB_std = [0.229, 0.224, 0.225]

norm_RGB = tvf.Compose([tvf.Normalize(mean=RGB_mean, std=RGB_std)])


def simple_nms(scores, nms_radius: int):
    """ Fast Non-maximum suppression to remove nearby points """
    assert (nms_radius >= 0)

    def max_pool(x):
        return torch.nn.functional.max_pool2d(
            x, kernel_size=nms_radius * 2 + 1, stride=1, padding=nms_radius)

    zeros = torch.zeros_like(scores)
    max_mask = scores == max_pool(scores)
    for _ in range(2):
        supp_mask = max_pool(max_mask.float()) > 0
        supp_scores = torch.where(supp_mask, zeros, scores)
        new_max_mask = supp_scores == max_pool(supp_scores)
        max_mask = max_mask | (new_max_mask & (~supp_mask))
    return torch.where(max_mask, scores, zeros)


def remove_borders(keypoints, scores, border: int, height: int, width: int):
    """ Removes keypoints too close to the border """
    mask_h = (keypoints[:, 0] >= border) & (keypoints[:, 0] < (height - border))
    mask_w = (keypoints[:, 1] >= border) & (keypoints[:, 1] < (width - border))
    mask = mask_h & mask_w
    return keypoints[mask], scores[mask]


def top_k_keypoints(keypoints, scores, k: int):
    if k >= len(keypoints):
        return keypoints, scores
    scores, indices = torch.topk(scores, k, dim=0)
    return keypoints[indices], scores


def sample_descriptors(keypoints, descriptors, s: int = 8):
    """ Interpolate descriptors at keypoint locations """
    b, c, h, w = descriptors.shape
    keypoints = keypoints - s / 2 + 0.5
    keypoints /= torch.tensor([(w * s - s / 2 - 0.5), (h * s - s / 2 - 0.5)],
                              ).to(keypoints)[None]
    keypoints = keypoints * 2 - 1  # normalize to (-1, 1)
    args = {'align_corners': True} if torch.__version__ >= '1.3' else {}
    descriptors = torch.nn.functional.grid_sample(
        descriptors, keypoints.view(b, 1, -1, 2), mode='bilinear', **args)
    descriptors = torch.nn.functional.normalize(
        descriptors.reshape(b, c, -1), p=2, dim=1)
    return descriptors


def extractor(scores, descs, nms_radius=4, conf_th=0.001, remove_borders=8, max_keypoints=1000, scale_factor=4):
    scores = simple_nms(scores, nms_radius)

    # Extract keypoints
    keypoints = [
        torch.nonzero(s > conf_th)
        for s in scores]
    scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]

    # Discard keypoints near the image borders
    keypoints, scores = list(zip(*[
        remove_borders(k, s, remove_borders, scores.shape[2], scores.shape[3])
        for k, s in zip(keypoints, scores)]))

    # Keep the k keypoints with highest score
    if max_keypoints >= 0:
        keypoints, scores = list(zip(*[
            top_k_keypoints(k, s, max_keypoints)
            for k, s in zip(keypoints, scores)]))

    # Convert (h, w) to (x, y)
    keypoints = [torch.flip(k, [1]).float() for k in keypoints]
    # Extract descriptors
    descriptors = [sample_descriptors(k[None], d[None], scale_factor)[0]
                   for k, d in zip(keypoints, descs)]

    return keypoints, descriptors


def extract_resnet_return(model, img, conf_th=0.001,
                          mask=None,
                          topK=-1,
                          **kwargs):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    img = norm_RGB(img.squeeze())
    img = img[None]
    img = img.cuda()

    B, one, H, W = img.shape

    all_pts = []
    all_descs = []

    if 'scales' in kwargs.keys():
        scales = kwargs.get('scales')
    else:
        scales = [1.0]

    for s in scales:
        if s == 1.0:
            new_img = img
        else:
            nh = int(H * s)
            nw = int(W * s)
            new_img = F.interpolate(img, size=(nh, nw), mode='bilinear', align_corners=False)
        nh, nw = new_img.shape[2:]

        with torch.no_grad():
            heatmap, stability, coarse_desc = model.det(new_img)

            # print("nh, nw, heatmap, desc: ", nh, nw, heatmap.shape, coarse_desc.shape)
            if len(heatmap.size()) == 3:
                heatmap = heatmap.unsqueeze(1)
            if len(heatmap.size()) == 2:
                heatmap = heatmap.unsqueeze(0)
                heatmap = heatmap.unsqueeze(1)
            # print(heatmap.shape)
            if heatmap.size(2) != nh or heatmap.size(3) != nw:
                heatmap = F.interpolate(heatmap, size=[nh, nw], mode='bilinear', align_corners=False)

            if stability is not None:
                heatmap = heatmap * stability
                # print('using stability....')

            conf_thresh = conf_th
            nms_dist = 4
            border_remove = 4
            # heatmap = heatmap.data.cpu().numpy().squeeze()
            # xs, ys = np.where(heatmap >= conf_thresh)  # Confidence threshold.
            #
            # pts = np.zeros((3, len(xs)))  # Populate point data sized 3xN.
            # pts[0, :] = ys
            # pts[1, :] = xs
            # pts[2, :] = heatmap[xs, ys]
            #
            # pts, _ = nms_fast(pts, nh, nw, dist_thresh=nms_dist)  # Apply NMS.

            scores = simple_nms(heatmap, nms_radius=nms_dist)
            keypoints = [
                torch.nonzero(s > conf_thresh)
                for s in scores]
            scores = [s[tuple(k.t())] for s, k in zip(scores, keypoints)]
            # print(keypoints[0].shape)
            keypoints = [torch.flip(k, [1]).float() for k in keypoints]
            scores = scores[0].data.cpu().numpy().squeeze()
            keypoints = keypoints[0].data.cpu().numpy().squeeze()
            pts = keypoints.transpose()
            pts[2, :] = scores
            # print(pts.shape, keypoints.shape)
            # pts = np.zeros((3, keypoints.shape[0]))  # Populate point data sized 3xN.
            # print(len(keypoints), keypoints[0].shape, keypoints[0][0])
            # exit(0)
            # pts[0, :] = keypoints[:, 1]
            # pts[1, :] = keypoints[:, 0]
            # pts[2, :] = scores

            inds = np.argsort(pts[2, :])
            pts = pts[:, inds[::-1]]  # Sort by confidence.
            # Remove points along border.
            bord = border_remove
            toremoveW = np.logical_or(pts[0, :] < bord, pts[0, :] >= (W - bord))
            toremoveH = np.logical_or(pts[1, :] < bord, pts[1, :] >= (H - bord))
            toremove = np.logical_or(toremoveW, toremoveH)
            pts = pts[:, ~toremove]

            # valid_idex = heatmap > conf_thresh
            # valid_score = heatmap[valid_idex]
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

            if pts.shape[1] == 0:
                continue

            # print(pts.shape, heatmap.shape, new_img.shape, img.shape, nw, nh, W, H)
            pts[0, :] = pts[0, :] * W / nw
            pts[1, :] = pts[1, :] * H / nh
            all_pts.append(np.transpose(pts, [1, 0]))
            all_descs.append(np.transpose(desc, [1, 0]))

    all_pts = np.vstack(all_pts)
    all_descs = np.vstack(all_descs)

    torch.backends.cudnn.benchmark = old_bm

    if all_pts.shape[0] == 0:
        return None, None, None

    # keypoints = np.transpose(pts, [1, 0])
    # descriptors = np.transpose(desc, [1, 0])
    # scores = keypoints[:, 2]
    # keypoints = keypoints[:, 0:2]

    keypoints = all_pts[:, 0:2]
    scores = all_pts[:, 2]
    descriptors = all_descs

    # if mask is not None:
    #     id_img = np.int32(mask[:, :, 2]) * 256 * 256 + np.int32(mask[:, :, 1]) * 256 + np.int32(mask[:, :, 0])
    #     labels = id_img[int(keypoints[:, 0])]
    #
    if mask is not None:
        # cv2.imshow("mask", mask)
        # cv2.waitKey(0)
        labels = []
        others = []
        keypoints_with_labels = []
        scores_with_labels = []
        descriptors_with_labels = []
        keypoints_without_labels = []
        scores_without_labels = []
        descriptors_without_labels = []

        id_img = np.int32(mask[:, :, 2]) * 256 * 256 + np.int32(mask[:, :, 1]) * 256 + np.int32(mask[:, :, 0])
        # print(img.shape, id_img.shape)

        for i in range(keypoints.shape[0]):
            x = keypoints[i, 0]
            y = keypoints[i, 1]
            # print("x-y", x, y, int(x), int(y))
            gid = id_img[int(y), int(x)]
            if gid == 0:
                keypoints_without_labels.append(keypoints[i])
                scores_without_labels.append(scores[i])
                descriptors_without_labels.append(descriptors[i])
                others.append(0)
            else:
                keypoints_with_labels.append(keypoints[i])
                scores_with_labels.append(scores[i])
                descriptors_with_labels.append(descriptors[i])
                labels.append(gid)

        # keypoints_with_labels = np.array(keypoints_with_labels)
        # scores_with_labels = np.array(scores_with_labels)
        # descriptors_with_labels = np.array(descriptors_with_labels)
        # labels = np.array(labels, np.int32)
        #
        # keypoints_without_labels = np.array(keypoints_without_labels)
        # scores_without_labels = np.array(scores_without_labels)
        # descriptors_without_labels = np.array(descriptors_without_labels)
        # others = np.array(others, np.int32)

        if topK > 0:
            if topK <= len(keypoints_with_labels):
                idxes = np.array(scores_with_labels, np.float).argsort()[::-1][:topK]
                keypoints = np.array(keypoints_with_labels, np.float)[idxes]
                scores = np.array(scores_with_labels, np.float)[idxes]
                labels = np.array(labels, np.int32)[idxes]
                descriptors = np.array(descriptors_with_labels, np.float)[idxes]
            elif topK >= len(keypoints_with_labels) + len(keypoints_without_labels):
                # keypoints = np.vstack([keypoints_with_labels, keypoints_without_labels])
                # scores = np.vstack([scorescc_with_labels, scores_without_labels])
                # descriptors = np.vstack([descriptors_with_labels, descriptors_without_labels])
                # labels = np.vstack([labels, others])
                keypoints = keypoints_with_labels
                scores = scores_with_labels
                descriptors = descriptors_with_labels
                for i in range(len(others)):
                    keypoints.append(keypoints_without_labels[i])
                    scores.append(scores_without_labels[i])
                    descriptors.append(descriptors_without_labels[i])
                    labels.append(others[i])
            else:
                n = topK - len(keypoints_with_labels)
                idxes = np.array(scores_without_labels, np.float).argsort()[::-1][:n]
                keypoints = keypoints_with_labels
                scores = scores_with_labels
                descriptors = descriptors_with_labels
                for i in idxes:
                    keypoints.append(keypoints_without_labels[i])
                    scores.append(scores_without_labels[i])
                    descriptors.append(descriptors_without_labels[i])
                    labels.append(others[i])
        keypoints = np.array(keypoints, np.float)
        descriptors = np.array(descriptors, np.float)
        # print(keypoints.shape, descriptors.shape)
        return {"keypoints": np.array(keypoints, np.float),
                "descriptors": np.array(descriptors, np.float),
                "scores": np.array(scores, np.float),
                "labels": np.array(labels, np.int32),
                }
    else:
        # print(topK)
        if topK > 0:
            idxes = np.array(scores, dtype=float).argsort()[::-1][:topK]
            keypoints = np.array(keypoints[idxes], dtype=float)
            scores = np.array(scores[idxes], dtype=float)
            descriptors = np.array(descriptors[idxes], dtype=float)

        keypoints = np.array(keypoints, dtype=float)
        scores = np.array(scores, dtype=float)
        descriptors = np.array(descriptors, dtype=float)

        # print(keypoints.shape, descriptors.shape)

        return {"keypoints": np.array(keypoints, dtype=float),
                "descriptors": descriptors,
                "scores": scores,
                }
    # return all_pts, all_descs, all_scores
