# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb

import torch
import torch.nn as nn
import torch.nn.functional as F

from nets.sampler import *
from nets.repeatability_loss import *
from nets.reliability_loss import *


class SupLoss(nn.Module):
    def __init__(self, ap_loss_fun, use_det=True, use_desc=False,
                 seg_desc=None, seg_det=None, desc_hard=False,
                 use_pred=False,
                 seg_feat=None,
                 weights=None,
                 margin=1.0,
                 use_weight=True,
                 ):
        nn.Module.__init__(self)

        self.ap_loss_fun = ap_loss_fun
        self.use_det = use_det
        self.use_desc = use_desc
        self.use_pred = use_pred
        self.seg_desc = seg_desc
        self.seg_det = seg_det
        self.desc_hard = desc_hard
        self.seg_feat = seg_feat
        self.weights = weights
        self.margin = margin
        self.use_weight = use_weight

    def det_loss_ce(self, pred_score, seg_confidence, seg_mask=None):
        return F.cross_entropy(pred_score, seg_confidence).mean()

    def det_loss(self, pred_score, gt_score, weight=None, seg_confidence=None, seg_mask=None):
        # print("pred_score: ", pred_score.size())
        # print("gt_score: ", gt_score.size())
        # print("conf: ", seg_confidence.size())
        # print("mask: ", seg_mask.size())

        if seg_confidence is not None:
            if self.seg_det == "ce":
                # loss = F.binary_cross_entropy(torch.sigmoid(pred_score), gt_score * seg_confidence, reduce=False)
                loss = F.binary_cross_entropy(pred_score, gt_score * seg_confidence, reduce=False)
            elif self.seg_det == "l1":
                loss = torch.abs(pred_score - gt_score * seg_confidence)
            # loss = torch.norm(pred_score - gt_score * seg_confidence)  # default F norm
        else:
            if self.seg_det == "ce":
                # loss = F.binary_cross_entropy(torch.sigmoid(pred_score), gt_score, reduce=False)
                loss = F.binary_cross_entropy(pred_score, gt_score, reduce=False)
            elif self.seg_det == "l1":
                loss = torch.abs(pred_score - gt_score)

        if weight is not None:
            loss = loss * weight

        # print("loss: ", torch.mean(loss[seg_mask]), torch.mean(loss))
        if seg_mask is not None:
            return torch.mean(loss[seg_mask])
        else:
            return torch.mean(loss)

    def desc_loss(self, pred_desc, gt_desc, weight=None):
        """
                desc_loss1 = (torch.mean(torch.abs(pred_desc1 - spp_desc1) * ds_weight1, dim=1)).mean()
        """
        # print("weight: ", weight.shape)
        # print("pred_desc: ", pred_desc.shape)
        if pred_desc.size(2) != gt_desc.size(2):
            pred_desc_sz = F.interpolate(pred_desc, (gt_desc.size(2), gt_desc.size(3)), mode="bilinear",
                                         align_corners=False)

            loss = torch.abs(pred_desc_sz - gt_desc)
        else:
            loss = torch.abs(pred_desc - gt_desc)

        if weight is not None:
            loss = torch.mean(loss, dim=1) * weight

        # print("weight: ", torch.sum(weight) / 1000.)
        # return torch.mean(loss, dim=1).mean()
        return loss.mean()

    def sem_desc_loss_wap(self, input, margin=1.0):
        # if self.use_pred:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        conf_th = input["score_th"]
        b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # print("sel_desc1: ", sel_desc1.size())
        # print("sel_seg1: ", sel_seg1.size())
        # print("sel_score1: ", sel_score1.size())

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        # dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.unsqueeze(0)
        # print("dist_map: ", dist_map12.size())
        # print("seg_cst_map: ", seg_const_map.size())
        # print("score_map: ", score_map12.size())
        # exit(0)

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        # print("pos_ids: ", torch.sum(pos_ids12), pos_ids12)
        # print("neg_ids: ", torch.sum(neg_ids12))

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

        dist12 = margin + pos_dist12 - neg_dist12
        return dist12

        # dist11 = sel_crrelation(desc=sel_desc1, segs=sel_seg1, score=sel_score1)
        # dist22 = sel_crrelation(desc=sel_desc2, segs=sel_seg2, score=sel_score2)
        #
        # return (dist12 + dist11 + dist22) / 3.
        # print("pos_num: ", pos_ids.sum())
        # print("neg_num: ", neg_ids.sum())

        # print("neg_dist: ", neg_dist)
        # print("pos_dist: ", pos_dist)
        # return pos_dist + torch.max(0, margin - neg_dist)
        # return margin + pos_dist12 - neg_dist12

    def sem_desc_loss(self, input, margin=1.0):
        # if self.use_pred:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        conf_th = input["score_th"]
        b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # print("sel_desc1: ", sel_desc1.size())
        # print("sel_seg1: ", sel_seg1.size())
        # print("sel_score1: ", sel_score1.size())

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = (sel_score1.unsqueeze(1) + sel_score2.unsqueeze(0)) / 2.
        # print("dist_map: ", dist_map12.size())
        # print("seg_cst_map: ", seg_const_map.size())
        # print("score_map: ", score_map12.size())
        # exit(0)

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        # print("pos_ids: ", torch.sum(pos_ids12), pos_ids12)
        # print("neg_ids: ", torch.sum(neg_ids12))

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

        dist12 = margin + pos_dist12 - neg_dist12
        return dist12

        # dist11 = sel_crrelation(desc=sel_desc1, segs=sel_seg1, score=sel_score1)
        # dist22 = sel_crrelation(desc=sel_desc2, segs=sel_seg2, score=sel_score2)
        #
        # return (dist12 + dist11 + dist22) / 3.
        # print("pos_num: ", pos_ids.sum())
        # print("neg_num: ", neg_ids.sum())

        # print("neg_dist: ", neg_dist)
        # print("pos_dist: ", pos_dist)
        # return pos_dist + torch.max(0, margin - neg_dist)
        # return margin + pos_dist12 - neg_dist12

    def sem_desc_loss_new(self, input, margin=1.0, N=256):
        def gen_grid(step, B, H, W, dev, border=4):
            # print("step: ", step)
            # B, two, H, W = aflow.shape
            # dev = aflow.device
            b1 = torch.arange(B, device=dev)
            if step > 0:
                # regular grid
                x1 = torch.arange(border, W - border, step, device=dev)
                y1 = torch.arange(border, H - border, step, device=dev)
                H1, W1 = len(y1), len(x1)
                x1 = x1[None, None, :].expand(B, H1, W1).reshape(-1)
                y1 = y1[None, :, None].expand(B, H1, W1).reshape(-1)
                b1 = b1[:, None, None].expand(B, H1, W1).reshape(-1)
                shape = (B, H1, W1)
            else:
                # randomly spread
                n = (H - 2 * border) * (W - 2 * border) // step ** 2
                x1 = torch.randint(border, W - border, (n,), device=dev)
                y1 = torch.randint(border, H - border, (n,), device=dev)
                x1 = x1[None, :].expand(B, n).reshape(-1)
                y1 = y1[None, :].expand(B, n).reshape(-1)
                b1 = b1[:, None].expand(B, n).reshape(-1)
                shape = (B, n)
            return b1, y1, x1

        # if self.use_pred:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]

        # print(input["score"].shape, input["gt_score"].shape)
        seg_confidence = input["seg_confidence"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)
        conf_th = input["score_th"]
        B, H, W, dev = scores.shape[0], scores.shape[1], scores.shape[2], scores.device
        b = B // 2
        # """
        bs, ys, xs = gen_grid(step=-4, B=b, H=H, W=W, dev=dev)
        mask1 = (scores[0:b][bs, ys, xs] >= conf_th) & masks[0:b][bs, ys, xs]
        mask2 = (scores[b:][bs, ys, xs] >= conf_th) & masks[b:][bs, ys, xs]
        b1 = bs[mask1]
        y1 = ys[mask1]
        x1 = xs[mask1]
        b2 = bs[mask2]
        y2 = ys[mask2]
        x2 = xs[mask2]
        sel_desc1 = descs[0:b][b1, y1, x1]
        sel_desc2 = descs[b:][b2, y2, x2]
        sel_seg1 = segs[0:b][b1, y1, x1]
        sel_seg2 = segs[b:][b2, y2, x2]
        # if self.use_segc:
        sel_score1 = (scores * seg_confidence)[0:b][b1, y1, x1]
        sel_score2 = (scores * seg_confidence)[b:][b2, y2, x2]
        # else:
        # sel_score1 = scores[0:b][b1, y1, x1]
        # sel_score2 = scores[b:][b2, y2, x2]

        """
        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = (scores * seg_confidence)[0:b][ids1]
        sel_score2 = (scores * seg_confidence)[b:][ids2]

        sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.001, max=1.)
        """

        # print("sel_desc1: ", sel_desc1.size())
        # print("sel_seg1: ", sel_seg1.size())
        # print("sel_score1: ", sel_score1.size())

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.squeeze(0)
        # print("dist_map: ", dist_map.size())
        # print("seg_cst_map: ", seg_const_map.size())
        # print("score_map: ", score_map.size())

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        # print("pos_ids: ", torch.sum(pos_ids12), pos_ids12)
        # print("neg_ids: ", torch.sum(neg_ids12))

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])
        # print("pos_num: ", pos_ids.sum())
        # print("neg_num: ", neg_ids.sum())

        # print("neg_dist: ", neg_dist)
        # print("pos_dist: ", pos_dist)
        # return pos_dist + torch.max(0, margin - neg_dist)
        return margin + pos_dist12 - neg_dist12

    def sem_desc_loss_hard(self, input, margin=1.5):
        scores = input["gt_score"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        conf_th = input["score_th"]
        b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        dist_map12 = torch.sqrt(dist_map12)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = (sel_score1.unsqueeze(1) + sel_score2.unsqueeze(0)) / 2
        neg_ids12 = (seg_const_map12 > 0)  # with different labels
        pos_dist_ones = torch.ones_like(dist_map12)
        pos_dist_ones[neg_ids12] *= 10
        max_pos_dist, max_pos_ids = torch.max(dist_map12 - pos_dist_ones, dim=1, keepdim=True)

        # print(max_pos_dist.shape, dist_map12.shape)

        dist = margin + max_pos_dist.expand_as(dist_map12) - dist_map12
        loss = (dist * score_map12)[dist > 0].mean()
        return loss

    def sem_desc_loss_hard1(self, input, margin=1.5):
        scores = input["gt_score"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        conf_th = input["score_th"]
        b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        loss = 0
        n_valid = 0
        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        dist_map12 = torch.sqrt(dist_map12)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = (sel_score1.unsqueeze(1) + sel_score2.unsqueeze(0)) / 2
        neg_ids12 = (seg_const_map12 > 0)  # with different labels
        # neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        # dist12 = margin - neg_dist12
        # pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])
        # zeros = torch.zeros_like(neg_dist12)
        # n_valid = ((margin - dist_map12[neg_ids12]) > 0).sum()
        # dist12 = torch.clamp(0.5 - dist_map12[neg_ids12], min=0.0) * score_map12[neg_ids12]
        # dist12 = dist12.mean()
        if torch.sum(neg_ids12) > 0:
            dist12 = (margin - dist_map12[neg_ids12]) * score_map12[neg_ids12]
            dist12 = dist12[dist12 > 0].mean()
            loss = loss + dist12
            n_valid += 1

        """
        dist_map11 = sel_desc1 @ sel_desc1.t()
        seg_const_map11 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg1.unsqueeze(0))  # M x N
        score_map11 = (sel_score1.unsqueeze(1) + sel_score1.unsqueeze(0)) / 2
        neg_ids11 = (seg_const_map11 > 0)  # with different labels
        if torch.sum(neg_ids11) > 0:
            dist11 = (margin - dist_map11[neg_ids11]) * score_map11[neg_ids11]
            dist11 = dist11[dist11 > 0].mean()
            loss = loss + dist11
            n_valid += 1

        dist_map22 = sel_desc2 @ sel_desc2.t()
        seg_const_map22 = torch.abs(sel_seg2.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map22 = (sel_score2.unsqueeze(1) + sel_score2.unsqueeze(0)) / 2
        neg_ids22 = (seg_const_map22 > 0)  # with different labels
        if torch.sum(neg_ids22) > 0:
            dist22 = (margin - dist_map22[neg_ids22]) * score_map22[neg_ids22]
            dist22 = dist22[dist22 > 0].mean()
            loss = loss + dist22
            n_valid += 1
        
        """

        if n_valid > 0:
            return loss / n_valid
        else:
            return None

        # print(n_valid, dist12.sum())
        # return (dist12 + dist11 + dist22) / 3.

    def sem_feat_consistecny_loss(self, input):
        pred_feats = input["pred_feats"]
        gt_feats = input["gt_feats"]
        loss = 0
        for pfeat, gfeat in zip(pred_feats, gt_feats):
            if pfeat.shape[2] != gfeat.shape[2]:
                pfeat = F.interpolate(pfeat, size=(gfeat.shape[2], gfeat.shape[3]))

            if self.seg_feat == "l2":
                loss += F.mse_loss(pfeat, gfeat, reduction="mean")
            elif self.seg_feat == "l1":
                loss += torch.abs(pfeat - gfeat).mean()
            elif self.seg_feat == "cs":
                loss += (1. - torch.cosine_similarity(pfeat, gfeat, dim=1).mean())
        return loss / len(pred_feats)

    def forward(self, output):
        d = dict()
        cum_loss = 0

        if self.use_det:
            if self.seg_det is not None:
                if "seg_confidence" in output.keys():
                    seg_confidence = output["seg_confidence"]
                if "seg_mask" in output.keys():
                    seg_mask = output["seg_mask"]
        else:
            seg_confidence = None
            seg_mask = None

        det_loss = self.det_loss(pred_score=output["score"], gt_score=output["gt_score"], weight=output["weight"],
                                 seg_confidence=seg_confidence, seg_mask=seg_mask)

        cum_loss = cum_loss + det_loss * self.weights["det_loss"]
        d["det_loss"] = det_loss

        if self.use_desc:
            # desc_loss = self.desc_loss(pred_desc=output["desc"], gt_desc=output["gt_desc"], weight=output["weight"])
            desc_loss = self.desc_loss(pred_desc=output["desc"], gt_desc=output["gt_desc"], weight=output["ds_weight"])

            cum_loss = cum_loss + desc_loss * self.weights["desc_loss"]
            d["sup_desc_loss"] = desc_loss

        if self.seg_feat is not None:
            seg_feat_loss = self.sem_feat_consistecny_loss(input=output)
            cum_loss = cum_loss + seg_feat_loss * self.weights["seg_feat_loss"]
            d["seg_feat_loss"] = seg_feat_loss

        if self.seg_desc is not None:
            segmentation = output["seg"]
            if self.seg_desc == 'normal':
                seg_desc_loss = self.sem_desc_loss_wap(input=output, margin=self.margin)
            elif self.seg_desc == 'hard':
                seg_desc_loss = self.sem_desc_loss_hard(input=output, margin=self.margin)
            if seg_desc_loss is not None:
                cum_loss = cum_loss + seg_desc_loss * self.weights["seg_desc_loss"]
                d["seg_desc_loss"] = seg_desc_loss
            else:
                d["seg_desc_loss"] = torch.from_numpy(np.array([0], np.float)).cuda()
        else:
            segmentation = None

        if self.ap_loss_fun is not None:
            pred_score = output["score"].unsqueeze(1)
            pred_desc = output["desc"]
            # print("pred_score: ", pred_score.size())
            # print("pred_desc: ", pred_desc.size())
            if pred_score.size(2) != pred_desc.size(2) or pred_score.size(3) != pred_desc.size(3):
                pred_desc = F.interpolate(output["desc"], (pred_score.size(2), pred_score.size(3)), mode='bilinear',
                                          align_corners=False)

            b = pred_desc.size(0) // 2

            if self.use_pred:
                score = output["score"]
                score = score.clamp(min=0.001, max=1.) * 2. + 0.5
                score = score.clamp(min=0.001, max=1.)
            elif self.use_weight:
                score = output["gt_score"]
                score = score.clamp(min=0.001, max=1.) * 2. + 0.5
                score = score.clamp(min=0.001, max=1.)
            else:
                score = torch.ones_like(output["gt_score"])  # no preference for all keypoints
            # score = (score * 10).clamp(min=0.00001, max=1.)
            # score = score.clamp(min=0.001, max=1.) * 2. + 0.5
            # score = score.clamp(min=0.001, max=1.)
            if len(score.size()) == 3:
                score = score.unsqueeze(1)

            if segmentation is not None:
                segs = [segmentation[0:b], segmentation[b:]]
                seg_masks = [seg_mask[0:b], seg_mask[b:]]
            else:
                segs = None
                seg_masks = None
            ap_loss = self.ap_loss_fun(descriptors=[pred_desc[0:b], pred_desc[b:]], aflow=output["aflow"],
                                       reliability=[score[0:b], score[b:]],
                                       seg=segs,
                                       seg_mask=seg_masks)

            if ap_loss is not None:
                cum_loss = cum_loss + ap_loss * self.weights["ap_loss"]
                d["unsup_desc_loss"] = ap_loss
            else:
                d["unsup_desc_loss"] = torch.zeros_like(cum_loss)
        d["loss"] = cum_loss

        return d


class UnsupLoss(nn.Module):
    def __init__(self, ap_loss_fun, cos_fn, peak_fn, use_det=True, use_desc=False,
                 seg_desc=None, seg_det=None, desc_hard=False,
                 use_pred=False,
                 seg_feat=None,
                 weights=None,
                 margin=1.0,
                 use_weight=True, ):
        super().__init__()
        self.ap_loss_fun = ap_loss_fun
        self.cos_fn = cos_fn
        self.peak_fn = peak_fn
        self.use_det = use_det
        self.use_desc = use_desc
        self.use_pred = use_pred
        self.seg_desc = seg_desc
        self.seg_det = seg_det
        self.desc_hard = desc_hard
        self.seg_feat = seg_feat
        self.weights = weights
        self.margin = margin
        self.use_weight = use_weight

    def sem_desc_loss_wap_old(self, input, margin=1.0):
        # if self.use_pred:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        conf_th = input["score_th"]
        b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # print("sel_desc1: ", sel_desc1.size())
        # print("sel_seg1: ", sel_seg1.size())
        # print("sel_score1: ", sel_score1.size())

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        # dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.unsqueeze(0)
        # print("dist_map: ", dist_map12.size())
        # print("seg_cst_map: ", seg_const_map.size())
        # print("score_map: ", score_map12.size())
        # exit(0)

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        # print("pos_ids: ", torch.sum(pos_ids12), pos_ids12)
        # print("neg_ids: ", torch.sum(neg_ids12))

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

        dist12 = margin + pos_dist12 - neg_dist12
        return dist12

        # dist11 = sel_crrelation(desc=sel_desc1, segs=sel_seg1, score=sel_score1)
        # dist22 = sel_crrelation(desc=sel_desc2, segs=sel_seg2, score=sel_score2)
        #
        # return (dist12 + dist11 + dist22) / 3.
        # print("pos_num: ", pos_ids.sum())
        # print("neg_num: ", neg_ids.sum())

        # print("neg_dist: ", neg_dist)
        # print("pos_dist: ", pos_dist)
        # return pos_dist + torch.max(0, margin - neg_dist)
        # return margin + pos_dist12 - neg_dist12

    def sem_desc_loss_wap(self, input, margin=1.0):
        # if self.use_pred:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]
        # scores = input["reliability"].squeeze()
        scores = scores.squeeze()
        descs = input["desc"]  # default [B, D, H, W]
        b = descs.shape[0] // 2
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        # seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        # conf_th = torch.median(scores)
        values, _ = torch.topk(scores[masks].reshape((1, -1)), k=1000 * b * 2, largest=True, dim=1)
        conf_th = values[0, -1]
        # b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        # ids1 = masks[0:b]
        # ids2 = masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        # sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        # sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        # sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        # sel_score2 = sel_score2.clamp(min=0.0005, max=1.)
        #
        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # print("sel_desc1: ", sel_desc1.size())
        # print("sel_seg1: ", sel_seg1.size())
        # print("sel_score1: ", sel_score1.size())

        # print(conf_th, sel_desc1.shape, sel_desc2.shape)

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        # dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.unsqueeze(0)
        # print("dist_map: ", dist_map12.size())
        # print("seg_cst_map: ", seg_const_map.size())
        # print("score_map: ", score_map12.size())
        # exit(0)

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

        n_pos = torch.sum(pos_ids12) > 0
        n_neg = torch.sum(neg_ids12) > 0

        if n_pos == 0:
            pos_dist12 = torch.zeros(size=[], device=descs.device)
        if n_neg == 0:
            neg_dist12 = torch.zeros(size=[], device=descs.device)

        dist12 = torch.relu(margin + pos_dist12 - neg_dist12)

        return dist12

    def sem_feat_consistecny_loss(self, input):
        pred_feats = input["pred_feats"]
        gt_feats = input["gt_feats"]
        loss = 0
        for pfeat, gfeat in zip(pred_feats, gt_feats):
            if pfeat.shape[2] != gfeat.shape[2]:
                pfeat = F.interpolate(pfeat, size=(gfeat.shape[2], gfeat.shape[3]))

            if self.seg_feat == "l2":
                loss += F.mse_loss(pfeat, gfeat, reduction="mean")
            elif self.seg_feat == "l1":
                loss += torch.abs(pfeat - gfeat).mean()
            elif self.seg_feat == "cs":
                loss += (1. - torch.cosine_similarity(pfeat, gfeat, dim=1).mean())
        return loss / len(pred_feats)

    def forward(self, output):
        dev = output['desc'].device
        d = dict()
        cum_loss = 0

        if self.use_det:
            pred_semi = output['semi']
            if pred_semi.shape[1] == 2:
                gt_semi = output['gt_score']
                if len(gt_semi.shape) == 3:
                    gt_semi = gt_semi.unsqueeze(1)

                gt_semi = torch.cat([gt_semi, 1 - gt_semi], dim=1)  # [B, 2, H, W]
            else:
                gt_semi = output['gt_semi']
            # print(gt_semi.shape, output['semi'].shape)

            neg_log = gt_semi * torch.log(output['semi'])
            det_loss = -torch.sum(neg_log, dim=1)
            det_loss = det_loss.mean()

            if torch.isnan(det_loss):
                print('det loss is NaN')
            else:
                cum_loss = cum_loss + det_loss

            reliability = output['gt_score']
            reliability = reliability.clamp(min=0.0005, max=1.) * 2. + 0.5
            reliability = reliability.clamp(min=0.0005, max=1.)
            if len(reliability.shape) == 3:
                reliability = reliability.unsqueeze(1)
        else:
            reliability = output['reliability']

        if self.seg_det is not None:
            if "seg_confidence" in output.keys():
                seg_confidence = output["seg_confidence"]
            if "seg_mask" in output.keys():
                seg_mask = output["seg_mask"]
            sta_loss = F.binary_cross_entropy(output['stability'].squeeze(), seg_confidence, reduce=False)
            sta_loss = torch.mean(sta_loss[seg_mask])

            if torch.isnan(sta_loss):
                print('sta loss is NaN')
                sta_loss = torch.zeros(size=[], device=dev)

            d["sta_loss"] = sta_loss
            cum_loss = cum_loss + sta_loss

        if self.cos_fn is not None:
            repeatability = output['repeatability']
            b = repeatability.shape[0] // 2
            cos_loss = self.cos_fn(repeatability=(repeatability[:b], repeatability[b:]), aflow=output["aflow"])
            if torch.isnan(cos_loss):
                print('cos loss is NaN')
                cos_loss = torch.zeros(size=[], device=dev)
        else:
            cos_loss = torch.zeros(size=[], device=dev)

        if self.peak_fn is not None:
            peak_loss = self.peak_fn(repeatability=(repeatability[:b], repeatability[b:]), aflow=output["aflow"])
            if torch.isnan(peak_loss):
                print('peak loss is NaN')
                peak_loss = torch.zeros(size=[], device=dev)

            cum_loss = cum_loss + cos_loss + peak_loss
        else:
            peak_loss = torch.zeros(size=[], device=dev)

        d["cos_loss"] = cos_loss
        d["peak_loss"] = peak_loss

        if self.use_desc:
            # desc_loss = self.desc_loss(pred_desc=output["desc"], gt_desc=output["gt_desc"], weight=output["weight"])
            desc_loss = self.desc_loss(pred_desc=output["desc"], gt_desc=output["gt_desc"], weight=output["ds_weight"])

            if torch.isnan(desc_loss):
                print('desc loss is NaN')
                desc_loss = torch.zeros(size=[], device=dev)

            cum_loss = cum_loss + desc_loss * self.weights["desc_loss"]
            d["sup_desc_loss"] = desc_loss

        if self.seg_feat is not None:
            seg_feat_loss = self.sem_feat_consistecny_loss(input=output)
            if torch.isnan(seg_feat_loss):
                print('seg feat loss is NaN')
                seg_feat_loss = torch.zeros(size=[], device=dev)

            cum_loss = cum_loss + seg_feat_loss * self.weights["seg_feat_loss"]
            d["seg_feat_loss"] = seg_feat_loss

        if self.seg_desc is not None:
            segmentation = output["seg"]
            if self.seg_desc == 'normal':
                seg_desc_loss = self.sem_desc_loss_wap(input=output, margin=self.margin)
            elif self.seg_desc == 'hard':
                seg_desc_loss = self.sem_desc_loss_hard(input=output, margin=self.margin)

            if torch.isnan(seg_desc_loss):
                print('seg desc loss is NaN')
                seg_desc_loss = torch.zeros(size=[], device=dev)

            if seg_desc_loss is not None:
                cum_loss = cum_loss + seg_desc_loss * self.weights["seg_desc_loss"]
                d["seg_desc_loss"] = seg_desc_loss
            else:
                d["seg_desc_loss"] = torch.from_numpy(np.array([0], np.float)).cuda()
        else:
            segmentation = None

        pred_desc = output["desc"]
        b = pred_desc.shape[0] // 2
        if segmentation is not None:
            segs = [segmentation[0:b], segmentation[b:]]
            seg_masks = [seg_mask[0:b], seg_mask[b:]]
        else:
            segs = None
            seg_masks = None

        # reliability = output['reliability']
        ap_loss = self.ap_loss_fun(descriptors=[pred_desc[0:b], pred_desc[b:]], aflow=output["aflow"],
                                   reliability=[reliability[0:b], reliability[b:]],
                                   seg=segs,
                                   seg_mask=seg_masks)
        if torch.isnan(ap_loss):
            print('ap loss is NaN')
            ap_loss = torch.zeros(size=[], device=dev)

        if ap_loss is not None:
            cum_loss = cum_loss + ap_loss * self.weights["ap_loss"]
            d["unsup_desc_loss"] = ap_loss
        else:
            d["unsup_desc_loss"] = torch.zeros_like(cum_loss)

        d["loss"] = cum_loss

        return d


class SwinLoss(nn.Module):
    def __init__(self, desc_loss_fn,
                 weights,
                 det_loss='bce',
                 seg_desc_loss_fn='wap',
                 use_pred_score_desc=False,
                 upsample_desc=False,
                 seg_desc=False,
                 seg_feat=False,
                 seg_det=False,
                 seg_cls=False,
                 margin=1.0,
                 ):
        super(SwinLoss, self).__init__()
        self.desc_loss_fn = desc_loss_fn
        self.weights = weights
        self.use_pred_score_desc = use_pred_score_desc
        self.detloss = det_loss
        self.seg_desc_loss_fn = seg_desc_loss_fn
        self.upsample_desc = upsample_desc

        self.seg_desc = seg_desc
        self.seg_feat = seg_feat
        self.seg_det = seg_det
        self.seg_cls = seg_cls
        self.margin = margin

        if self.seg_cls:
            self.det_seg_cls_func = torch.nn.CrossEntropyLoss().cuda()

    def sem_desc_loss_wap_ori(self, input, margin=1.0):
        # if self.use_pred:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        conf_th = input["score_th"]
        b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # print("sel_desc1: ", sel_desc1.size())
        # print("sel_seg1: ", sel_seg1.size())
        # print("sel_score1: ", sel_score1.size())

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        # dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.unsqueeze(0)
        # print("dist_map: ", dist_map12.size())
        # print("seg_cst_map: ", seg_const_map.size())
        # print("score_map: ", score_map12.size())
        # exit(0)

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        # print("pos_ids: ", torch.sum(pos_ids12), pos_ids12)
        # print("neg_ids: ", torch.sum(neg_ids12))

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

        dist12 = margin + pos_dist12 - neg_dist12
        return dist12

    def sem_desc_loss_wap_ds(self, input, margin=1.0):
        if self.use_pred_score_desc:
            scores = input["score"]
        else:
            scores = input["gt_score"]
        scores = scores.squeeze()
        descs = input["desc"]  # default [B, D, H, W]
        b = descs.shape[0] // 2
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]
        scale = scores.shape[1] // descs.shape[1]
        values, _ = torch.topk(scores.reshape((1, -1)), k=1000 * b * 2, largest=True, dim=1)
        conf_th = values[0, -1]

        bids1, bhs1, bws1 = torch.where(scores[0:b] >= conf_th)  # (b, h, w)
        bids2, bhs2, bws2 = torch.where(scores[b:] >= conf_th)  # (b, h, w)
        sel_mask1 = masks[0:b][bids1, bhs1, bws1]
        sel_mask2 = masks[b:][bids2, bhs2, bws2]

        bids1 = bids1[sel_mask1]
        bhs1 = bhs1[sel_mask1]
        bws1 = bws1[sel_mask1]
        bids2 = bids2[sel_mask2]
        bhs2 = bhs2[sel_mask2]
        bws2 = bws2[sel_mask2]

        sel_seg1 = segs[0:b][bids1, bhs1, bws1]
        sel_seg2 = segs[b:][bids2, bhs2, bws2]

        # print('bhws1: ', bids1.shape, bhs1.shape, bws1.shape, sel_seg1.shape)
        # print('bhws2: ', bids2.shape, bhs2.shape, bws2.shape, sel_seg2.shape)

        sel_score1 = scores[0:b][bids1, bhs1, bws1]
        sel_score2 = scores[b:][bids2, bhs2, bws2]

        bhs1_ds = self.downscale_positions(pos=bhs1, scaling_steps=scale).long()
        bws1_ds = self.downscale_positions(pos=bws1, scaling_steps=scale).long()
        bhs2_ds = self.downscale_positions(pos=bhs2, scaling_steps=scale).long()
        bws2_ds = self.downscale_positions(pos=bws2, scaling_steps=scale).long()

        sel_desc1 = descs[0:b][bids1, bhs1_ds, bws1_ds]
        sel_desc2 = descs[b:][bids2, bhs2_ds, bws2_ds]

        sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1  # * seg_confidences[0:b][bids1, bhs1, bws1]
        sel_score2 = sel_score2  # * seg_confidences[b:][bids2, bhs2, bws2]

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        # dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.unsqueeze(0)
        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

        n_pos = torch.sum(pos_ids12) > 0
        n_neg = torch.sum(neg_ids12) > 0

        if n_pos == 0:
            pos_dist12 = torch.zeros(size=[], device=descs.device)
        if n_neg == 0:
            neg_dist12 = torch.zeros(size=[], device=descs.device)

        # dist12 = torch.relu(margin + pos_dist12 - neg_dist12)
        dist12 = margin + pos_dist12 - neg_dist12

        return dist12

    def sem_desc_loss_wap_ds_two_margin(self, input, margin_inter=1.0, margin_intra=1.5, with_self=False):
        def cross_dist(desc1, desc2, score1, score2, seg1, seg2):
            dist_map12 = desc1 @ desc2.t()  # M x N
            dist_map12 = 2 - 2 * dist_map12
            seg_const_map12 = torch.abs(seg1.unsqueeze(1) - seg2.unsqueeze(0))  # M x N
            score_map12 = score1.unsqueeze(1) * score2.unsqueeze(0)
            pos_ids12 = (seg_const_map12 == 0)  # with the same label
            neg_ids12 = (seg_const_map12 > 0)  # with different labels

            # neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
            # pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

            pos_dist12 = torch.mean(F.relu(dist_map12[pos_ids12] - margin_inter) * score_map12[
                pos_ids12])  # minimize those with dist > margin_inter
            neg_dist12 = torch.mean(F.relu(margin_intra - dist_map12[neg_ids12]) * score_map12[
                neg_ids12])  # maximize those with dist < margin_intra

            n_pos = torch.sum(pos_ids12) > 0
            n_neg = torch.sum(neg_ids12) > 0

            if n_pos == 0:
                pos_dist12 = torch.zeros(size=[], device=descs.device)
            if n_neg == 0:
                neg_dist12 = torch.zeros(size=[], device=descs.device)

            # dist12 = torch.relu(margin + pos_dist12 - neg_dist12)
            # dist12 = margin + pos_dist12 - neg_dist12
            dist12 = pos_dist12 + neg_dist12

            return dist12

        # if self.use_pred_score_desc:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]
        scores = scores.squeeze()
        descs = input["desc"]  # default [B, D, H, W]
        b = descs.shape[0] // 2
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]
        scale = scores.shape[1] // descs.shape[1]
        values, _ = torch.topk(scores.reshape((1, -1)), k=1000 * b * 2, largest=True, dim=1)
        conf_th = values[0, -1]

        bids1, bhs1, bws1 = torch.where(scores[0:b] >= conf_th)  # (b, h, w)
        bids2, bhs2, bws2 = torch.where(scores[b:] >= conf_th)  # (b, h, w)
        sel_mask1 = masks[0:b][bids1, bhs1, bws1]
        sel_mask2 = masks[b:][bids2, bhs2, bws2]

        bids1 = bids1[sel_mask1]
        bhs1 = bhs1[sel_mask1]
        bws1 = bws1[sel_mask1]
        bids2 = bids2[sel_mask2]
        bhs2 = bhs2[sel_mask2]
        bws2 = bws2[sel_mask2]

        sel_seg1 = segs[0:b][bids1, bhs1, bws1]
        sel_seg2 = segs[b:][bids2, bhs2, bws2]

        # print('bhws1: ', bids1.shape, bhs1.shape, bws1.shape, sel_seg1.shape)
        # print('bhws2: ', bids2.shape, bhs2.shape, bws2.shape, sel_seg2.shape)

        sel_score1 = scores[0:b][bids1, bhs1, bws1]
        sel_score2 = scores[b:][bids2, bhs2, bws2]

        bhs1_ds = self.downscale_positions(pos=bhs1, scaling_steps=scale).long()
        bws1_ds = self.downscale_positions(pos=bws1, scaling_steps=scale).long()
        bhs2_ds = self.downscale_positions(pos=bhs2, scaling_steps=scale).long()
        bws2_ds = self.downscale_positions(pos=bws2, scaling_steps=scale).long()

        sel_desc1 = descs[0:b][bids1, bhs1_ds, bws1_ds]
        sel_desc2 = descs[b:][bids2, bhs2_ds, bws2_ds]

        sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.)

        # apply semantic confidence
        # sel_score1 = sel_score1  # * seg_confidences[0:b][bids1, bhs1, bws1]
        # sel_score2 = sel_score2  # * seg_confidences[b:][bids2, bhs2, bws2]

        dist12 = cross_dist(desc1=sel_desc1, desc2=sel_desc2, score1=sel_score1, score2=sel_score2, seg1=sel_seg1,
                            seg2=sel_seg2)
        if with_self:
            dist11 = cross_dist(desc1=sel_desc1, desc2=sel_desc1, score1=sel_score1, score2=sel_score1, seg1=sel_seg1,
                                seg2=sel_seg1)
            dist22 = cross_dist(desc1=sel_desc2, desc2=sel_desc2, score1=sel_score2, score2=sel_score2, seg1=sel_seg2,
                                seg2=sel_seg2)
            dist = (dist12 + dist11 + dist22) / 3.
            return dist
        else:
            return dist12

    def sem_desc_loss_wap(self, input, margin=1.0):
        # if self.use_pred:
        #     scores = input["score"]
        # else:
        scores = input["gt_score"]
        # scores = input["reliability"].squeeze()
        scores = scores.squeeze()
        descs = input["desc"]  # default [B, D, H, W]
        b = descs.shape[0] // 2
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        # seg_confidences = input["seg_confidence"]

        # print("scores: ", scores.shape)
        # print("descs: ", descs.shape)
        # print("masks: ", masks.shape)
        # print("segs: ", segs.shape)

        # conf_th = torch.median(scores)
        values, _ = torch.topk(scores[masks].reshape((1, -1)), k=1000 * b * 2, largest=True, dim=1)
        conf_th = values[0, -1]
        # b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        # ids1 = masks[0:b]
        # ids2 = masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # sel_score1 = sel_score1.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score1 = sel_score1.clamp(min=0.001, max=1.)
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.) * 2. + 0.5
        # sel_score2 = sel_score2.clamp(min=0.001, max=1.)

        # sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        # sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        # sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        # sel_score2 = sel_score2.clamp(min=0.0005, max=1.)
        #
        # apply semantic confidence
        # sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        # sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        # print("sel_desc1: ", sel_desc1.size())
        # print("sel_seg1: ", sel_seg1.size())
        # print("sel_score1: ", sel_score1.size())

        # print(conf_th, sel_desc1.shape, sel_desc2.shape)

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        # dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.unsqueeze(0)
        # print("dist_map: ", dist_map12.size())
        # print("seg_cst_map: ", seg_const_map.size())
        # print("score_map: ", score_map12.size())
        # exit(0)

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

        neg_dist12 = torch.mean(dist_map12[neg_ids12] * score_map12[neg_ids12])
        pos_dist12 = torch.mean(dist_map12[pos_ids12] * score_map12[pos_ids12])

        n_pos = torch.sum(pos_ids12) > 0
        n_neg = torch.sum(neg_ids12) > 0

        if n_pos == 0:
            pos_dist12 = torch.zeros(size=[], device=descs.device)
        if n_neg == 0:
            neg_dist12 = torch.zeros(size=[], device=descs.device)

        dist12 = torch.relu(margin + pos_dist12 - neg_dist12)

        return dist12

    def sem_feat_consistecny_loss(self, input):
        pred_feats = input["pred_feats"]
        gt_feats = input["gt_feats"]
        loss = 0
        for pfeat, gfeat in zip(pred_feats, gt_feats):
            # print(pfeat.shape, gfeat.shape)
            if pfeat.shape[2] != gfeat.shape[2]:
                pfeat = F.interpolate(pfeat, size=(gfeat.shape[2], gfeat.shape[3]))

            loss += torch.abs(pfeat - gfeat).mean()

            # if self.seg_feat == "l2":
            #     loss += F.mse_loss(pfeat, gfeat, reduction="mean")
            # elif self.seg_feat == "l1":
            #     loss += torch.abs(pfeat - gfeat).mean()
            # elif self.seg_feat == "cs":
            #     loss += (1. - torch.cosine_similarity(pfeat, gfeat, dim=1).mean())
        return loss / len(pred_feats)

    def det_loss(self, pred_score, gt_score, weight=None, stability_map=None):
        if self.detloss == 'l1':
            if stability_map is not None:
                loss = torch.abs(pred_score - gt_score * stability_map)
            else:
                loss = torch.abs(pred_score - gt_score)

            if weight is not None:
                loss = loss * weight

        elif self.detloss == 'bce':
            if stability_map is not None:
                loss = F.binary_cross_entropy(pred_score, gt_score * stability_map, reduce=False)
            else:
                loss = F.binary_cross_entropy(pred_score, gt_score, reduce=False)

            if weight is not None:
                loss = loss * weight
        elif self.detloss in ['ce', 'sce']:
            neg_log = gt_score * torch.log(pred_score)
            loss = -torch.sum(neg_log, dim=1)

        return torch.mean(loss)

    def forward(self, output):
        dev = output['desc'].device
        d = dict()
        cum_loss = 0

        pred_desc = output["desc"]
        b = pred_desc.size(0) // 2

        if self.use_pred_score_desc:
            score = output["score"]
        else:
            score = output["gt_score"]

        score = score.clamp(min=0.0005, max=1.) * 4. + 0.5
        score = score.clamp(min=0.0005, max=1.)

        # if pred_desc.shape[2] != score.shape[2] or pred_desc.shape[3] != score.shape[3]:
        if self.upsample_desc:
            pred_desc = self.grid_sample_desc(desc=output["desc"], ref=score)
            pred_desc = torch.nn.functional.normalize(pred_desc, p=2, dim=1)

        if self.detloss in ['l1', 'bce']:
            det_loss = self.det_loss(pred_score=output["score"], gt_score=output["gt_score"], weight=output["weight"],
                                     stability_map=None)
        elif self.detloss in ['ce']:
            det_loss = self.det_loss(pred_score=output["semi"], gt_score=output["gt_semi"], weight=output["weight"],
                                     stability_map=None)

        elif self.detloss in ['sce']:
            if "seg_confidence" in output.keys():
                seg_confidence = output["seg_confidence"]
            if "seg_mask" in output.keys():
                seg_mask = output["seg_mask"]

            seg_confidence[~seg_mask] = 1.
            r = seg_confidence
            gt_score = output['gt_semi']
            a = gt_score[:, :64, :, :]
            Hc, Wc = a.size(2), a.size(3)
            a = a.permute([0, 2, 3, 1])
            a = a.view(a.size(0), Hc, Wc, 8, 8)
            a = a.permute([0, 1, 3, 2, 4])
            a = a.reshape(a.size(0), Hc * 8, Wc * 8)

            m = r - r * a / (1 - r * a)
            m = m.view(m.size(0), Hc, 8, Wc, 8)
            m = m.permute([0, 1, 3, 2, 4]).reshape(m.size(0), Hc, Wc, 64).permute([0, 3, 1, 2])

            sgt_score = torch.cat([m, gt_score[:, 64:, :, :]], dim=1)
            sgt_score = sgt_score / torch.sum(sgt_score, dim=1, keepdim=True)
            det_loss = self.det_loss(pred_score=output["semi"], gt_score=gt_score, weight=output["weight"],
                                     stability_map=None)
            if torch.isnan(det_loss):
                print('seg det loss is NaN')
                det_loss = torch.zeros(size=[], device=dev)

            # cum_loss = cum_loss + seg_det_loss * self.weights["seg_det_loss"]
            # d["seg_det_loss"] = seg_det_loss

        cum_loss = cum_loss + det_loss * self.weights["det_loss"]
        d["det_loss"] = det_loss

        desc_loss = self.desc_loss_fn(descriptors=[pred_desc[0:b], pred_desc[b:]],
                                      aflow=output["aflow"],
                                      reliability=[score[0:b], score[b:]],
                                      output=output)

        if torch.isnan(desc_loss):
            print('desc loss is nan')
        if torch.isnan(det_loss):
            print('det loss is nan')
        # print('un_fn: ', self.desc_loss_fn, desc_loss)

        d["unsup_desc_loss"] = desc_loss
        cum_loss = cum_loss + desc_loss * self.weights["desc_loss"]

        if self.seg_det:
            if "seg_confidence" in output.keys():
                seg_confidence = output["seg_confidence"]
            if "seg_mask" in output.keys():
                seg_mask = output["seg_mask"]
            if not self.seg_cls:
                seg_det_loss = F.binary_cross_entropy(output['stability'].squeeze(), seg_confidence, reduce=False)
                seg_det_loss = torch.mean(seg_det_loss[seg_mask])
            else:
                # print(seg_confidence.shape, output['stability'].shape)
                # print(seg_confidence)
                # print(output['stability'])
                gt_seg_cls = torch.ones_like(seg_confidence)
                gt_seg_cls[seg_confidence == 0.1] = 0
                gt_seg_cls[seg_confidence == 0.5] = 1
                gt_seg_cls[seg_confidence == 1.0] = 2

                seg_det_loss = self.det_seg_cls_func(output['stability'], gt_seg_cls.long())

            if torch.isnan(seg_det_loss):
                print('seg det loss is NaN')
                seg_det_loss = torch.zeros(size=[], device=dev)

            cum_loss = cum_loss + seg_det_loss * self.weights["seg_det_loss"]
            d["seg_det_loss"] = seg_det_loss

        if self.seg_feat:
            seg_feat_loss = self.sem_feat_consistecny_loss(input=output)
            if torch.isnan(seg_feat_loss):
                print('seg feat loss is NaN')
                seg_feat_loss = torch.zeros(size=[], device=dev)

            cum_loss = cum_loss + seg_feat_loss * self.weights["seg_feat_loss"]
            d["seg_feat_loss"] = seg_feat_loss

        if self.seg_desc:
            if self.seg_desc_loss_fn == 'wap':
                seg_desc_loss = self.sem_desc_loss_wap_ds(input=output, margin=self.margin)
            elif self.seg_desc_loss_fn == '2m':
                seg_desc_loss = self.sem_desc_loss_wap_ds_two_margin(input=output, margin_inter=1.0, margin_intra=1.0,
                                                                     with_self=False)
            elif self.seg_desc_loss_fn == '2mf':
                seg_desc_loss = self.sem_desc_loss_wap_ds_two_margin(input=output, margin_inter=1.0, margin_intra=1.0,
                                                                     with_self=True)
            if torch.isnan(seg_desc_loss):
                print('seg desc loss is NaN')
                seg_desc_loss = torch.zeros(size=[], device=dev)
            cum_loss = cum_loss + seg_desc_loss * self.weights["seg_desc_loss"]
            d["seg_desc_loss"] = seg_desc_loss

        d["loss"] = cum_loss
        return d

    def grid_sample_desc(self, desc, ref):
        B, C, H, W = ref.shape
        with torch.no_grad():
            full_y1, full_x1 = self.gen_xy(step=1, feat=ref, border=0)
            norm_x = full_x1.float() / (float(W) / 2.) - 1.
            norm_y = full_y1.float() / (float(H) / 2.) - 1.
            norm_xys = torch.stack([norm_x, norm_y]).transpose(0, 1)
            norm_xys = norm_xys.view(1, 1, -1, 2)  # [1, 1, N, 2]

            norm_xys_batch = []
            for i in range(B):
                norm_xys_batch.append(norm_xys)
            norm_xys_batch = torch.cat(norm_xys_batch, dim=0)

        D = desc.shape[1]
        desc_up = F.grid_sample(desc, norm_xys_batch, align_corners=False).view(B, D, H, W)
        return desc_up

    def upscale_positions(self, pos, scaling_steps=0):
        for _ in range(scaling_steps):
            pos = pos * 2 + 0.5
        return pos

    def downscale_positions(self, pos, scaling_steps=0):
        for _ in range(scaling_steps):
            pos = (pos - 0.5) / 2
        return pos


class MultiLoss(nn.Module):
    """ Combines several loss functions for convenience.
    *args: [loss weight (float), loss creator, ... ]
    
    Example:
        loss = MultiLoss( 1, MyFirstLoss(), 0.5, MySecondLoss() )
    """

    def __init__(self, *args, dbg=()):
        nn.Module.__init__(self)
        assert len(args) % 2 == 0, 'args must be a list of (float, loss)'
        self.weights = []
        self.losses = nn.ModuleList()
        for i in range(len(args) // 2):
            weight = float(args[2 * i + 0])
            loss = args[2 * i + 1]
            assert isinstance(loss, nn.Module), "%s is not a loss!" % loss
            self.weights.append(weight)
            self.losses.append(loss)

    def forward(self, select=None, **variables):
        assert not select or all(1 <= n <= len(self.losses) for n in select)
        d = dict()
        cum_loss = 0
        for num, (weight, loss_func) in enumerate(zip(self.weights, self.losses), 1):
            if select is not None and num not in select: continue
            l = loss_func(**{k: v for k, v in variables.items()})
            if isinstance(l, tuple):
                assert len(l) == 2 and isinstance(l[1], dict)
            else:
                l = l, {loss_func.name: l}
            cum_loss = cum_loss + weight * l[0]
            for key, val in l[1].items():
                d['loss_' + key] = float(val)
        d['loss'] = float(cum_loss)
        return cum_loss, d
