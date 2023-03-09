# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import pdb
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nets.ap_loss import APLoss


class PixelAPLoss(nn.Module):
    """ Computes the pixel-wise AP loss:
        Given two images and ground-truth optical flow, computes the AP per pixel.
        
        feat1:  (B, C, H, W)   pixel-wise features extracted from img1
        feat2:  (B, C, H, W)   pixel-wise features extracted from img2
        aflow:  (B, 2, H, W)   absolute flow: aflow[...,y1,x1] = x2,y2
    """

    def __init__(self, sampler, nq=20):
        nn.Module.__init__(self)
        self.aploss = APLoss(nq, min=0, max=1, euc=False)
        self.name = 'pixAP'
        self.sampler = sampler

    def loss_from_ap(self, ap, rel):
        return 1 - ap

    def forward(self, descriptors, aflow, **kw):
        # subsample things
        scores, gt, msk, qconf = self.sampler(descriptors, kw.get('reliability'), aflow, seg=kw.get('seg'))
        # scores, gt, msk, qconf = self.sampler(descriptors, None, aflow)

        # compute pixel-wise AP
        n = qconf.numel()
        if n == 0: return 0
        scores, gt = scores.view(n, -1), gt.view(n, -1)
        ap = self.aploss(scores, gt).view(msk.shape)
        # print("qconf: ", qconf.shape, torch.max(qconf), torch.min(qconf), torch.median(qconf))

        pixel_loss = self.loss_from_ap(ap, qconf)

        loss = pixel_loss[msk].mean()
        return loss


class ReliabilityLoss(PixelAPLoss):
    """ same than PixelAPLoss, but also train a pixel-wise confidence
        that this pixel is going to have a good AP.
    """

    def __init__(self, sampler, base=0.5, use_rel=True, **kw):
        PixelAPLoss.__init__(self, sampler, **kw)
        assert 0 <= base < 1
        self.base = base
        self.name = 'reliability'
        self.use_rel = use_rel

    def loss_from_ap(self, ap, rel):
        if not self.use_rel:
            return 1 - ap  # - self.base
        else:
            # num = torch.sum(rel >= 0.53)
            # print("rel: ", torch.max(rel), torch.median(rel), torch.min(rel), num, rel.shape)
            return 1 - ap * rel - (1 - rel) * self.base


class TripletLoss(nn.Module):
    def __init__(self, step=8, margin=1., border=16, v=1, hard=True):
        super(TripletLoss, self).__init__()
        self.step = step
        self.margin = margin
        self.border = border
        self.v = v
        self.hard = hard

    def gen_grid(self, step, aflow):
        # print("step: ", step)
        B, two, H, W = aflow.shape
        dev = aflow.device
        b1 = torch.arange(B, device=dev)
        if step > 0:
            # regular grid
            x1 = torch.arange(self.border, W - self.border, step, device=dev)
            y1 = torch.arange(self.border, H - self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            x1 = x1[None, None, :].expand(B, H1, W1).reshape(-1)
            y1 = y1[None, :, None].expand(B, H1, W1).reshape(-1)
            b1 = b1[:, None, None].expand(B, H1, W1).reshape(-1)
            shape = (B, H1, W1)
        else:
            # randomly spread
            n = (H - 2 * self.border) * (W - 2 * self.border) // step ** 2
            x1 = torch.randint(self.border, W - self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H - self.border, (n,), device=dev)
            x1 = x1[None, :].expand(B, n).reshape(-1)
            y1 = y1[None, :].expand(B, n).reshape(-1)
            b1 = b1[:, None].expand(B, n).reshape(-1)
            shape = (B, n)
        return b1, y1, x1, shape

    def gen_xy(self, step, aflow):
        B, two, H, W = aflow.shape
        dev = aflow.device
        if step > 0:
            x1 = torch.arange(self.border, W - self.border, step, device=dev)
            y1 = torch.arange(self.border, H - self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            x1 = x1[None, :].expand(H1, W1).reshape(-1)
            y1 = y1[:, None].expand(H1, W1).reshape(-1)
        else:
            n = (H - 2 * self.border) * (W - 2 * self.border) // step ** 2
            x1 = torch.randint(self.border, W - self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H - self.border, (n,), device=dev)

        return y1, x1

    def gen_xy_by_conf(self, conf, th=0.005):
        pass

    def forward(self, descriptors, aflow, **kw):
        if self.v == 1:
            return self.forward_v1(descriptors, aflow, **kw)
        elif self.v == 2:
            return self.forward_v2(descriptors, aflow, **kw)
        elif self.v == 3:
            return self.forward_v3(descriptors, aflow, **kw)

    def forward_v1(self, descriptors, aflow, **kw):
        confs = kw.get('reliability')
        B, two, H, W = aflow.shape
        assert two == 2
        feat1, conf1 = descriptors[0], (confs[0] if confs else None)
        feat2, conf2 = descriptors[1], (confs[1] if confs else None)

        # b1c, y1c, x1c = self.gen_samples(confs=conf1)
        # b2c, y2c, x2c = self.gen_samples(confs=conf2)

        # feat1c = feat1[b1c, :, y1c, x1c]
        # feat2c = feat2[b2c, :, y2c, x2c]
        # qconf = conf1[b1c, :, y1c, x1c].view(-1) if confs else None
        # b2 = b1c
        # xy2 = (aflow[b1c, :, y1c, x1c] + 0.5).long().t()
        # mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
        # mask = mask.view(-1)

        # positions in the first image
        with torch.no_grad():
            y1, x1 = self.gen_xy(step=self.step, aflow=aflow)
        pos1_all = torch.stack([x1, y1])
        pos2_all = torch.stack([x1, y1])

        # print("pos1_all: ", pos1_all.size())
        # print("pos2_all: ", pos2_all.size())

        loss = torch.tensor(np.array([0], dtype=np.float32), device=aflow.device)
        n_valid_samples = 0
        for b in range(B):
            # sample GT from second image
            xy2 = (aflow[b, :, y1, x1] + 0.5).long()  # .t()
            # print("xy2: ", xy2.size())
            mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
            x2_match = xy2[0][mask]
            y2_match = xy2[1][mask]

            if x2_match.size(0) < 64:
                print("correspondence: ", x2_match.size(0))
                continue

            feat1_all = feat1[b]
            feat2_all = feat2[b]
            conf1_all = conf1[b].squeeze()
            conf2_all = conf2[b].squeeze()
            # print("f1_a: ", feat1_all.size())
            # print("f2_a: ", feat2_all.size())
            # print("c1_a: ", conf1_all.size())
            # print("c2_a: ", conf2_all.size())
            # qconf1 = conf1[b, y1, x1].view(-1) if confs else None
            # print("xy2: ", xy2.size())
            # print("x2y2: ", x2_match.size(), y2_match.size())
            # print("mask: ", mask.size())

            feat1_match = feat1_all[:, y1[mask], x1[mask]]
            feat2_match = feat2_all[:, y2_match, x2_match]

            # print("f1m", feat1_match.size())
            # print("f2m", feat2_match.size())

            pos_dist = feat1_match.t().unsqueeze(1) @ feat2_match.t().unsqueeze(2)
            pos_dist = 2 - 2 * pos_dist.squeeze()
            # print("pos_dist: ", pos_dist.size())

            x1_match = x1[mask]
            y1_match = y1[mask]

            pos2_diff = torch.abs(x2_match.unsqueeze(1) - pos2_all[0].unsqueeze(0)) + torch.abs(
                y2_match.unsqueeze(1) - pos2_all[1].unsqueeze(0))
            neg_dist1 = 2 - 2 * (feat1_match.t() @ feat2_all[:, pos2_all[1], pos2_all[0]])

            # print("pos2_diff: ", pos2_diff.size(), torch.min(pos2_diff), torch.max(pos2_diff))
            # print("neg_dist1: ", neg_dist1.size())
            #
            # print(torch.max(neg_dist1 + (pos2_diff < 1) * 10,
            #                 dim=1)[0])

            neg_dist1 = torch.min(
                neg_dist1 + (pos2_diff < 3) * 10,
                dim=1
            )[0]
            # print("neg_dist1: ", neg_dist1.size())
            # ids = (neg_dist1 + (pos2_diff < 1) * 10) > 10
            # print("ids:", ids)
            # print("mask", mask)
            # exit(0)
            pos1_diff = torch.abs(x1_match.unsqueeze(1) - pos1_all[0].unsqueeze(0)) + torch.abs(
                y1_match.unsqueeze(1) - pos1_all[1].unsqueeze(0))
            neg_dist2 = 2 - 2 * (feat2_match.t() @ feat1_all[:, pos1_all[1], pos1_all[0]])
            neg_dist2 = torch.min(
                neg_dist2 + (pos1_diff < 3) * 10,
                dim=1
            )[0]

            # print("pos_dist: ", pos_dist.size(), torch.min(pos_dist), torch.max(pos_dist))
            # print("neg_dist1: ", neg_dist1.size(), torch.min(neg_dist1), torch.max(neg_dist1))
            # print("neg_dist2: ", neg_dist2.size(), torch.min(neg_dist2), torch.max(neg_dist2))
            # exit(0)
            #
            diff = F.relu(self.margin + pos_dist - torch.min(neg_dist1, neg_dist2))
            conf1_match = conf1_all[y1_match, x1_match]
            conf2_match = conf2_all[y2_match, x2_match]

            # print("conf1_m: ", conf1_match.size())
            # print("conf2_m: ", conf2_match.size())

            diff = diff * ((conf1_match + conf2_match) / 2.)
            loss = loss + diff.mean()

            # print("diff: ", diff.mean())
            n_valid_samples += 1

        if n_valid_samples == 0:
            return None
        else:
            return loss / n_valid_samples

    def forward_v2(self, descriptors, aflow, **kw):
        confs = kw.get('reliability')
        B, two, H, W = aflow.shape
        assert two == 2
        feat1, conf1 = descriptors[0], (confs[0] if confs else None)
        feat2, conf2 = descriptors[1], (confs[1] if confs else None)

        # positions in the first image
        with torch.no_grad():
            y, x = self.gen_xy(step=1, aflow=aflow)
        # print("pos1_all: ", pos1_all.size())
        # print("pos2_all: ", pos2_all.size())

        loss = torch.tensor(np.array([0], dtype=np.float32), device=aflow.device)
        n_valid_samples = 0
        for b in range(B):
            feat1_all = feat1[b]
            feat2_all = feat2[b]
            conf1_all = conf1[b].squeeze()
            conf2_all = conf2[b].squeeze()
            # pos1_all = torch.stack([y1, x1])
            # pos2_all = torch.stack([y1, x1])

            # print("f1_a: ", feat1_all.size())
            # print("f2_a: ", feat2_all.size())
            # print("c1_a: ", conf1_all.size())
            # print("c2_a: ", conf2_all.size())
            # qconf1 = conf1[b, y1, x1].view(-1) if confs else None

            conf1_vec = conf1_all[y, x]
            mask = (conf1_vec >= 0.52)
            y1 = y[mask]
            x1 = x[mask]
            pos1_all = torch.stack([y1, x1])

            conf2_vec = conf2_all[y, x]
            mask = (conf2_vec >= 0.52)
            y2 = y[mask]
            x2 = x[mask]
            pos2_all = torch.stack([y2, x2])

            # sample GT from second image
            xy2 = (aflow[b, :, y1, x1] + 0.5).long()  # .t()
            # print("xy2: ", xy2.size())
            mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
            x2_match = xy2[0][mask]
            y2_match = xy2[1][mask]

            if x2_match.size(0) < 64 or pos1_all.shape[1] < 64 or pos2_all.shape[1] < 64:
                continue
            # print("xy2: ", xy2.shape, pos1_all.shape, pos2_all.shape, x.shape)
            # print("x2y2: ", x2_match.size(), y2_match.size())
            # print("mask: ", mask.size())

            feat1_match = feat1_all[:, y1[mask], x1[mask]]
            feat2_match = feat2_all[:, y2_match, x2_match]

            # print("f1m", feat1_match.size())
            # print("f2m", feat2_match.size())

            pos_dist = feat1_match.t().unsqueeze(1) @ feat2_match.t().unsqueeze(2)
            pos_dist = 2 - 2 * pos_dist.squeeze()
            # print("pos_dist: ", pos_dist.size())

            x1_match = x1[mask]
            y1_match = y1[mask]

            pos2_diff = torch.abs(x2_match.unsqueeze(1) - pos2_all[1].unsqueeze(0)).float() ** 2 + torch.abs(
                y2_match.unsqueeze(1) - pos2_all[0].unsqueeze(0)).float() ** 2
            pos2_diff = torch.sqrt(pos2_diff)

            neg_dist1 = 2 - 2 * (feat1_match.t() @ feat2_all[:, pos2_all[0], pos2_all[1]])

            # print("pos2_diff: ", pos2_diff.size(), torch.min(pos2_diff), torch.max(pos2_diff))
            # print("neg_dist1: ", neg_dist1.size())
            #
            # print(torch.max(neg_dist1 + (pos2_diff < 1) * 10,
            #                 dim=1)[0])

            neg_dist1 = torch.min(
                neg_dist1 + (pos2_diff <= 3) * 10,
                dim=1
            )[0]
            # print("neg_dist1: ", neg_dist1.size())
            # ids = (neg_dist1 + (pos2_diff < 1) * 10) > 10
            # print("ids:", ids)
            # print("mask", mask)
            # exit(0)
            pos1_diff = torch.abs(x1_match.unsqueeze(1) - pos1_all[1].unsqueeze(0)).float() ** 2 + torch.abs(
                y1_match.unsqueeze(1) - pos1_all[0].unsqueeze(0)).float() ** 2
            pos1_diff = torch.sqrt(pos1_diff)
            neg_dist2 = 2 - 2 * (feat2_match.t() @ feat1_all[:, pos1_all[0], pos1_all[1]])
            neg_dist2 = torch.min(
                neg_dist2 + (pos1_diff <= 3) * 10,
                dim=1
            )[0]

            # print("pos_dist: ", pos_dist.size(), torch.min(pos_dist), torch.max(pos_dist))
            # print("neg_dist1: ", neg_dist1.size(), torch.min(neg_dist1), torch.max(neg_dist1))
            # print("neg_dist2: ", neg_dist2.size(), torch.min(neg_dist2), torch.max(neg_dist2))
            # exit(0)
            #
            diff = F.relu(self.margin + pos_dist - torch.min(neg_dist1, neg_dist2))
            # diff = torch.clamp(self.margin + pos_dist - torch.min(neg_dist1, neg_dist2), min=0.0)
            # conf1_match = conf1_all[y1_match, x1_match]
            # conf2_match = conf2_all[y2_match, x2_match]
            # diff = diff * ((conf1_match + conf2_match) / 2.)

            # print("conf1_m: ", conf1_match.size())
            # print("conf2_m: ", conf2_match.size())

            loss = loss + diff.mean()

            # print("diff: ", diff.mean())
            n_valid_samples += 1
        if n_valid_samples == 0:
            return None
        else:
            return loss / n_valid_samples

    def forward_v3(self, descriptors, aflow, **kw):
        confs = kw.get('reliability')
        segs = kw.get("seg")
        seg_masks = kw.get("seg_mask")
        seg1 = segs[0]
        seg2 = segs[1]
        seg_mask1 = seg_masks[0]
        seg_mask2 = seg_masks[1]

        B, two, H, W = aflow.shape
        assert two == 2
        feat1, conf1 = descriptors[0], (confs[0] if confs else None)
        feat2, conf2 = descriptors[1], (confs[1] if confs else None)

        # positions in the first image
        with torch.no_grad():
            y, x = self.gen_xy(step=-1, aflow=aflow)
        # print("pos1_all: ", pos1_all.size())
        # print("pos2_all: ", pos2_all.size())

        loss = torch.tensor(np.array([0], dtype=np.float32), device=aflow.device)
        n_valid_samples = 0
        for b in range(B):
            feat1_all = feat1[b]
            feat2_all = feat2[b]
            conf1_all = conf1[b].squeeze()
            conf2_all = conf2[b].squeeze()
            seg_mask1_all = seg_mask1[b].squeeze()
            seg_mask2_all = seg_mask2[b].squeeze()
            # pos1_all = torch.stack([y1, x1])
            # pos2_all = torch.stack([y1, x1])

            # print("f1_a: ", feat1_all.size())
            # print("f2_a: ", feat2_all.size())
            # print("c1_a: ", conf1_all.size())
            # print("c2_a: ", conf2_all.size())
            # qconf1 = conf1[b, y1, x1].view(-1) if confs else None

            conf1_vec = conf1_all[y, x]
            mask = (conf1_vec >= 0.51) * (seg_mask1_all[y, x])
            y1 = y[mask]
            x1 = x[mask]
            pos1_all = torch.stack([y1, x1])
            seg1_all = seg1[b][y1, x1]

            conf2_vec = conf2_all[y, x]
            mask = (conf2_vec >= 0.51) * (seg_mask2_all[y, x])
            y2 = y[mask]
            x2 = x[mask]
            pos2_all = torch.stack([y2, x2])
            seg2_all = seg2[b][y2, x2]

            # sample GT from second image
            xy2 = (aflow[b, :, y1, x1] + 0.5).long()  # .t()
            # print("xy2: ", xy2.size())
            mask = (0 <= xy2[0]) * (0 <= xy2[1]) * (xy2[0] < W) * (xy2[1] < H)
            x2_match = xy2[0][mask]
            y2_match = xy2[1][mask]

            if x2_match.size(0) < 64 or pos1_all.shape[1] < 64 or pos2_all.shape[1] < 64:
                continue
            # print("xy2: ", x2_match.shape, pos1_all.shape, pos2_all.shape, x.shape)
            # print("x2y2: ", x2_match.size(), y2_match.size())
            # print("mask: ", mask.size())

            feat1_match = feat1_all[:, y1[mask], x1[mask]]
            feat2_match = feat2_all[:, y2_match, x2_match]
            seg1_match = seg1[b][y1[mask], x1[mask]]
            seg2_match = seg2[b][y2_match, x2_match]

            # print("f1m", feat1_match.size())
            # print("f2m", feat2_match.size())

            pos_dist = feat1_match.t().unsqueeze(1) @ feat2_match.t().unsqueeze(2)
            pos_dist = 2 - 2 * pos_dist.squeeze()
            pos_dist = torch.sqrt(pos_dist + 1e-4)
            # print("pos_dist: ", pos_dist.size())

            x1_match = x1[mask]
            y1_match = y1[mask]

            pos2_diff = torch.abs(x2_match.unsqueeze(1) - pos2_all[1].unsqueeze(0)).float() ** 2 + torch.abs(
                y2_match.unsqueeze(1) - pos2_all[0].unsqueeze(0)).float() ** 2
            pos2_diff = torch.sqrt(pos2_diff)

            neg_dist1 = 2 - 2 * (feat1_match.t() @ feat2_all[:, pos2_all[0], pos2_all[1]])
            neg_dist1 = torch.sqrt(neg_dist1 + 1e-4)

            seg2_diff = torch.abs(seg2_match.unsqueeze(1) - seg2_all.unsqueeze(0))

            # print("pos2_diff: ", pos2_diff.shape, seg2_diff.shape)
            # print("neg_dist1: ", neg_dist1.size())
            #
            # print(torch.max(neg_dist1 + (pos2_diff < 1) * 10,
            #                 dim=1)[0])

            neg_dist1 = torch.min(
                neg_dist1 + (pos2_diff <= 3) * 10 + (seg2_diff > 0) * 10,
                dim=1
            )[0]
            # print("neg_dist1: ", neg_dist1.size())
            # ids = (neg_dist1 + (pos2_diff < 1) * 10) > 10
            # print("ids:", ids)
            # print("mask", mask)
            # exit(0)
            pos1_diff = torch.abs(x1_match.unsqueeze(1) - pos1_all[1].unsqueeze(0)).float() ** 2 + torch.abs(
                y1_match.unsqueeze(1) - pos1_all[0].unsqueeze(0)).float() ** 2
            pos1_diff = torch.sqrt(pos1_diff)

            seg1_diff = torch.abs(seg1_match.unsqueeze(1) - seg1_all.unsqueeze(0))

            # print("pos1_diff: ", pos1_diff.shape, seg1_diff.shape)

            neg_dist2 = 2 - 2 * (feat2_match.t() @ feat1_all[:, pos1_all[0], pos1_all[1]])
            neg_dist2 = torch.sqrt(neg_dist2 + 1e-4)

            neg_dist2 = torch.min(
                neg_dist2 + (pos1_diff <= 3) * 10 + (seg1_diff > 0) * 10,
                dim=1
            )[0]

            # print("pos_dist: ", pos_dist.size(), torch.min(pos_dist), torch.max(pos_dist))
            # print("neg_dist1: ", neg_dist1.size(), torch.min(neg_dist1), torch.max(neg_dist1))
            # print("neg_dist2: ", neg_dist2.size(), torch.min(neg_dist2), torch.max(neg_dist2))
            # exit(0)
            #
            diff = self.margin + pos_dist - torch.min(neg_dist1, neg_dist2)
            # diff = torch.clamp(self.margin + pos_dist - torch.min(neg_dist1, neg_dist2), min=0.0)
            conf1_match = conf1_all[y1_match, x1_match]
            conf2_match = conf2_all[y2_match, x2_match]
            conf12 = (conf1_match + conf2_match) / 2.

            # print("conf1_m: ", conf1_match.size())
            # print("conf2_m: ", conf2_match.size())

            loss = loss + (diff * conf12)[diff > 0].mean()

            # print("diff: ", diff.mean())
            n_valid_samples += 1
        if n_valid_samples == 0:
            return None
        else:
            return loss / n_valid_samples


class TripletLossV2(nn.Module):
    def __init__(self, step=1, margin=1., border=0, scaling_step=2, safe_radius=5, use_score=False):
        super(TripletLossV2, self).__init__()
        self.step = step
        self.margin = margin
        self.border = border
        self.scaling_step = scaling_step
        self.safe_radius = safe_radius
        self.name = 'tripletv2'
        self.use_score = use_score

    def gen_grid(self, step, B, H, W, dev):
        # print("feat: ", feat.shape)
        # B, C, H, W = feat.shape
        # dev = feat.device
        b1 = torch.arange(B, device=dev)
        if step > 0:
            # regular grid
            x1 = torch.arange(self.border, W - self.border, step, device=dev)
            y1 = torch.arange(self.border, H - self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            x1 = x1[None, None, :].expand(B, H1, W1).reshape(-1)
            y1 = y1[None, :, None].expand(B, H1, W1).reshape(-1)
            b1 = b1[:, None, None].expand(B, H1, W1).reshape(-1)
            shape = (B, H1, W1)
        else:
            # randomly spread
            n = (H - 2 * self.border) * (W - 2 * self.border) // step ** 2
            x1 = torch.randint(self.border, W - self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H - self.border, (n,), device=dev)
            x1 = x1[None, :].expand(B, n).reshape(-1)
            y1 = y1[None, :].expand(B, n).reshape(-1)
            b1 = b1[:, None].expand(B, n).reshape(-1)
            shape = (B, n)
        return b1, y1, x1, shape

    def gen_xy(self, step, feat):
        B, C, H, W = feat.shape
        dev = feat.device
        if step > 0:
            x1 = torch.arange(self.border, W - self.border, step, device=dev)
            y1 = torch.arange(self.border, H - self.border, step, device=dev)
            H1, W1 = len(y1), len(x1)
            x1 = x1[None, :].expand(H1, W1).reshape(-1)
            y1 = y1[:, None].expand(H1, W1).reshape(-1)
        else:
            n = (H - 2 * self.border) * (W - 2 * self.border) // step ** 2
            x1 = torch.randint(self.border, W - self.border, (n,), device=dev)
            y1 = torch.randint(self.border, H - self.border, (n,), device=dev)

        return y1, x1

    def forward(self, descriptors, aflow, **kw):
        return self.forward_v1(descriptors, aflow, **kw)

    def forward_v1(self, descriptors, aflow, **kw):
        confs = kw.get('reliability')
        B, two, H, W = aflow.shape  # full resolution
        assert two == 2
        feat1, conf1 = descriptors[0], (confs[0] if confs else None)
        feat2, conf2 = descriptors[1], (confs[1] if confs else None)
        # print("conf: ", conf1.shape, conf2.shape)

        # positions in the first image
        with torch.no_grad():
            y1, x1 = self.gen_xy(step=self.step, feat=feat1)
            # y1, x1 = self.gen_xy(step=self.step, feat=feat1)
        pos1_all = torch.stack([y1, x1])
        pos2_all = torch.stack([y1, x1])
        # print("pos1_all: ", torch.min(x1), torch.max(x1), feat1.shape)
        # print("pos2_all: ", torch.min(y1), torch.max(y1), feat2.shape)

        loss = torch.tensor(np.array([0], dtype=np.float32), device=aflow.device)
        # loss = 0
        n_valid_batch = 0
        for b in range(B):
            B, C, H, W = feat1.shape
            feat1_all = feat1[b]
            feat2_all = feat2[b]
            # feat2_all = feat2[b]
            conf1_all = conf1[b].squeeze()
            conf2_all = conf2[b].squeeze()
            # print("f1_a: ", feat1_all.size())
            # print("f2_a: ", feat2_all.size())
            # print("c1_a: ", conf1_all.size())
            # print("c2_a: ", conf2_all.size())
            # qconf1 = conf1[b, y1, x1].view(-1) if confs else None

            # sample GT from second image

            if aflow.shape[2] != H or aflow.shape[3] != W:
                # sy = aflow.shape[2] // H
                # sx = aflow.shape[3] // W
                # upy = (y1 * 4 + 1).long()
                # upx = (x1 * 4 + 1).long()
                # xy2 = aflow[b, :, upy, upx] + 0.5
                # print("xy2---:", xy2.shape)
                # x2 = (xy2[0] / 4).long()
                # y2 = (xy2[1] / 4).long()

                upy = self.upscale_positions(pos=y1, scaling_steps=self.scaling_step).long()
                upx = self.upscale_positions(pos=x1, scaling_steps=self.scaling_step).long()
                xy2 = aflow[b, :, upy, upx] + 0.5
                x2 = self.downscale_positions(pos=xy2[0], scaling_steps=self.scaling_step).long()
                y2 = self.downscale_positions(pos=xy2[1], scaling_steps=self.scaling_step).long()

            else:
                xy2 = (aflow[b, :, y1, x1] + 0.5).long()  # [2, N]
                # print('xy2: ', xy2.shape)
                # print('tri xy2: ', xy2.shape, y1.shape, x1.shape)
                x2 = xy2[0]
                y2 = xy2[1]

            mask = (0 <= x2) * (0 <= y2) * (x2 < W) * (y2 < H)
            x2_match = x2[mask]
            y2_match = y2[mask]

            # print('x2_match: ', x2_match.shape)

            if x2_match.size(0) < 64:
                continue
            # print("xy2: ", xy2.size())
            # print("x2y2: ", x2_match.size(), y2_match.size())
            # print("mask: ", mask.size())

            feat1_match = feat1_all[:, y1[mask], x1[mask]]
            feat2_match = feat2_all[:, y2_match, x2_match]

            # print("f1m", feat1_match.size())
            # print("f2m", feat2_match.size())

            pos_dist = feat1_match.t().unsqueeze(1) @ feat2_match.t().unsqueeze(2)
            pos_dist = 2 - 2 * pos_dist.squeeze().float()
            # pos_dist = torch.sqrt(pos_dist + 1e-4)
            # print("pos_dist: ", pos_dist.size())

            x1_match = x1[mask]
            y1_match = y1[mask]

            pos2_diff = torch.abs(x2_match.unsqueeze(1) - pos2_all[1].unsqueeze(0)) ** 2 + torch.abs(
                y2_match.unsqueeze(1) - pos2_all[0].unsqueeze(0)) ** 2
            pos2_diff = torch.sqrt(pos2_diff.float() + 1e-4)

            neg_dist1 = 2 - 2 * (feat1_match.t() @ feat2_all[:, pos2_all[0], pos2_all[1]])
            # neg_dist1 = torch.sqrt(neg_dist1 + 1e-4)

            # print("pos2_diff: ", pos2_diff.size(), torch.min(pos2_diff), torch.max(pos2_diff))
            # print("neg_dist1: ", neg_dist1.size())
            #
            # print(torch.max(neg_dist1 + (pos2_diff < 1) * 10,
            #                 dim=1)[0])

            neg_dist1 = torch.min(
                neg_dist1 + (pos2_diff <= self.safe_radius) * 10,
                dim=1
            )[0]
            # print("neg_dist1: ", neg_dist1.size())
            # ids = (neg_dist1 + (pos2_diff < 1) * 10) > 10
            # print("ids:", ids)
            # print("mask", mask)
            # exit(0)
            pos1_diff = torch.abs(x1_match.unsqueeze(1) - pos1_all[1].unsqueeze(0)) ** 2 + torch.abs(
                y1_match.unsqueeze(1) - pos1_all[0].unsqueeze(0)) ** 2
            pos1_diff = torch.sqrt(pos1_diff.float())
            neg_dist2 = 2 - 2 * (feat2_match.t() @ feat1_all[:, pos1_all[0], pos1_all[1]])
            # neg_dist2 = torch.sqrt(neg_dist2 + 1e-4)
            neg_dist2 = torch.min(
                neg_dist2 + (pos1_diff <= self.safe_radius) * 10,
                dim=1
            )[0]

            # print("pos_dist: ", pos_dist.size(), torch.min(pos_dist), torch.max(pos_dist))
            # print("neg_dist1: ", neg_dist1.size(), torch.min(neg_dist1), torch.max(neg_dist1))
            # print("neg_dist2: ", neg_dist2.size(), torch.min(neg_dist2), torch.max(neg_dist2))
            # exit(0)
            #
            diff = F.relu(self.margin + pos_dist - torch.min(neg_dist1, neg_dist2))
            # diff = F.relu(pos_dist - 0.2) + F.relu(self.margin - torch.min(neg_dist1, neg_dist2))

            if self.use_score:
                y1_match_up = self.upscale_positions(pos=y1_match, scaling_steps=self.scaling_step).long()
                x1_match_up = self.upscale_positions(pos=x1_match, scaling_steps=self.scaling_step).long()
                y2_match_up = self.upscale_positions(pos=y2_match, scaling_steps=self.scaling_step).long()
                x2_match_up = self.upscale_positions(pos=x2_match, scaling_steps=self.scaling_step).long()
                conf1_match = conf1_all[y1_match_up, x1_match_up]
                conf2_match = conf2_all[y2_match_up, x2_match_up]

                diff_mean = torch.sum(conf1_match * conf2_match * diff) / torch.sum(conf1_match * conf2_match)
            else:
                diff_mean = diff.mean()

            loss = loss + diff_mean

            # print("diff: ", diff.mean())

            # loss = loss + diff.mean()

            # print("diff: ", diff.mean())
            n_valid_batch += 1
        if n_valid_batch == 0:
            return None
        else:
            return loss / n_valid_batch

    def upscale_positions(self, pos, scaling_steps=0):
        for _ in range(scaling_steps):
            pos = pos * 2 + 0.5
        return pos

    def downscale_positions(self, pos, scaling_steps=0):
        for _ in range(scaling_steps):
            pos = (pos - 0.5) / 2
        return pos
