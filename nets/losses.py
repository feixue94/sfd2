from nets.reliability_loss import *


class SegLoss(nn.Module):
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
        super().__init__()
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
        scores = input["gt_score"]
        descs = input["desc"]  # default [B, D, H, W]
        descs = descs.permute([0, 2, 3, 1])
        masks = input["seg_mask"]
        segs = input["seg"]
        seg_confidences = input["seg_confidence"]

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

        sel_score1 = sel_score1.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score1 = sel_score1.clamp(min=0.0005, max=1.)
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.) * 2. + 0.5
        sel_score2 = sel_score2.clamp(min=0.0005, max=1.)

        # apply semantic confidence
        sel_score1 = sel_score1 * seg_confidences[0:b][ids1]
        sel_score2 = sel_score2 * seg_confidences[b:][ids2]

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
        # dist_map12 = torch.sqrt(dist_map12 + 1e-4)
        seg_const_map12 = torch.abs(sel_seg1.unsqueeze(1) - sel_seg2.unsqueeze(0))  # M x N
        score_map12 = sel_score1.unsqueeze(1) * sel_score2.unsqueeze(0)

        pos_ids12 = (seg_const_map12 == 0)  # with the same label
        neg_ids12 = (seg_const_map12 > 0)  # with different labels

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

            dist12 = pos_dist12 + neg_dist12

            return dist12

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

        # conf_th = torch.median(scores)
        values, _ = torch.topk(scores[masks].reshape((1, -1)), k=1000 * b * 2, largest=True, dim=1)
        conf_th = values[0, -1]
        # b = scores.size(0) // 2

        ids1 = (scores[0:b] > conf_th) & masks[0:b]
        ids2 = (scores[b:] > conf_th) & masks[b:]

        sel_desc1 = descs[0:b][ids1, :]
        sel_desc2 = descs[b:][ids2, :]
        sel_seg1 = segs[0:b][ids1]
        sel_seg2 = segs[b:][ids2]

        sel_score1 = scores[0:b][ids1]
        sel_score2 = scores[b:][ids2]

        dist_map12 = sel_desc1 @ sel_desc2.t()  # M x N
        dist_map12 = 2 - 2 * dist_map12
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
        elif self.detloss in ['cel']:
            loss = torch.nn.functional.cross_entropy(pred_score, gt_score, reduce=False)
            if weight is not None:
                loss = loss * weight
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
        elif self.detloss in ['cel']:
            det_loss = self.det_loss(pred_score=output["logits"], gt_score=output["gt_semi"], weight=None,
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
