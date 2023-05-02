# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   sfd2 -> trainer
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   09/03/2023 16:11
=================================================='''
import torch
import torch.optim as optim
from tqdm import tqdm
import numpy as np
from tensorboardX import SummaryWriter
import os
import os.path as osp
from datetime import datetime

from nets.superpoint import SuperPointNet
from nets.semseg.segnet import SegNet
from nets.convnext import ConvNeXt
from tools.common import save_args
from nets.semseg.utils import get_semantic_dict, get_conf_dict, segmantic_to_confidence


class Trainer:
    def __init__(self, net, loader, loss, args=None):
        self.model = net
        self.loader = loader
        self.loss_func = loss
        self.optimizer = optim.Adam([p for p in net.parameters() if p.requires_grad],
                                    lr=args.lr, weight_decay=args.weight_decay)
        self.do_val = args.do_eval > 0

        self.spp = SuperPointNet()
        self.spp.load_state_dict(torch.load("weights/superpoint_v1.pth"))
        self.spp = self.spp.eval().cuda()

        self.args = args
        self.score_th = self.args.score_th

        self.seg_feat = self.args.seg_feat > 0
        if self.seg_feat:
            self.seg_encoder = ConvNeXt(arch='base').cuda().eval()
            self.seg_encoder.load_state_dict(torch.load("weights/convxts-base_ade20k.pth"))
        else:
            self.seg_encoder = None

        self.use_seg = (self.args.seg_det > 0) or (self.args.seg_desc > 0)
        if self.use_seg:
            self.conf_dict = get_conf_dict()
            self.semantic_dict = get_semantic_dict()
            self.seg = SegNet(model_name="convxts-base-ade20k", device="cuda:0")
            self.grid = np.mgrid[:self.args.R, :self.args.R][::-1]
            # self.grid = torch.from_numpy(self.grid).int().cuda()
            self.hs = self.grid[0, :, :].reshape(-1)
            self.ws = self.grid[1, :, :].reshape(-1)

            self.hs = torch.from_numpy(self.hs).cuda().long().requires_grad_(False)
            self.ws = torch.from_numpy(self.ws).cuda().long().requires_grad_(False)

            self.init_warp = torch.zeros(size=(self.args.bs, self.args.R, self.args.R, 1),
                                         dtype=torch.float).requires_grad_(False)
        else:
            self.seg = None

        self.num_epochs = self.args.epochs

        if self.args.resume == 'None':
            self.start_epoch = 0
            self.epoch = 0
            self.iteration = 0
            self.init_lr = self.args.lr
            now = datetime.datetime.now()
            tag = now.strftime("%Y_%m_%d_%H_%M_%S")
            tag += ('_' + args.net + ('_' + args.loss) + '_B' + str(args.bs) + '_D' + str(
                args.dim) + "_R" + str(args.R) + '_' + args.det_loss)
            if self.args.seg_det > 0:
                tag += '_sd'
            if self.args.seg_desc > 0:
                tag += ('_' + self.args.seg_desc_loss_fn)
            if self.args.seg_feat > 0:
                tag += '_sf'
            if args.use_score > 0:
                tag += '_us'
                if args.use_pred_score_desc > 0:
                    tag += 'p'
                else:
                    tag += 'g'

            save_dir = osp.join(self.args.root, tag)
            if not os.path.exists(save_dir):
                os.mkdir(save_dir)
                save_args(args=args, save_path=osp.join(save_dir, 'args.txt'))
            self.save_dir = save_dir
            self.tag = tag
            self.log_file = open(os.path.join(self.save_dir, "log.txt"), 'a+')
            self.writer = SummaryWriter(self.save_dir)
        else:
            data = torch.load(osp.join(self.args.root, self.args.resume))
            self.model.load_state_dict(data["model"])
            self.start_epoch = data["epoch"] + 1
            self.iteration = data['iteration']

            print("Resume from {:s}".format(self.args.resume))
            self.tag = str(self.args.resume).split('/')[0]

            self.save_dir = osp.join(self.args.root, str(self.args.resume).split('/')[0])
            self.log_file = open(os.path.join(self.save_dir, "log.txt"), 'a+')
            self.writer = SummaryWriter(self.save_dir)

    def iscuda(self):
        return next(self.model.parameters()).device != torch.device('cpu')

    def todevice(self, x):
        if isinstance(x, dict):
            return {k: self.todevice(v) for k, v in x.items()}
        if isinstance(x, (tuple, list)):
            return [self.todevice(v) for v in x]

        if self.iscuda():
            return x.contiguous().cuda(non_blocking=True)
        else:
            return x.cpu()

    def process_epoch(self, train=True):
        epoch_losses = []
        epoch_det_losses = []
        epoch_unsup_desc_losses = []
        epoch_seg_det_losses = []
        epoch_seg_desc_losses = []
        epoch_seg_feat_losses = []

        self.model.train()

        for batch_idx, inputs in enumerate(tqdm(self.loader)):
            if self.args.iterations_per_epoch > 0:
                if batch_idx >= self.args.iterations_per_epoch:
                    break
            inputs = self.todevice(inputs)
            # compute gradient and do model update
            if train:
                self.iteration = self.iteration + 1
                self.optimizer.zero_grad()

            loss, loss_items = self.forward_backward(inputs)
            # print(loss_items)
            if len(self.args.gpu) > 1:
                for i in range(loss.shape[0]):
                    if loss[i] is None:
                        print("loss is None.")
                        continue
                    if torch.isnan(loss[i]):
                        print('Loss is NaN')
                        continue

                loss = torch.mean(loss)
                for v in loss_items.keys():
                    loss_items[v] = torch.mean(loss_items[v])
            else:
                if loss is None:
                    print("loss is None.")
                    continue
                if torch.isnan(loss):
                    print('Loss is NaN')
                    continue

            lr = min(self.args.lr * self.args.decay_rate ** (self.iteration - self.args.decay_iter), self.args.lr)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr

            if train:
                loss.backward()
                self.optimizer.step()

            det_loss = torch.zeros_like(loss)
            seg_det_loss = torch.zeros_like(loss)
            seg_desc_loss = torch.zeros_like(loss)
            seg_feat_loss = torch.zeros_like(loss)
            unsup_desc_loss = torch.zeros_like(loss)
            if "det_loss" in loss_items.keys():
                det_loss = loss_items["det_loss"]
            if "unsup_desc_loss" in loss_items.keys():
                unsup_desc_loss = loss_items["unsup_desc_loss"]
            if "seg_det_loss" in loss_items.keys():
                seg_det_loss = loss_items["seg_det_loss"]
            if "seg_desc_loss" in loss_items.keys():
                seg_desc_loss = loss_items["seg_desc_loss"]
            if "seg_feat_loss" in loss_items.keys():
                seg_feat_loss = loss_items["seg_feat_loss"]

            current_loss = loss.item()
            epoch_losses.append(current_loss)
            epoch_det_losses.append(det_loss.item())
            epoch_unsup_desc_losses.append(unsup_desc_loss.item())
            epoch_seg_det_losses.append(seg_det_loss.item())
            epoch_seg_desc_losses.append(seg_desc_loss.item())
            epoch_seg_feat_losses.append(seg_feat_loss.item())

            # logging
            if batch_idx % self.args.log_interval == 0 and self.args.local_rank == 0:
                self.log_file.write(
                    '[{:s}] epoch {:d}-batch {:d}/{:d}| avg:{:.3f}, det:{:.3f}, undesc:{:.3f}, segdet:{:.3f}, segdesc:{:.3f}, segfeat:{:.3f}\n'.format(
                        'train' if train else 'val',
                        self.epoch, batch_idx, len(self.loader),
                        np.mean(epoch_losses),
                        np.mean(epoch_det_losses),
                        np.mean(epoch_unsup_desc_losses),
                        np.mean(epoch_seg_det_losses),
                        np.mean(epoch_seg_desc_losses),
                        np.mean(epoch_seg_feat_losses),
                    ))

                print(
                    '[{:s}] epoch {:d} batch {:d}/{:d}| loss:{:.3f}, det:{:.3f}, undesc:{:.3f}, segdet:{:.3f}, segdesc:{:.3f}, segfeat:{:.3f}'.format(
                        'train' if train else 'valid',
                        self.epoch, batch_idx, len(self.loader), loss.item(), det_loss.item(), unsup_desc_loss.item(),
                        seg_det_loss.item(), seg_desc_loss.item(), seg_feat_loss.item()))

                if train:
                    infos = {
                        "loss": loss.item(),
                        "det_loss": det_loss.item(),
                        "undesc_loss": unsup_desc_loss.item(),
                        "seg_det_loss": seg_det_loss.item(),
                        "seg_desc_loss": seg_desc_loss.item(),
                        "seg_feat_loss": seg_feat_loss.item(),
                        "lr": self.optimizer.param_groups[0]['lr'],
                    }

                for tag, value in infos.items():
                    self.writer.add_scalar(tag=tag, scalar_value=value, global_step=self.iteration + 1)
                # self.logger.scalar_summary(tag=tag, value=value, step=self.iteration + 1)

        print(
            "[{:s}] epoch {:d} avg:{:.3f}, det:{:.3f}, undesc:{:.3f}, segdet:{:.3f}, segdesc:{:.3f}, segfeat:{:.3f}".format(
                'train' if train else 'valid',
                self.epoch,
                np.mean(epoch_losses),
                np.mean(epoch_det_losses),
                np.mean(epoch_unsup_desc_losses),
                np.mean(epoch_seg_det_losses),
                np.mean(epoch_seg_desc_losses),
                np.mean(epoch_seg_feat_losses),
            ))
        self.log_file.write(
            "[{:s}] epoch {:d} avg loss:{:.3f}, det:{:.3f}, undesc:{:.3f}, segdet:{:.3f}, segdesc:{:.3f}, segfeat:{:.3f}\n\n".format(
                'train' if train else 'valid',
                self.epoch,
                np.mean(epoch_losses),
                np.mean(epoch_det_losses),
                np.mean(epoch_unsup_desc_losses),
                np.mean(epoch_seg_det_losses),
                np.mean(epoch_seg_desc_losses),
                np.mean(epoch_seg_feat_losses),
            ))
        self.log_file.flush()
        return np.mean(epoch_losses)

    def forward_backward(self, inputs):
        image1 = inputs.pop("img1")  # torch.mean(inputs.pop("img1"), dim=1, keepdim=True)
        image2 = inputs.pop("img2")  # torch.mean(inputs.pop("img2"), dim=1, keepdim=True)
        aflow = inputs["aflow"]

        raw_image1 = inputs.pop("raw_img1")
        raw_image2 = inputs.pop("raw_img2")
        gray_image1 = inputs.pop("gray_img1")
        gray_image2 = inputs.pop("gray_img2")

        batch = {
            "image1": image1,
            "image2": image2,
        }
        output = self.model(batch)

        allvars = dict(inputs, **output)
        allvars["score_th"] = self.args.score_th

        if self.seg_encoder is not None:
            with torch.no_grad():
                gt_feats = self.seg_encoder.extract(torch.cat([image1, image2], dim=0))
                allvars["gt_feats"] = gt_feats
        if self.seg is not None:
            with torch.no_grad():
                seg1 = []
                bs = raw_image1.shape[0]
                for b in range(bs):
                    img_numpy = raw_image1[b, :, :, :].cpu().numpy()
                    seg_result = self.seg.evaluate(img_numpy)
                    seg1.append(torch.Tensor(seg_result).cuda())

                seg1 = torch.cat(seg1, dim=0) + 1
                seg2 = torch.zeros_like(seg1)
                mask2 = []
                for b in range(aflow.size(0)):
                    # print("batch: ", b)
                    nhs = aflow[b, 0].view(-1) + 0.5
                    nws = aflow[b, 1].view(-1) + 0.5
                    # mask = amask[b].view(-1)
                    valid_ids = (nhs < self.args.R) & (nws < self.args.R) \
                                & (nhs >= 0) & (nws >= 0) \
                                & (~torch.isnan(nhs)) & (~torch.isnan(nws))
                    mask2.append(valid_ids.view(self.args.R, self.args.R))
                    invlalid_ids = ~valid_ids
                    nhs[invlalid_ids] = 0
                    nws[invlalid_ids] = 0
                    seg2[b, nws.long(), nhs.long()] = seg1[b, self.ws, self.hs]
                seg_conf1 = segmantic_to_confidence(seg_map=seg1, conf_dict=self.conf_dict,
                                                    semantic_dict=self.semantic_dict)
                seg_conf2 = segmantic_to_confidence(seg_map=seg2, conf_dict=self.conf_dict,
                                                    semantic_dict=self.semantic_dict)
                mask2 = torch.stack(mask2).requires_grad_(False).bool().requires_grad_(False)

                mask1 = torch.ones_like(mask2).requires_grad_(False)

                allvars["seg_confidence"] = torch.cat([seg_conf1, seg_conf2], dim=0)
                allvars["seg_mask"] = torch.cat([mask1, mask2], dim=0)
                allvars["seg"] = torch.cat([seg1, seg2], dim=0)

                del raw_image1
                del raw_image2

        with torch.no_grad():
            # spp_score, spp_desc, spp_semi = self.spp(torch.cat([gray_image1, gray_image2], dim=0))
            spp_out = self.spp(torch.cat([gray_image1, gray_image2], dim=0))
            spp_score = spp_out['scores']
            spp_desc = spp_out['descs']
            spp_semi = spp_out['semi']
            spp_semi_norm = spp_out['semi_norm']
            del gray_image1
            del gray_image2
            del spp_desc

            if len(spp_score.shape) == 3:
                spp_score = spp_score.unsqueeze(1)

            weight = torch.ones_like(spp_score)
            weight[spp_score >= self.score_th] = self.args.det_weight
            # print("weight: ", torch.sum(weight1[0]) / 1000.)

            allvars["gt_score"] = spp_score  # torch.cat([spp_score1, spp_score2], dim=0)
            allvars["gt_semi"] = spp_semi  # torch.cat([spp_score1, spp_score2], dim=0)
            allvars["gt_semi_norm"] = spp_semi_norm  # torch.cat([spp_score1, spp_score2], dim=0)
            # allvars["gt_desc"] = spp_desc  # torch.cat([spp_desc1, spp_desc2], dim=0)
            allvars["weight"] = weight  # torch.cat([weight1, weight2], dim=0)

        loss_items = self.loss_func(allvars)

        return loss_items["loss"], loss_items

    def train(self, resume=None):
        min_loss = 1e10
        train_loss_history = []
        for epoch in range(self.start_epoch, self.num_epochs):
            self.epoch = epoch
            if self.args.with_dist > 0:
                self.loader.sampler.set_epoch(epoch=epoch)

            if self.args.local_rank == 0:
                train_loss = self.process_epoch(train=True)
                train_loss_history.append(train_loss)

                if len(self.args.gpu) > 1 and self.args.with_dist == 0:
                    state_dict = self.model.module.state_dict()
                else:
                    state_dict = self.model.state_dict()

                checkpoint = {
                    'args': self.args,
                    'epoch': self.epoch,
                    'model': state_dict,
                    'iteration': self.iteration,
                }

                if train_loss_history[-1] < min_loss:
                    min_loss = train_loss_history[-1]
                    best_checkpoint_path = os.path.join(
                        self.save_dir,
                        '%s.best.pth' % (self.tag)
                    )
                    torch.save(checkpoint, best_checkpoint_path)

                # if epoch % 5 == 0:
                torch.save(checkpoint, osp.join(self.save_dir, 'epoch_{:d}.pth'.format(epoch)))

                if self.do_val:
                    eval_out = self.eval_on_data()
                    text = "Eval - Epoch {:d}".format(self.epoch)
                    for k in eval_out.keys():
                        text += ('_{:s}:{:.2f}'.format(k, eval_out[k]))

                    print(text)
                    self.log_file.write(text + "\n")
                    self.log_file.flush()
        # self.log_file.close()
        if self.args.local_rank == 0:
            self.log_file.close()
