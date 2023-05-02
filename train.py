# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   sfd2 -> train
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   09/03/2023 16:11
=================================================='''
import torch
import torch.nn as nn
import os.path as osp
from tools.dataloader import *
from nets.losses import SegLoss
from nets.sfd2 import ResSegNetV2
from nets.sampler import NghSampler2DS
from nets.reliability_loss import ReliabilityLoss
from trainer import Trainer
import warnings
import torch.multiprocessing as mp
import torch.distributed as dist
from datasets import *

warnings.filterwarnings("ignore")

toy_db_debug = """SyntheticPairDataset(
    ImgFolder('imgs'), 
            'RandomScale(`R`,1024,can_upscale=True)', 
            'RandomTilting(0.5), PixelNoise(25)')"""

db_web_images = """SyntheticPairDataset(
    web_images, 
        'RandomScale(`R`,1024,can_upscale=True)',
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_images = """SyntheticPairDataset(
    aachen_db_images, 
        'RandomScale(`R`,1024,can_upscale=True)', 
        'RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_style_transfer = """TransformedPairs(
    aachen_style_transfer_pairs,
            'RandomScale(`R`,1024,can_upscale=True), RandomTilting(0.5), PixelNoise(25)')"""

db_aachen_flow = "aachen_flow_pairs"

data_sources = dict(
    D=toy_db_debug,
    W=db_web_images,
    A=db_aachen_images,
    F=db_aachen_flow,
    S=db_aachen_style_transfer,
)

default_dataloader = """PairLoader(CatPairDataset(`data`),
    scale   = 'RandomScale(`R`,512,can_upscale=True)',
    distort = 'ColorJitter(0.2,0.2,0.2,0.1)',
    crop    = 'RandomCrop(`R`)')"""

default_sampler = """NghSampler2(ngh=7, subq=-8, subd=1, pos_d=3, neg_d=5, border=16,
                            subd_neg=-8,maxpool_pos=True)"""

default_loss = """MultiLoss(
        1, ReliabilityLoss(`sampler`, base=0.5, nq=20),
        1, CosimLoss(N=`N`),
        1, PeakyLoss(N=`N`))"""


def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12357'
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def train_DDP(rank, world_size, model, args):
    print('Using distributed parallel training...')
    upsample_desc = False
    sampler = NghSampler2DS(ngh=7, subq=-4, subd=1, pos_d=3, neg_d=5, border=8,
                            subd_neg=-4, maxpool_pos=True,
                            scaling_step=args.scaling_step)  # default subq=-4, subd_neg=-4
    reliability_loss = ReliabilityLoss(sampler=sampler, base=0.5, nq=20).cuda()

    loss = SegLoss(desc_loss_fn=reliability_loss,
                   weights={
                       "det_loss": args.wdet,
                       "desc_loss": args.wdesc,
                       "seg_det_loss": args.wsdet,
                       "seg_desc_loss": args.wsdesc,
                       "seg_feat_loss": args.wsfeat,
                   },
                   use_pred_score_desc=args.use_pred_score_desc > 0,
                   det_loss=args.det_loss,
                   seg_desc_loss_fn=args.seg_desc_loss_fn,
                   upsample_desc=upsample_desc,
                   seg_desc=args.seg_desc > 0,
                   seg_feat=args.seg_feat > 0,
                   seg_det=args.seg_det > 0,
                   seg_cls=args.seg_cls > 0,
                   )

    db = [data_sources[key] for key in args.train_data]
    train_set = eval(args.data_loader.replace('`data`', ','.join(db)).replace('`R`', str(args.R)).replace('\n', ''))
    print("Training image database =", train_set)

    device = torch.device(f'cuda:{rank}')
    model.to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    setup(rank=rank, world_size=world_size)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_set, shuffle=False, drop_last=True)
    train_loader = torch.utils.data.DataLoader(train_set,
                                               batch_size=args.bs // world_size,
                                               num_workers=args.threads // world_size,
                                               pin_memory=False,
                                               sampler=train_sampler,
                                               collate_fn=collate,
                                               )
    print('train loader: ', len(train_loader))
    args.local_rank = rank
    # trainer = Trainer(model=model, train_loader=train_loader, eval_loader=None, args=args)
    trainer = Trainer(net=model, args=args, loader=train_loader, loss=loss)
    trainer.train()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser("Train R2D2")

    parser.add_argument("--data-loader", type=str, default=default_dataloader)
    parser.add_argument("--train-data", type=str, default=list('WASF'), nargs='+',
                        choices=set(data_sources.keys()))
    parser.add_argument("--net", type=str, help='network architecture')
    parser.add_argument("--root", type=str, default='/home/mifs/fx221/fx221/exp/sfd2')
    parser.add_argument("--tag", type=str, default='')
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--iterations_per_epoch", type=int, default=-1, help='number of iteraions per epoch')
    parser.add_argument("--pretrained_weight", type=str, default=None, help='pretrained model path')

    parser.add_argument("--resume", type=str, default=None, help="checkpoint for resume")
    parser.add_argument("--loss", type=str, default=default_loss, help="loss function")
    parser.add_argument("--sampler", type=str, default=default_sampler, help="AP sampler")
    parser.add_argument("--R", type=int, default=192, help="image resolution")
    parser.add_argument("--N", type=int, default=16, help="patch size for repeatability")

    parser.add_argument("--dim", type=int, default=128, help='dim of descriptors')
    parser.add_argument("--epochs", type=int, default=80, help='number of training epochs')
    parser.add_argument("--bs", "--bs", type=int, default=6, help="batch size")
    parser.add_argument("--lr", "--lr", type=str, default=1e-4)
    parser.add_argument("--weight-decay", "--wd", type=float, default=5e-4)

    parser.add_argument("--threads", type=int, default=4, help='number of worker threads')
    parser.add_argument("--gpu", type=int, nargs='+', default=[0], help='-1 for CPU')
    parser.add_argument("--upsampling", action="store_true", default=False)
    parser.add_argument("--do_eval", action="store_true", default=False)
    parser.add_argument('--wdet', type=float, default=1.0, help='weight of det loss)')
    parser.add_argument('--wdesc', type=float, default=1.0, help='weight of desc loss)')
    parser.add_argument('--wsdesc', type=float, default=0.5, help='weight of seg desc loss)')
    parser.add_argument('--wsfeat', type=float, default=1.0, help='weight of seg feat loss)')
    parser.add_argument('--wsdet', type=float, default=1.0, help='weight of seg feat loss)')
    parser.add_argument('--seg_det', type=int, default=0, help='seg det loss')
    parser.add_argument('--seg_desc', type=int, default=0, help='seg desc loss')
    parser.add_argument('--seg_feat', type=int, default=0, help='seg feat loss')
    parser.add_argument('--seg_desc_loss_fn', type=str, default='wap', help='seg desc loss fn')

    parser.add_argument("--score_th", type=float, default=0.001, help='score threshold for using superpoint detector')
    parser.add_argument("--det_weight", type=float, default=1.0, help='weight for selected keypoints')
    parser.add_argument("--log_interval", type=int, default=50, help='weight for selected keypoints')
    parser.add_argument('--eval_root', type=str, default="/data/cornucopia/fx221/exp/swd2/test_images/trans")
    parser.add_argument('--eval_ref_fn', type=str, default="img.jpg")
    parser.add_argument('--eval_query_list', type=str, default="evaluate_image_list.txt")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--with_dist", type=int, default=0)

    args = parser.parse_args()
    with open(args.config, 'rt') as f:
        t_args = argparse.Namespace()
        t_args.__dict__.update(json.load(f))
        args = parser.parse_args(namespace=t_args)

    iscuda = len(args.gpu) > 0
    net = ResSegNetV2(outdim=args.dim, require_feature=args.seg_feat > 0, require_stability=args.seg_det > 0)

    if osp.isfile(args.resume):
        print("Load pretrained weight from {:s}".format(osp.join(args.root, args.resume)))
        net.load_state_dict(torch.load(osp.join(args.root, args.resume))["state_dict"], strict=True)

    if args.with_dist > 0:
        mp.spawn(train_DDP, nprocs=len(args.gpu), args=(len(args.gpu), net, args), join=True)
    else:
        upsample_desc = False
        sampler = NghSampler2DS(ngh=7, subq=-4, subd=1, pos_d=3, neg_d=5, border=8,
                                subd_neg=-4, maxpool_pos=True,
                                scaling_step=args.scaling_step)  # default subq=-4, subd_neg=-4
        reliability_loss = ReliabilityLoss(sampler=sampler, base=0.5, nq=20).cuda()

        loss = SegLoss(desc_loss_fn=reliability_loss,
                       weights={
                           "det_loss": args.wdet,
                           "desc_loss": args.wdesc,
                           "seg_det_loss": args.wsdet,
                           "seg_desc_loss": args.wsdesc,
                           "seg_feat_loss": args.wsfeat,
                       },
                       use_pred_score_desc=args.use_pred_score_desc > 0,
                       det_loss=args.det_loss,
                       seg_desc_loss_fn=args.seg_desc_loss_fn,
                       upsample_desc=upsample_desc,
                       seg_desc=args.seg_desc > 0,
                       seg_feat=args.seg_feat > 0,
                       seg_det=args.seg_det > 0,
                       seg_cls=args.seg_cls > 0,
                       )

        if len(args.gpu) > 1:
            print('gpu: ', args.gpu)
            device_ids = [i for i in range(len(args.gpu))]
            net = nn.DataParallel(net, device_ids=device_ids).cuda()
            loss = nn.DataParallel(loss, device_ids=device_ids).cuda()
        else:
            net = net.cuda()
        db = [data_sources[key] for key in args.train_data]
        db = eval(args.data_loader.replace('`data`', ','.join(db)).replace('`R`', str(args.R)).replace('\n', ''))
        print("Training image database =", db)
        loader = threaded_loader(db, iscuda, args.threads, args.bs, shuffle=True)
        trainer = Trainer(net=net, args=args, loader=loader, loss=loss)
        trainer.train(resume=args.resume)
