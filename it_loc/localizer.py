# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   Hierarchical-Localization -> it_loc
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   05/07/2022 10:10
=================================================='''
import os
import os.path as osp
from tqdm import tqdm
import argparse
import time
import logging
import h5py
import numpy as np
from pathlib import Path

from it_loc.read_write_model import read_model
from it_loc.parsers import parse_image_lists_with_intrinsics
from it_loc.loc_tools import read_retrieval_results, compute_pose_error
from it_loc.localize_cv2 import pose_from_cluster_with_matcher, do_covisibility_clustering
from it_loc.matcher import Matcher, confs


def run(args):
    # for visualization only (not used for localization)
    if args.gt_pose_fn is not None:
        gt_poses = {}
        with open(args.gt_pose_fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split(' ')
                gt_poses[l[0].split('/')[-1]] = {
                    'qvec': np.array([float(v) for v in l[1:5]], float),
                    'tvec': np.array([float(v) for v in l[5:]], float),
                }
    else:
        gt_poses = {}

    retrievals = read_retrieval_results(args.retrieval)
    save_root = args.save_root  # path to save
    matcher_name = args.matcher_method  # matching method
    matcher = Matcher(conf=confs[args.matcher_method])
    matcher = matcher.eval().cuda()
    local_feat_name = args.features.as_posix().split("/")[-1].split(".")[0]  # name of local features

    save_fn = '{:s}_{:s}_{:s}_{:.0f}_{:d}'.format(local_feat_name, matcher_name, args.init_type, args.ransac_thresh,
                                                  args.inlier_thresh)
    save_fn = osp.join(save_root, save_fn)

    queries = parse_image_lists_with_intrinsics(args.queries)
    _, db_images, points3D = read_model(str(args.reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    feature_file = h5py.File(args.features, 'r')

    tag = 'all'
    if args.do_covisible_opt:
        tag = tag + "_o" + str(int(args.obs_thresh)) + 'op' + str(int(args.covisibility_frame))
        tag = tag + args.opt_type + "th" + str(int(args.opt_thresh)) + "r" + str(args.radius)
        if args.iters > 0:
            tag = tag + "i" + str(int(args.iters))
    log_fn = save_fn + tag
    vis_dir = save_fn + tag
    results = save_fn + tag

    full_log_fn = log_fn + '_full.log'
    log_fn = Path(log_fn + '.log')
    results = Path(results + '.txt')
    vis_dir = Path(vis_dir)
    if vis_dir is not None:
        Path(vis_dir).mkdir(exist_ok=True)
    print("save_fn: ", log_fn)

    logging.info('Starting localization...')
    poses = {}
    failed_cases = []
    all_logs = []
    n_total = 0
    n_failed = 0
    n_top1 = 0
    full_log_info = ''

    error_ths = ((0.25, 2), (0.5, 5), (5, 10))
    success = [0, 0, 0]
    n_gt_total = 0

    for qname, qinfo in tqdm(queries):
        time_start = time.time()

        if qname in retrievals.keys():
            cans = retrievals[qname]
        db_ids = []

        if args.init_type == 'sng':
            for c in cans:
                if c not in db_name_to_id:
                    logging.warning(f'Image {c} was retrieved but not in database')
                    continue
                # full_log_info += ('{:s} {:s} {:.2f} by global search\n'.format(qname, c, 1))
                db_ids.append([db_name_to_id[c]])
        elif args.init_type == 'clu':
            frame_ids = []
            for c in cans:
                if c not in db_name_to_id:
                    logging.warning(f'Image {c} was retrieved but not in database')
                    continue
                frame_ids.append(db_name_to_id[c])

            clusters = do_covisibility_clustering(frame_ids=frame_ids, all_images=db_images, points3D=points3D)
            # print('clu: ', clusters)
            db_ids = clusters
            # print('clu: ', db_ids)
        time_coarse = time.time()

        qvec, tvec, n_inliers, logs = pose_from_cluster_with_matcher(qname=qname,
                                                                     qinfo=qinfo,
                                                                     matcher=matcher,
                                                                     db_ids=db_ids,
                                                                     db_images=db_images,
                                                                     points3D=points3D,
                                                                     feature_file=feature_file,
                                                                     thresh=args.ransac_thresh,
                                                                     image_dir=args.image_dir,
                                                                     do_covisility_opt=args.do_covisible_opt,
                                                                     vis_dir=vis_dir,
                                                                     covisibility_frame=args.covisibility_frame,
                                                                     log_info='',
                                                                     inlier_th=args.inlier_thresh,
                                                                     opt_type=args.opt_type,
                                                                     iters=args.iters,
                                                                     radius=args.radius,
                                                                     obs_th=args.obs_thresh,
                                                                     opt_th=args.opt_thresh,
                                                                     gt_qvec=gt_poses[qname.split('/')[-1]][
                                                                         'qvec'] if qname.split('/')[
                                                                                        -1] in gt_poses.keys() else None,
                                                                     gt_tvec=gt_poses[qname.split('/')[-1]][
                                                                         'tvec'] if qname.split('/')[
                                                                                        -1] in gt_poses.keys() else None,
                                                                     )
        time_full = time.time()
        all_logs.append(logs)
        n_total += 1
        if n_inliers == 0:
            failed_cases.append(qname)
            n_failed += 1
        if logs['order'] == 1:
            n_top1 += 1
        full_log_info = full_log_info + logs['log_info']
        poses[qname] = (qvec, tvec)
        print_text = "All {:d}/{:d} failed cases top@1:{:.2f}, time[cs/fn]: {:.2f}/{:.2f}".format(
            n_failed, n_total,
            n_top1 / n_total,
            time_coarse - time_start,
            time_full - time_coarse,
        )

        if qname.split('/')[-1] in gt_poses.keys():
            gt_qvec = gt_poses[qname.split('/')[-1]]['qvec']
            gt_tvec = gt_poses[qname.split('/')[-1]]['tvec']

            q_error, t_error, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=gt_qvec, gt_tcw=gt_tvec)

            for error_idx, th in enumerate(error_ths):
                if t_error <= th[0] and q_error <= th[1]:
                    success[error_idx] += 1
            n_gt_total += 1
            print_text += (
                ', q_error:{:.2f} t_error:{:.2f} {:d}/{:d}/{:d}/{:d}'.format(q_error, t_error, success[0], success[1],
                                                                             success[2], n_gt_total))

        print(print_text)
        full_log_info += (print_text + "\n")

    logs_path = f'{results}.failed'
    with open(logs_path, 'w') as f:
        for v in failed_cases:
            print(v)
            f.write(v + "\n")

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            if args.dataset in ['aachen', 'aachen_v1.1']:
                name = q.split('/')[-1]
            elif args.dataset == 'robotcar':
                name = q.split('/')[-2] + "/" + q.split('/')[-1]  # rear/img_id

            f.write(f'{name} {qvec} {tvec}\n')

    with open(log_fn, 'w') as f:
        for v in all_logs:
            f.write('{:s} {:s} {:d} {:d}\n'.format(v['qname'], v['dbname'], v['num_inliers'], v['order']))

    with open(full_log_fn, 'w') as f:
        f.write(full_log_info)
    logging.info('Done!')


def run_ecmu(args):
    # for visualization only (not used for localization)
    if args.gt_pose_fn is not None:
        gt_poses = {}
        with open(args.gt_pose_fn, 'r') as f:
            lines = f.readlines()
            for l in lines:
                l = l.strip().split(' ')
                gt_poses[l[0].split('/')[-1]] = {
                    'qvec': np.array([float(v) for v in l[1:5]], float),
                    'tvec': np.array([float(v) for v in l[5:]], float),
                }
    else:
        gt_poses = {}

    retrievals = read_retrieval_results(args.retrieval)
    save_root = args.save_root  # path to save
    matcher_name = args.matcher_method  # matching method
    matcher = Matcher(conf=confs[args.matcher_method])
    matcher = matcher.eval().cuda()
    local_feat_name = args.features.as_posix().split("/")[-1].split(".")[0]  # name of local features

    save_fn = 'slice{:d}_{:s}_{:s}_{:s}_{:.0f}_{:d}'.format(args.slice,
                                                            local_feat_name, matcher_name, args.init_type,
                                                            args.ransac_thresh,
                                                            args.inlier_thresh)
    save_fn = osp.join(save_root, save_fn)

    queries = parse_image_lists_with_intrinsics(args.queries)
    _, db_images, points3D = read_model(str(args.reference_sfm), '.bin')
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    feature_file = h5py.File(args.features, 'r')

    # tag = 'tst1'
    # tag = 'all'
    # tag = 'day'
    # tag = 'nht'
    # tag = 'nnr'
    # tag = 'day'
    tag = 'vis'
    if args.do_covisible_opt:
        tag = tag + "_o" + str(int(args.obs_thresh)) + 'op' + str(int(args.covisibility_frame))
        tag = tag + args.opt_type + "th" + str(int(args.opt_thresh)) + "r" + str(args.radius)
        if args.iters > 0:
            tag = tag + "i" + str(int(args.iters))
    log_fn = save_fn + tag
    vis_dir = save_fn + tag
    results = save_fn + tag

    full_log_fn = log_fn + '_full.log'
    log_fn = Path(log_fn + '.log')
    results = Path(results + '.txt')
    vis_dir = Path(vis_dir)
    if vis_dir is not None:
        Path(vis_dir).mkdir(exist_ok=True)
    print("save_fn: ", log_fn)

    logging.info('Starting localization...')
    poses = {}
    failed_cases = []
    all_logs = []
    n_total = 0
    n_failed = 0
    n_top1 = 0
    full_log_info = ''

    error_ths = ((0.25, 2), (0.5, 5), (5, 10))
    success = [0, 0, 0]
    n_gt_total = 0

    for qname, qinfo in tqdm(queries):
        time_start = time.time()

        if qname in retrievals.keys():
            cans = retrievals[qname]
        db_ids = []

        if args.init_type == 'sng':
            for c in cans:
                if c not in db_name_to_id:
                    logging.warning(f'Image {c} was retrieved but not in database')
                    continue
                # full_log_info += ('{:s} {:s} {:.2f} by global search\n'.format(qname, c, 1))
                db_ids.append([db_name_to_id[c]])
        elif args.init_type == 'clu':
            frame_ids = []
            for c in cans:
                if c not in db_name_to_id:
                    logging.warning(f'Image {c} was retrieved but not in database')
                    continue
                frame_ids.append(db_name_to_id[c])

            clusters = do_covisibility_clustering(frame_ids=frame_ids, all_images=db_images, points3D=points3D)
            # print('clu: ', clusters)
            db_ids = clusters
            # print('clu: ', db_ids)
        time_coarse = time.time()

        qvec, tvec, n_inliers, logs = pose_from_cluster_with_matcher(qname=qname,
                                                                     qinfo=qinfo,
                                                                     matcher=matcher,
                                                                     db_ids=db_ids,
                                                                     db_images=db_images,
                                                                     points3D=points3D,
                                                                     feature_file=feature_file,
                                                                     thresh=args.ransac_thresh,
                                                                     image_dir=args.image_dir,
                                                                     do_covisility_opt=args.do_covisible_opt,
                                                                     vis_dir=vis_dir,
                                                                     covisibility_frame=args.covisibility_frame,
                                                                     log_info='',
                                                                     inlier_th=args.inlier_thresh,
                                                                     opt_type=args.opt_type,
                                                                     iters=args.iters,
                                                                     radius=args.radius,
                                                                     obs_th=args.obs_thresh,
                                                                     opt_th=args.opt_thresh,
                                                                     gt_qvec=gt_poses[qname.split('/')[-1]][
                                                                         'qvec'] if qname.split('/')[
                                                                                        -1] in gt_poses.keys() else None,
                                                                     gt_tvec=gt_poses[qname.split('/')[-1]][
                                                                         'tvec'] if qname.split('/')[
                                                                                        -1] in gt_poses.keys() else None,
                                                                     query_img_prefix='query',
                                                                     db_img_prefix='database',
                                                                     )
        time_full = time.time()
        all_logs.append(logs)
        n_total += 1
        if n_inliers == 0:
            failed_cases.append(qname)
            n_failed += 1
        if logs['order'] == 1:
            n_top1 += 1
        full_log_info = full_log_info + logs['log_info']
        poses[qname] = (qvec, tvec)
        print_text = "All {:d}/{:d} failed cases top@1:{:.2f}, time[cs/fn]: {:.2f}/{:.2f}".format(
            n_failed, n_total,
            n_top1 / n_total,
            time_coarse - time_start,
            time_full - time_coarse,
        )

        if qname.split('/')[-1] in gt_poses.keys():
            gt_qvec = gt_poses[qname.split('/')[-1]]['qvec']
            gt_tvec = gt_poses[qname.split('/')[-1]]['tvec']

            q_error, t_error, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=gt_qvec, gt_tcw=gt_tvec)

            for error_idx, th in enumerate(error_ths):
                if t_error <= th[0] and q_error <= th[1]:
                    success[error_idx] += 1
            n_gt_total += 1
            print_text += (
                ', q_error:{:.2f} t_error:{:.2f} {:d}/{:d}/{:d}/{:d}'.format(q_error, t_error, success[0], success[1],
                                                                             success[2], n_gt_total))

        print(print_text)
        full_log_info += (print_text + "\n")

    logs_path = f'{results}.failed'
    with open(logs_path, 'w') as f:
        for v in failed_cases:
            print(v)
            f.write(v + "\n")

    logging.info(f'Localized {len(poses)} / {len(queries)} images.')
    logging.info(f'Writing poses to {results}...')
    with open(results, 'w') as f:
        for q in poses:
            qvec, tvec = poses[q]
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            if args.dataset in ['aachen', 'aachen_v1.1', 'ecmu']:
                name = q.split('/')[-1]
            elif args.dataset == 'robotcar':
                name = q.split('/')[-2] + "/" + q.split('/')[-1]  # rear/img_id

            f.write(f'{name} {qvec} {tvec}\n')

    with open(log_fn, 'w') as f:
        for v in all_logs:
            f.write('{:s} {:s} {:d} {:d}\n'.format(v['qname'], v['dbname'], v['num_inliers'], v['order']))

    with open(full_log_fn, 'w') as f:
        f.write(full_log_info)
    logging.info('Done!')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str,
                        default="/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright", )
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--db_imglist_fn', type=str, required=True)
    parser.add_argument('--reference_sfm', type=Path, required=True)
    parser.add_argument('--queries', type=Path, required=True)
    parser.add_argument('--features', type=Path, required=True)
    parser.add_argument('--ransac_thresh', type=float, default=12)
    parser.add_argument('--covisibility_frame', type=int, default=50)
    parser.add_argument('--with_match', action='store_true')
    parser.add_argument('--do_covisible_opt', action='store_true')
    parser.add_argument('--init_type', type=str, default='clu', choices=['sng', 'clu'])
    parser.add_argument('--opt_type', type=str, default='cluster')
    parser.add_argument('--matcher_method', type=str, default="NNM")

    parser.add_argument('--inlier_thresh', type=int, default=50)
    parser.add_argument('--obs_thresh', type=float, default=3)
    parser.add_argument('--opt_thresh', type=float, default=12)
    parser.add_argument('--radius', type=int, default=20)
    parser.add_argument('--iters', type=int, default=1)
    parser.add_argument('--save_root', type=str, default="/data/cornucopia/fx221/exp/shloc/aachen")
    parser.add_argument('--retrieval', type=Path, default=None)
    parser.add_argument('--gt_pose_fn', type=str, default=None)
    parser.add_argument('--only_gt', type=int, default=0)
    parser.add_argument('--slice', type=int, default=2)

    args = parser.parse_args()
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'

    if args.dataset == 'ecmu':
        run_ecmu(args=args)
    else:
        run(args=args)
