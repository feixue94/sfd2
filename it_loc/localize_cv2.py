import argparse
import logging
import numpy as np
from pathlib import Path
from collections import defaultdict
import h5py
from tqdm import tqdm
import pickle
import pycolmap
import cv2
from copy import deepcopy
import os.path as osp
from it_loc.common import resize_img, sort_dict_by_value, reproject, reproject_fromR, calc_depth, ColmapQ2R, \
    compute_pose_error, plot_matches, plot_reprojpoint2D


def plot_matches_vis(img1, img2, pts1, pts2, inliers, horizon=True, plot_outlier=False, plot_match=True):
    # print(type(pts1), type(inliers), inliers.shape)
    # print(inliers)
    if type(inliers) == list:
        inliers = np.array(inliers, dtype=bool).reshape(-1, 1)
    rows1 = img1.shape[0]
    cols1 = img1.shape[1]
    rows2 = img2.shape[0]
    cols2 = img2.shape[1]
    r = 3
    if horizon:
        img_out = np.zeros((max([rows1, rows2]), cols1 + cols2, 3), dtype='uint8')
        # Place the first image to the left
        img_out[:rows1, :cols1] = img1
        # Place the next image to the right of it
        img_out[:rows2, cols1:] = img2  # np.dstack([img2, img2, img2])

        for idx in range(inliers.shape[0]):
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)
            pt1 = pts1[idx]
            pt2 = pts2[idx]
            nr = r
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), nr, color, 2)

            img_out = cv2.circle(img_out, (int(pt2[0]) + cols1, int(pt2[1])), nr, color, 2)

            img_out = cv2.line(img_out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]) + cols1, int(pt2[1])), color,
                               2)
    else:
        img_out = np.zeros((rows1 + rows2, max([cols1, cols2]), 3), dtype='uint8')
        # Place the first image to the left
        img_out[:rows1, :cols1] = img1
        # Place the next image to the right of it
        img_out[rows1:, :cols2] = img2  # np.dstack([img2, img2, img2])

        for idx in range(inliers.shape[0]):
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)

            nr = r

            pt1 = pts1[idx]
            pt2 = pts2[idx]
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), r, color, 2)

            img_out = cv2.circle(img_out, (int(pt2[0]), int(pt2[1]) + rows1), r, color, 2)

            img_out = cv2.line(img_out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]) + rows1), color,
                               2)

    # img_rs = cv2.resize(img_out, None, fx=0.5, fy=0.5)
    # return img_rs
    img_out = cv2.resize(img_out, dsize=None, fx=0.5, fy=0.5)
    img_out = cv2.putText(img_out,
                          'inliers:{:d}'.format(np.sum(inliers)),
                          (20, 30),
                          cv2.FONT_HERSHEY_SIMPLEX, 1,
                          (0, 0, 255), 2)
    return img_out


def do_covisibility_clustering(frame_ids, all_images, points3D):
    clusters = []
    visited = set()

    for frame_id in frame_ids:
        # Check if already labeled
        if frame_id in visited:
            continue

        # New component
        clusters.append([])
        queue = {frame_id}
        while len(queue):
            exploration_frame = queue.pop()

            # Already part of the component
            if exploration_frame in visited:
                continue

            visited.add(exploration_frame)
            clusters[-1].append(exploration_frame)

            observed = all_images[exploration_frame].point3D_ids
            connected_frames = set(
                j for i in observed if i != -1 for j in points3D[i].image_ids)
            connected_frames &= set(frame_ids)
            connected_frames -= visited
            queue |= connected_frames

    clusters = sorted(clusters, key=len, reverse=True)
    return clusters


def get_covisibility_frames(frame_id, all_images, points3D, covisibility_frame=50, ref_3Dpoints=None, obs_th=0,
                            pred_qvec=None, pred_tvec=None):
    if ref_3Dpoints is not None:
        observed = ref_3Dpoints
        connected_frames = [j for i in ref_3Dpoints if i != -1 for j in points3D[i].image_ids]
        connected_frames = np.unique(connected_frames)
    else:
        observed = all_images[frame_id].point3D_ids
        connected_frames = [j for i in observed if i != -1 for j in points3D[i].image_ids]
        connected_frames = np.unique(connected_frames)
    print('Find {:d} connected frames'.format(len(connected_frames)))
    valid_db_ids = []
    db_obs = {}
    for db_id in connected_frames:
        p3d_ids = all_images[db_id].point3D_ids
        covisible_p3ds = [v for v in observed if v != -1 and len(points3D[v].image_ids) >= obs_th and v in p3d_ids]
        db_obs[db_id] = len(covisible_p3ds)

    sorted_db_obs = sort_dict_by_value(data=db_obs, reverse=True)

    not_use_db_ids = []
    for item in sorted_db_obs:

        if pred_qvec is not None and pred_tvec is not None:
            db_qvec = all_images[item[0]].qvec
            db_tvec = all_images[item[0]].tvec
            q_error, t_error, _ = compute_pose_error(pred_qcw=pred_qvec, pred_tcw=pred_tvec, gt_qcw=db_qvec,
                                                     gt_tcw=db_tvec)
            if q_error >= 30 or t_error >= 30 or item[1] <= 30:
                not_use_db_ids.append(item[0])
                continue

        valid_db_ids.append(item[0])

        if covisibility_frame > 0:
            if len(valid_db_ids) >= covisibility_frame:
                break
        # print(item[0], item[1], q_error, t_error)

    # if frame_id not in valid_db_ids:
    #     valid_db_ids.append(frame_id)

    if len(valid_db_ids) <= 3:
        for v in not_use_db_ids:
            valid_db_ids.append(v)
            if len(valid_db_ids) >= covisibility_frame:
                break

    print('Retain {:d} valid connected frames'.format(len(valid_db_ids)))
    return valid_db_ids


def get_covisibility_frames_by_pose(frame_id, pred_qvec, pred_tvec, points3D, all_images, covisibility_frame=50, q_th=5,
                                    obs_th=5,
                                    t_th=10, ref_3Dpoints=None):
    if ref_3Dpoints is not None:
        observed = ref_3Dpoints
        connected_frames = [j for i in ref_3Dpoints if i != -1 for j in points3D[i].image_ids]
        connected_frames = np.unique(connected_frames)
    else:
        observed = all_images[frame_id].point3D_ids
        connected_frames = [j for i in observed if i != -1 for j in points3D[i].image_ids]
        connected_frames = np.unique(connected_frames)
    print('Find {:d} connected frames'.format(len(connected_frames)))
    db_ids = []
    db_t_dists = []
    db_q_dists = []

    db_obs = {}
    for db_id in connected_frames:
        name = all_images[db_id].name
        if name.find('left') >= 0 or name.find("right") >= 0:
            continue

        p3d_ids = all_images[db_id].point3D_ids
        covisible_p3ds = [v for v in observed if v != -1 and len(points3D[v].image_ids) >= obs_th and v in p3d_ids]
        db_obs[db_id] = len(covisible_p3ds)

        db_qvec = all_images[db_id].qvec
        db_tvec = all_images[db_id].tvec

        q_error, t_error, _ = compute_pose_error(pred_qcw=pred_qvec, pred_tcw=pred_tvec, gt_qcw=db_qvec, gt_tcw=db_tvec)
        if q_error > q_th:
            continue
        db_ids.append(db_id)
        db_t_dists.append(t_error)
        db_q_dists.append(q_error)

    resort_ids = np.argsort(db_t_dists)
    valid_db_ids = []
    for did in resort_ids:
        valid_db_ids.append(db_ids[did])
        # print('Ref frame {:d}, q_error:{:.3f}, t_error: {:.3f} from pose'.format(len(valid_db_ids), db_q_dists[did],
        #                                                                          db_t_dists[did]))
        if covisibility_frame > 0:
            if len(valid_db_ids) >= covisibility_frame:
                break

    if len(valid_db_ids) >= covisibility_frame:
        print('Retain {:d} valid connected frames'.format(len(valid_db_ids)))
        return valid_db_ids

    sorted_db_obs = sort_dict_by_value(data=db_obs, reverse=True)
    for item in sorted_db_obs:
        if item[0] in valid_db_ids:
            continue
        valid_db_ids.append(item[0])

        # print('Ref frame {:d} from obs'.format(len(valid_db_ids)))
        if covisibility_frame > 0:
            if len(valid_db_ids) >= covisibility_frame:
                break
    print('Retain {:d} valid connected frames'.format(len(valid_db_ids)))
    return valid_db_ids


def pose_refinement_covisibility(qname, cfg, feature_file, db_frame_id, db_images, points3D, thresh, matcher,
                                 with_label=False,
                                 covisibility_frame=50,
                                 ref_3Dpoints=None,
                                 iters=1,
                                 obs_th=3,
                                 opt_th=12,
                                 qvec=None,
                                 tvec=None,
                                 radius=20,
                                 log_info='',
                                 opt_type="ref",
                                 image_dir=None,
                                 vis_dir=None,
                                 gt_qvec=None,
                                 gt_tvec=None,
                                 ):
    if opt_type.find('obs') >= 0:
        db_ids = get_covisibility_frames(frame_id=db_frame_id, all_images=db_images, points3D=points3D,
                                         covisibility_frame=covisibility_frame, ref_3Dpoints=ref_3Dpoints,
                                         obs_th=obs_th, pred_qvec=qvec, pred_tvec=tvec)
    elif opt_type.find('pos') >= 0:
        db_ids = get_covisibility_frames_by_pose(frame_id=db_frame_id, all_images=db_images, points3D=points3D,
                                                 covisibility_frame=covisibility_frame, ref_3Dpoints=ref_3Dpoints,
                                                 pred_qvec=qvec, pred_tvec=tvec, q_th=10, t_th=10, obs_th=obs_th)
    else:
        print('ERROR: Please specify method for getting reference images {:s}'.format(opt_type))
        exit(0)
    # db_ids = get_covisibility_frames(frame_id=db_frame_id, all_images=db_images, points3D=points3D,
    #                                  covisibility_frame=covisibility_frame, ref_3Dpoints=ref_3Dpoints,
    #                                  obs_th=obs_th)

    kpq = feature_file[qname]['keypoints'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    score_q = feature_file[qname]['scores'].__array__()
    desc_q = desc_q.transpose()
    if with_label:
        label_q = feature_file[qname]['labels'].__array__()
    else:
        label_q = None

    # do matching between query and candidate frames
    mp3d = []
    mkpq = []
    mkpdb = []
    # all_obs = []
    all_3D_ids = []
    all_score_q = []
    qid_p3ds = {}
    valid_qid_mp3d_ids = {}
    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        kpdb = feature_file[db_name]['keypoints'].__array__()

        desc_db = feature_file[db_name]["descriptors"].__array__()
        desc_db = desc_db.transpose()

        points3D_ids = db_images[db_id].point3D_ids
        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)
            continue

        if with_label:
            label_db = feature_file[db_name]["labels"].__array__()
            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=label_q, label_db=label_db,
                                       matcher=matcher,
                                       db_3D_ids=points3D_ids)
        else:
            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=None, label_db=None,
                                       matcher=matcher,
                                       db_3D_ids=points3D_ids)

        # matches = matcher(match_data)["matches0"]

        valid = np.where(matches > -1)[0]
        valid = valid[points3D_ids[matches[valid]] != -1]

        # matched_mkq = []
        # matched_mkdb = []
        inliers = []

        for idx in valid:
            id_3D = points3D_ids[matches[idx]]
            if len(points3D[id_3D].image_ids) < obs_th:
                continue

            if idx in valid_qid_mp3d_ids.keys():
                if id_3D in valid_qid_mp3d_ids[idx]:
                    continue
                else:
                    valid_qid_mp3d_ids[idx].append(id_3D)
            else:
                valid_qid_mp3d_ids[idx] = [id_3D]

            if idx in qid_p3ds.keys():
                if id_3D in qid_p3ds[idx]:
                    continue
                else:
                    qid_p3ds[idx].append(id_3D)
            else:
                qid_p3ds[idx] = [id_3D]

            # matched_mkq.append(kpq[idx])
            # matched_mkdb.append(kpdb[matches[idx]])

            if qvec is not None and tvec is not None:
                proj_2d = reproject(points3D=np.array(points3D[id_3D].xyz).reshape(-1, 3), rvec=qvec, tvec=tvec,
                                    camera=cfg)

                proj_error = (kpq[idx] - proj_2d) ** 2
                proj_error = np.sqrt(np.sum(proj_error))
                if proj_error > radius:
                    inliers.append(False)
                    continue

            inliers.append(True)

            mp3d.append(points3D[id_3D].xyz)
            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])
            all_3D_ids.append(id_3D)

            all_score_q.append(score_q[idx])

            # all_obs.append(len(points3D[id_3D].image_ids))

        ### visualize matches
        '''
        q_img = cv2.imread(osp.join(image_dir, qname))
        db_img = cv2.imread(osp.join(image_dir, db_name))
        inliers = np.array(inliers, np.uint8)
        matched_mkq = np.array(matched_mkq).reshape(-1, 2)
        matched_mkdb = np.array(matched_mkdb).reshape(-1, 2)
        img_match = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkq, pts2=matched_mkdb, inliers=inliers, plot_outlier=True)
        img_match = resize_img(img_match, nh=512)
        cv2.imshow("match-q-db", img_match)
        cv2.waitKey(5)
        
        if vis_dir is not None:
            save_fn = 'exmatch_with_proj_{:s}_{:s}.png'.format(qname.replace('/', '-'), db_name.replace('/', '-'))
            cv2.imwrite(osp.join(vis_dir, save_fn), img_match)
        '''

    mp3d = np.array(mp3d, float).reshape(-1, 3)
    mkpq = np.array(mkpq, float).reshape(-1, 2)

    mkpq = mkpq + 0.5

    print_text = 'Get {:d} covisible frames with {:d} matches from cluster optimization'.format(len(db_ids),
                                                                                                mp3d.shape[0])
    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, opt_th)
    # if np.sum(ret['inliers']) <= 20:  # pose estimation failed
    if not ret['success']:
        ret['mkpq'] = mkpq
        ret['3D_ids'] = all_3D_ids
        ret['db_ids'] = db_ids
        ret['score_q'] = all_score_q
        ret['log_info'] = log_info
        ret['qvec'] = qvec
        ret['tvec'] = tvec
        ret['inliers'] = [False for i in range(mkpq.shape[0])]
        ret['num_inliers'] = 0
        return ret

    init_qvec = ret['qvec']
    init_tvec = ret['tvec']
    proj_mkp = reproject(mp3d, rvec=qvec, tvec=tvec, camera=cfg)
    proj_error = (mkpq - proj_mkp) ** 2
    proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
    inlier_mask = (np.array(ret['inliers'], int) > 0)
    mn_error = np.min(proj_error[inlier_mask])
    md_error = np.median(proj_error[inlier_mask])
    mx_error = np.max(proj_error[inlier_mask])

    # depth = calc_depth(points3D=mp3d, rvec=qvec, tvec=tvec, camera=cfg)
    # mn_depth = np.min(depth[inlier_mask])
    # md_depth = np.median(depth[inlier_mask])
    # mx_depth = np.max(depth[inlier_mask])

    if gt_qvec is None or gt_tvec is None:
        q_diff, t_diff, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=init_qvec, gt_tcw=init_tvec)
    else:
        q_diff, t_diff, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=gt_qvec, gt_tcw=gt_tvec)
    print_text = 'Iter: {:d} inliers: {:d} mn_error: {:.2f}, md_error: {:.2f} mx_error: {:.2f}, q_error:{:.1f} t_error:{:.2f}'.format(
        0,
        ret['num_inliers'],
        mn_error,
        md_error,
        mx_error,
        q_diff,
        t_diff,
    )

    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    inliers_rsac = ret['inliers']
    ret['num_inliers'] = np.sum(ret['inliers'])
    # init_qvec = ret['qvec']
    # init_tvec = ret['tvec']
    if opt_type.find("ref") >= 0 and np.sum(inliers_rsac) >= 10:
        for i in range(iters):
            inlier_mask_opt = []
            for pi in range(proj_error.shape[0]):
                if proj_error[pi] <= opt_th and inliers_rsac[pi]:
                    keep = True
                else:
                    keep = False
                inlier_mask_opt.append(keep)

            ret = pycolmap.pose_refinement(tvec, qvec, mkpq, mp3d, inlier_mask_opt, cfg)

            qvec = ret['qvec']
            tvec = ret['tvec']

            proj_mkp = reproject(mp3d, rvec=qvec, tvec=tvec, camera=cfg)
            proj_error = (mkpq - proj_mkp) ** 2
            proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])

            # depth = calc_depth(points3D=mp3d, rvec=qvec, tvec=tvec, camera=cfg)
            # inlier_mask = np.array(inlier_mask_opt).reshape(-1,)
            inlier_mask = (proj_error <= opt_th)  # np.array(inlier_mask_opt).reshape(-1,)

            # print('heiheihei - inliers: ', proj_error[inlier_mask].shape)
            # print('inlier_mask: ', np.sum(inlier_mask))
            # if np.sum(inlier_mask) == 0:
            #     continue

            mn_error = np.min(proj_error[inlier_mask])
            md_error = np.median(proj_error[inlier_mask])
            mx_error = np.max(proj_error[inlier_mask])

            # mn_depth = np.min(depth[inlier_mask])
            # md_depth = np.median(depth[inlier_mask])
            # mx_depth = np.max(depth[inlier_mask])

            if gt_qvec is None or gt_tvec is None:
                q_diff, t_diff, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=init_qvec, gt_tcw=init_tvec)
            else:
                q_diff, t_diff, _ = compute_pose_error(pred_qcw=qvec, pred_tcw=tvec, gt_qcw=gt_qvec, gt_tcw=gt_tvec)
            print_text = 'After Iter:{:d} inliers:{:d}/{:d} mn_error:{:.1f}, md_error:{:.1f} mx_error:{:.1f}, q_error:{:.1f}, t_error:{:.2f}'.format(
                i + 1,
                np.sum(
                    inlier_mask),
                np.sum(inlier_mask_opt),
                mn_error,
                md_error,
                mx_error,
                q_diff,
                t_diff
            )
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

            ret['inliers'] = inlier_mask_opt
            ret['num_inliers'] = np.sum(inlier_mask_opt)

    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')

    ret['mkpq'] = mkpq
    ret['3D_ids'] = all_3D_ids
    ret['db_ids'] = db_ids
    ret['score_q'] = all_score_q
    ret['log_info'] = log_info
    return ret


def feature_matching(desc_q, desc_db, matcher, label_q=None, label_db=None, db_3D_ids=None):
    with_label = (label_q is not None and label_db is not None)
    # print(desc_q.shape, desc_db.shape, db_3D_ids.shape)
    if db_3D_ids is None:
        if with_label:
            match_data = {
                "descriptors0": desc_q,
                "labels0": label_q,
                "descriptors1": desc_db,
                "labels1": label_db,
            }
        else:
            match_data = {
                "descriptors0": desc_q,
                "descriptors1": desc_db,
            }

        # keep the order: 1st: query, 2nd: db
        matches = matcher(match_data)["matches0"]
        return matches
    else:  # perform matching between desc_q and desc_db (with valid 3D points)
        masks = (db_3D_ids != -1)
        # valid_ids = np.where(db_3D_ids != -1)
        valid_desc_db = desc_db[masks]
        valid_ids = [i for i in range(desc_db.shape[0]) if masks[i]]

        if np.sum(masks) <= 3:
            return np.ones((desc_q.shape[0],), dtype=int) * -1

        if with_label:
            valid_label_db = label_db[masks]
            match_data = {
                "descriptors0": desc_q,
                "labels0": label_q,
                "descriptors1": valid_desc_db,
                "labels1": valid_label_db,
            }
        else:
            match_data = {
                "descriptors0": desc_q,
                "descriptors1": valid_desc_db,
            }

        # keep the order: 1st: query, 2nd: db
        matches = matcher(match_data)["matches0"]
        # print('matches: ', matches.shape)
        for i in range(desc_q.shape[0]):
            if matches[i] >= 0:
                matches[i] = valid_ids[matches[i]]
    return matches


def match_cluster_2D(kpq, desc_q, label_q, db_ids, points3D, feature_file, db_images, with_label, matcher, obs_th=0):
    all_mp3d = []
    all_mkpq = []
    all_mp3d_ids = []
    all_q_ids = []
    outputs = {}

    valid_2D_3D_matches = {}
    for i, db_id in enumerate(db_ids):
        db_name = db_images[db_id].name
        kpdb = feature_file[db_name]['keypoints'].__array__()
        desc_db = feature_file[db_name]["descriptors"].__array__()
        desc_db = desc_db.transpose()

        points3D_ids = db_images[db_id].point3D_ids
        if points3D_ids.size == 0:
            print("No 3D points in this db image: ", db_name)
            continue

        # print("desc_q/desc_db", desc_q.shape, desc_db.shape, db_name)
        if with_label:
            label_db = feature_file[db_name]["labels"].__array__()

            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=label_q, label_db=label_db,
                                       matcher=matcher,
                                       db_3D_ids=points3D_ids,
                                       )
        else:
            matches = feature_matching(desc_q=desc_q, desc_db=desc_db, label_q=None, label_db=None,
                                       matcher=matcher, db_3D_ids=points3D_ids)

        mkpdb = []
        mp3d_ids = []
        q_ids = []
        mkpq = []
        mp3d = []
        valid_matches = []
        for idx in range(matches.shape[0]):
            if matches[idx] == -1:
                continue
            if points3D_ids[matches[idx]] == -1:
                continue
            id_3D = points3D_ids[matches[idx]]

            # reject 3d points without enough observations
            if len(points3D[id_3D].image_ids) < obs_th:
                continue

            # remove duplicated matches
            if idx in valid_2D_3D_matches.keys():
                if id_3D in valid_2D_3D_matches[idx]:
                    continue
                else:
                    valid_2D_3D_matches[idx].append(id_3D)
            else:
                valid_2D_3D_matches[idx] = [id_3D]

            mp3d.append(points3D[id_3D].xyz)
            mp3d_ids.append(id_3D)
            all_mp3d_ids.append(id_3D)

            mkpq.append(kpq[idx])
            mkpdb.append(kpdb[matches[idx]])
            q_ids.append(idx)
            all_q_ids.append(idx)

            all_mkpq.append(kpq[idx])
            all_mp3d.append(points3D[id_3D].xyz)

            valid_matches.append(matches[idx])

        outputs[db_name] = {}
        outputs[db_name]['mkpq'] = mkpq
        outputs[db_name]['mkpdb'] = mkpdb
        outputs[db_name]['qids'] = q_ids
        outputs[db_name]['matches'] = np.array(valid_matches, dtype=int)
        outputs[db_name]['mp_3d_ids'] = mp3d_ids
        outputs[db_name]['mp3d'] = np.array(mp3d, dtype=float).reshape(-1, 3)

        print('Find {:d} valid matches from {:d}th candidate'.format(len(valid_matches), i))

    all_mp3d = np.array(all_mp3d, float).reshape(-1, 3)
    all_mkpq = np.array(all_mkpq, float).reshape(-1, 2)

    all_mkpq = all_mkpq + 0.5

    return outputs, all_mp3d, all_mkpq, all_mp3d_ids, all_q_ids


def pose_from_cluster_with_matcher(qname, qinfo, db_ids, db_images, points3D,
                                   feature_file,
                                   thresh,
                                   image_dir,
                                   matcher,
                                   do_covisility_opt=False,
                                   vis_dir=None,
                                   inlier_th=50,
                                   covisibility_frame=50,
                                   log_info=None,
                                   opt_type="cluster",
                                   iters=1,
                                   radius=0,
                                   obs_th=0,
                                   opt_th=12,
                                   gt_qvec=None,
                                   gt_tvec=None,
                                   query_img_prefix='',
                                   db_img_prefix=''
                                   ):
    print("qname: ", qname)
    db_name_to_id = {image.name: i for i, image in db_images.items()}
    q_img = cv2.imread(osp.join(image_dir, query_img_prefix, qname))
    # print('q_img: ', q_img.shape)
    # exit(0)
    kpq = feature_file[qname]['keypoints'].__array__()
    score_q = feature_file[qname]['scores'].__array__()
    desc_q = feature_file[qname]['descriptors'].__array__()
    desc_q = desc_q.transpose()

    # results = {}
    camera_model, width, height, params = qinfo
    cfg = {
        'model': camera_model,
        'width': width,
        'height': height,
        'params': params,
    }

    best_results = {
        'tvec': None,
        'qvec': None,
        'num_inliers': 0,
        'single_num_inliers': 0,
        'db_id': -1,
        'order': -1,
        'qname': qname,
        'optimize': False,
        'dbname': db_images[db_ids[0][0]].name,
        "ret_source": "",
        "inliers": [],
    }

    for cluster_idx, db_id_cls in enumerate(db_ids):
        db_id = db_id_cls[0]
        db_name = db_images[db_id].name
        cluster_info, mp3d, mkpq, mp3d_ids, q_ids = match_cluster_2D(kpq=kpq, desc_q=desc_q,
                                                                     label_q=None,
                                                                     db_ids=db_id_cls,
                                                                     points3D=points3D,
                                                                     feature_file=feature_file,
                                                                     db_images=db_images,
                                                                     with_label=False,
                                                                     matcher=matcher,
                                                                     obs_th=3,
                                                                     )

        if mp3d.shape[0] < 8:  # inlier_th:
            print_text = "qname: {:s} dbname: {:s}({:d}/{:d}) failed because of insufficient 3d points {:d}".format(
                qname,
                db_name,
                cluster_idx + 1,
                len(db_ids),
                mp3d.shape[0])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        ret = pycolmap.absolute_pose_estimation(mkpq, mp3d, cfg, thresh)

        if not ret["success"]:
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed after optimization".format(qname, db_name,
                                                                                                 cluster_idx + 1,
                                                                                                 len(db_ids))
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')
            continue

        inliers = ret['inliers']
        inlier_p3d_ids = [mp3d_ids[i] for i in range(len(inliers)) if inliers[i]]

        # visualize the matches
        q_p3d_ids = np.zeros(shape=(desc_q.shape[0], 1), dtype=int) - 1
        for idx, qid in enumerate(q_ids):
            if inliers[idx]:
                q_p3d_ids[qid] = mp3d_ids[idx]

        best_dbname = None
        best_inliers = -1
        for db_name in cluster_info.keys():
            matched_mp3d_ids = cluster_info[db_name]['mp_3d_ids']
            matched_qids = cluster_info[db_name]['qids']
            n = 0
            for idx, qid in enumerate(matched_qids):
                if matched_mp3d_ids[idx] == q_p3d_ids[qid]:
                    n += 1
            if n > best_inliers:
                best_inliers = n
                best_dbname = db_name



        show_matches = False

        if show_matches and best_dbname is not None:
            # print('best_dbname: ', best_dbname)
            vis_matches = cluster_info[best_dbname]['matches']
            vis_p3d_ids = cluster_info[best_dbname]['mp_3d_ids']
            vis_mkpdb = cluster_info[best_dbname]['mkpdb']
            vis_mkpq = cluster_info[best_dbname]['mkpq']
            vis_mp3d = cluster_info[best_dbname]['mp3d']
            vis_qids = cluster_info[best_dbname]['qids']
            vis_inliers = []  # np.zeros(shape=(vis_matches.shape[0], 1), dtype=np.int) - 1
            for idx, vid in enumerate(vis_qids):
                if vis_p3d_ids[idx] == q_p3d_ids[vid]:
                    vis_inliers.append(True)
                else:
                    vis_inliers.append(False)
            vis_inliers = np.array(vis_inliers, dtype=bool).reshape(-1, 1)

            show_proj = False
            if show_proj:
                matched_points2Ddb = [vis_mkpdb[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
                matched_points2D = [vis_mkpq[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
                matched_points3D = [vis_mp3d[i] for i in range(len(vis_inliers)) if vis_inliers[i]]
                matched_points2Ddb = np.vstack(matched_points2Ddb)
                matched_points2D = np.vstack(matched_points2D)
                matched_points3D = np.vstack(matched_points3D)

                reproj_points2D = reproject(points3D=matched_points3D, rvec=ret['qvec'], tvec=ret['tvec'],
                                            camera=cfg)
                proj_error = (matched_points2D - reproj_points2D) ** 2
                proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])

                min_proj_error = np.min(proj_error)
                max_proj_error = np.max(proj_error)
                med_proj_error = np.median(proj_error)
                # print('proj_error: ',  np.max(proj_error))

                img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_points2D, reproj_points2D=reproj_points2D)
                img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 255, 0), 2)
                img_proj = cv2.putText(img_proj,
                                       'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                              max_proj_error),
                                       (20, 60),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 0, 255), 2)

            db_img = cv2.imread(osp.join(image_dir, db_img_prefix, best_dbname))
            img_pair = plot_matches(img1=q_img, img2=db_img,
                                    pts1=vis_mkpq, pts2=vis_mkpdb,
                                    inliers=vis_inliers, plot_outlier=False, plot_match=False)

            ## for vis comparison
            id_str = '{:03d}'.format(cluster_idx + 1)
            img_vis = plot_matches_vis(img1=q_img, img2=db_img, pts1=vis_mkpq, pts2=vis_mkpdb, inliers=vis_inliers)
            cv2.imwrite(osp.join(vis_dir.as_posix(),
                                 (qname.replace('/', '-') + '+{:s}+'.format(id_str) + best_dbname.replace('/', '-'))),
                        img_vis)

            img_match = plot_matches(img1=q_img, img2=db_img,
                                     pts1=vis_mkpq, pts2=vis_mkpdb,
                                     inliers=vis_inliers, plot_outlier=False)
            img_match_ntext = deepcopy(img_match)
            img_pair = np.hstack([img_pair, img_match_ntext])

            img_match = cv2.putText(img_match, 'm/i/r/o:{:d}/{:d}/{:.2f}/{:d}'.format(
                vis_matches.shape[0], best_inliers, best_inliers / vis_matches.shape[0], cluster_idx + 1),
                                    (20, 30),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)
            vis_obs = [len(points3D[v].image_ids) for v in vis_p3d_ids]
            mn_obs = np.min(vis_obs)
            md_obs = np.median(vis_obs)
            mx_obs = np.max(vis_obs)
            img_match = cv2.putText(img_match,
                                    'obs: mn/md/mx:{:d}/{:d}/{:d}'.format(int(mn_obs), int(md_obs),
                                                                          int(mx_obs)),
                                    (20, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

            ref_q_error, ref_t_error, ref_t_error_xyz = compute_pose_error(pred_qcw=ret['qvec'], pred_tcw=ret['tvec'],
                                                                           gt_qcw=db_images[
                                                                               db_name_to_id[best_dbname]].qvec,
                                                                           gt_tcw=db_images[
                                                                               db_name_to_id[best_dbname]].tvec)
            img_match = cv2.putText(img_match,
                                    'w/ref q:{:.2f}deg t:{:.2f}m'.format(ref_q_error, ref_t_error),
                                    (20, 90),
                                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                                    (0, 0, 255), 2)

            if gt_qvec is not None and gt_tvec is not None:
                gt_reproj_points2D = reproject(points3D=matched_points3D, rvec=gt_qvec, tvec=gt_tvec,
                                               camera=cfg)
                gt_proj_error = (matched_points2D - gt_reproj_points2D) ** 2
                gt_proj_error = np.sqrt(gt_proj_error[:, 0] + gt_proj_error[:, 1])
                gt_min_proj_error = np.min(gt_proj_error)
                gt_max_proj_error = np.max(gt_proj_error)
                gt_med_proj_error = np.median(gt_proj_error)
                img_proj = cv2.putText(img_proj,
                                       'gt-mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(gt_min_proj_error,
                                                                                 gt_med_proj_error,
                                                                                 gt_max_proj_error),
                                       (20, 90),
                                       cv2.FONT_HERSHEY_SIMPLEX, 1,
                                       (0, 0, 255), 2)
                # gt_proj_error = np.sqrt(gt_proj_error[:, 0] + gt_proj_error[:, 1])
                gt_inliers = []
                for i in range(gt_proj_error.shape[0]):
                    if gt_proj_error[i] <= thresh:
                        gt_inliers.append(True)
                    else:
                        gt_inliers.append(False)

                gt_inliers = np.array(gt_inliers, dtype=bool).reshape(-1, 1)
                gt_img_match = plot_matches(img1=q_img, img2=db_img,
                                            pts1=matched_points2D, pts2=matched_points2Ddb,
                                            inliers=gt_inliers, plot_outlier=True)
                gt_img_match = cv2.putText(gt_img_match, 'gt-m/i/r/o:{:d}/{:d}/{:.2f}/{:d}'.format(
                    vis_matches.shape[0], np.sum(gt_inliers), np.sum(gt_inliers) / gt_proj_error.shape[0],
                                                              cluster_idx + 1),
                                           (20, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                q_error, t_error, t_error_xyz = compute_pose_error(pred_qcw=ret['qvec'],
                                                                   pred_tcw=ret['tvec'],
                                                                   gt_qcw=gt_qvec,
                                                                   gt_tcw=gt_tvec)
                gt_img_match = cv2.putText(gt_img_match,
                                           'gt-q_err:{:.2f}deg t_err:{:.2f}m'.format(q_error, t_error),
                                           (20, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)
                gt_img_match = cv2.putText(gt_img_match,
                                           'gt-tx:{:.2f} ty:{:.2f} tz:{:.2f}'.format(t_error_xyz[0], t_error_xyz[1],
                                                                                     t_error_xyz[2]),
                                           (20, 90),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                img_match = np.hstack([img_match, gt_img_match])

            img_pair = resize_img(img_pair, nh=img_match.shape[0])
            img_match = np.hstack([img_pair, img_match])
            img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)

            if show_proj:
                img_proj = resize_img(img_proj, nh=img_match.shape[0])
                # img_match = np.hstack([img_match, img_proj])
                # print('img_match: ', img_match.shape, img_proj.shape)

            # cv2.imshow("match", img_match)

            key = cv2.waitKey(5)
            if vis_dir is not None:
                id_str = '{:03d}'.format(cluster_idx + 1)
                cv2.imwrite(osp.join(vis_dir.as_posix(),
                                     (qname.replace('/', '-') + '_' + id_str + '_' + best_dbname.replace('/', '-'))),
                            img_match)

        if best_inliers < 8:  # at least 8 inliers from a single image
            keep = False
        elif ret['num_inliers'] <= best_results['num_inliers']:
            keep = False
        else:
            keep = True
        if keep:
            best_results['qvec'] = ret['qvec']
            best_results['tvec'] = ret['tvec']
            best_results['inlier'] = ret['inliers']
            best_results['num_inliers'] = ret['num_inliers']
            best_results['single_num_inliers'] = best_inliers
            best_results['dbname'] = best_dbname
            best_results['order'] = cluster_idx + 1
        if ret['num_inliers'] < inlier_th or best_inliers < 10:
            print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) failed insufficient {:d}/{:d} inliers".format(qname,
                                                                                                             best_dbname,

                                                                                                             cluster_idx + 1,
                                                                                                             len(
                                                                                                                 db_ids),
                                                                                                             best_inliers,
                                                                                                             ret[
                                                                                                                 "num_inliers"])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + '\n')

            continue

        if not keep:
            best_results['qvec'] = ret['qvec']
            best_results['tvec'] = ret['tvec']
            best_results['inlier'] = ret['inliers']
            best_results['num_inliers'] = ret['num_inliers']
            best_results['single_num_inliers'] = best_inliers
            best_results['dbname'] = best_dbname
            best_results['order'] = cluster_idx + 1

        print_text = "qname: {:s} dbname: {:s} ({:d}/{:d}) initialization succeed with {:d}/{:d} inliers".format(
            qname,
            best_dbname,
            cluster_idx + 1,
            len(db_ids),
            best_inliers,
            ret["num_inliers"]
        )
        print(print_text)
        if log_info is not None:
            log_info += (print_text + '\n')

        if do_covisility_opt:
            if opt_type.find('clu') >= 0:
                ret = pose_refinement_covisibility(qname=qname,
                                                   cfg=cfg,
                                                   feature_file=feature_file,
                                                   db_frame_id=db_name_to_id[best_dbname],
                                                   db_images=db_images, points3D=points3D,
                                                   thresh=thresh,
                                                   with_label=False,
                                                   covisibility_frame=covisibility_frame,
                                                   matcher=matcher,
                                                   # ref_3Dpoints=inlier_p3d_ids,
                                                   ref_3Dpoints=None,
                                                   iters=iters,
                                                   obs_th=obs_th,
                                                   opt_th=opt_th,
                                                   radius=radius,
                                                   qvec=ret['qvec'],
                                                   tvec=ret['tvec'],
                                                   log_info='',
                                                   opt_type=opt_type,
                                                   image_dir=image_dir,
                                                   vis_dir=vis_dir,
                                                   gt_qvec=gt_qvec,
                                                   gt_tvec=gt_tvec,
                                                   )
            log_info = log_info + ret['log_info']
            print_text = 'Find {:d} inliers after optimization'.format(ret['num_inliers'])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

            if not ret['success']:
                continue

            show_refinment = False
            if show_refinment:
                all_ref_db_ids = ret['db_ids']
                all_mkpq = ret['mkpq']
                all_3D_ids = ret['3D_ids']
                inlier_mask = ret['inliers']

                dbname_ninliers = {}
                dbname_matches = {}
                for did in all_ref_db_ids:
                    db_p3D_ids = db_images[did].point3D_ids
                    dbname = db_images[did].name
                    mkdb = feature_file[dbname]['keypoints'].__array__()

                    matched_mkpq = []
                    matched_p3d = []
                    matched_mkdb = []
                    matched_obs = []
                    for pi in range(len(all_3D_ids)):
                        if not inlier_mask[pi]:
                            continue
                        p3D_id = all_3D_ids[pi]
                        if p3D_id in db_p3D_ids:
                            matched_mkpq.append(all_mkpq[pi])
                            matched_p3d.append(points3D[p3D_id].xyz)
                            mkdb_idx = list(db_p3D_ids).index(p3D_id)
                            matched_mkdb.append(mkdb[mkdb_idx])

                            obs = len(points3D[p3D_id].image_ids)
                            matched_obs.append(obs)

                    if len(matched_p3d) == 0:
                        continue

                    dbname_matches[dbname] = {
                        'mkpq': np.array(matched_mkpq).reshape(-1, 2),
                        'mp3d': np.array(matched_p3d).reshape(-1, 3),
                        'mkpdb': np.array(matched_mkdb).reshape(-1, 2),
                        'min_obs': np.min(matched_obs),
                        'median_obs': np.median(matched_obs),
                        'max_obs': np.max(matched_obs),
                        'all_obs': matched_obs,
                    }
                    dbname_ninliers[dbname] = len(matched_p3d)

                sorted_dbname_ninliers = sort_dict_by_value(data=dbname_ninliers, reverse=True)

                for idx, item in enumerate(sorted_dbname_ninliers):
                    if item[1] == 0:
                        continue

                    dbname = item[0]
                    db_img = cv2.imread(osp.join(image_dir, dbname))

                    matched_mkpq = dbname_matches[dbname]['mkpq']
                    matched_mkdb = dbname_matches[dbname]['mkpdb']
                    matched_p3d = dbname_matches[dbname]['mp3d']
                    reproj_mkpq = reproject(points3D=matched_p3d, rvec=ret['qvec'], tvec=ret['tvec'], camera=cfg)
                    proj_error = (matched_mkpq - reproj_mkpq) ** 2
                    proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
                    min_proj_error = np.min(proj_error)
                    max_proj_error = np.max(proj_error)
                    med_proj_error = np.median(proj_error)
                    img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_mkpq, reproj_points2D=reproj_mkpq)
                    img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                    img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 255, 0), 2)
                    img_proj = cv2.putText(img_proj,
                                           'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                                  max_proj_error),
                                           (20, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                    img_match = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkpq, pts2=matched_mkdb,
                                             inliers=np.array([True for i in range(matched_mkpq.shape[0])], np.uint8))
                    img_match = cv2.putText(img_match, 'i/o:{:d}/{:d}'.format(matched_mkpq.shape[0], idx + 1),
                                            (50, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2
                                            )

                    mn_obs = dbname_matches[dbname]['min_obs']
                    md_obs = dbname_matches[dbname]['median_obs']
                    mx_obs = dbname_matches[dbname]['max_obs']
                    img_match = cv2.putText(img_match,
                                            'obs-mn/md/mx:{:d}/{:d}/{:d}'.format(int(mn_obs),
                                                                                 int(md_obs),
                                                                                 int(mx_obs)),
                                            (20, 60),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2)
                    img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)
                    img_proj = resize_img(img_proj, nh=img_match.shape[0])
                    img_match = np.hstack([img_match, img_proj])

                    cv2.imshow('match_ref', img_match)
                    key = cv2.waitKey(5)
                    if vis_dir is not None:
                        id_str = '{:03d}'.format(idx + 1)
                        cv2.imwrite(osp.join(vis_dir.as_posix(),
                                             (qname.replace('/',
                                                            '-') + '_' + opt_type + id_str + '_' + dbname.replace(
                                                 '/',
                                                 '-'))),
                                    img_match)

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        ret['cfg'] = cfg
        num_inliers = ret['num_inliers']

        return qvec, tvec, num_inliers, {**best_results, **{'log_info': log_info}}

    if best_results['num_inliers'] >= 10:  # 20 for aachen
        qvec = best_results['qvec']
        tvec = best_results['tvec']
        num_inliers = best_results['num_inliers']
        best_dbname = best_results['dbname']
        inliers = best_results['inliers']

        if do_covisility_opt:
            if opt_type.find('clu') >= 0:
                ret = pose_refinement_covisibility(qname=qname,
                                                   cfg=cfg,
                                                   feature_file=feature_file,
                                                   db_frame_id=db_name_to_id[best_dbname],
                                                   db_images=db_images, points3D=points3D,
                                                   thresh=thresh,
                                                   with_label=False,
                                                   covisibility_frame=covisibility_frame,
                                                   matcher=matcher,
                                                   # ref_3Dpoints=inlier_p3d_ids,
                                                   ref_3Dpoints=None,
                                                   iters=iters,
                                                   obs_th=obs_th,
                                                   opt_th=opt_th,
                                                   radius=radius,
                                                   qvec=qvec,
                                                   tvec=tvec,
                                                   log_info='',
                                                   opt_type=opt_type,
                                                   image_dir=image_dir,
                                                   vis_dir=vis_dir,
                                                   gt_qvec=gt_qvec,
                                                   gt_tvec=gt_tvec,
                                                   )
            log_info = log_info + ret['log_info']
            print_text = 'Find {:d} inliers after optimization'.format(ret['num_inliers'])
            print(print_text)
            if log_info is not None:
                log_info += (print_text + "\n")

            show_refinment = False
            if show_refinment:
                all_ref_db_ids = ret['db_ids']
                all_mkpq = ret['mkpq']
                all_3D_ids = ret['3D_ids']
                inlier_mask = ret['inliers']

                dbname_ninliers = {}
                dbname_matches = {}
                for did in all_ref_db_ids:
                    db_p3D_ids = db_images[did].point3D_ids
                    dbname = db_images[did].name
                    mkdb = feature_file[dbname]['keypoints'].__array__()

                    matched_mkpq = []
                    matched_p3d = []
                    matched_mkdb = []
                    for pi in range(len(all_3D_ids)):
                        if not inlier_mask[pi]:
                            continue
                        p3D_id = all_3D_ids[pi]
                        if p3D_id in db_p3D_ids:
                            matched_mkpq.append(all_mkpq[pi])
                            matched_p3d.append(points3D[p3D_id].xyz)
                            mkdb_idx = list(db_p3D_ids).index(p3D_id)
                            matched_mkdb.append(mkdb[mkdb_idx])

                    if len(matched_p3d) == 0:
                        continue

                    dbname_matches[dbname] = {
                        'mkpq': np.array(matched_mkpq).reshape(-1, 2),
                        'mp3d': np.array(matched_p3d).reshape(-1, 3),
                        'mkpdb': np.array(matched_mkdb).reshape(-1, 2),
                    }
                    dbname_ninliers[dbname] = len(matched_p3d)

                sorted_dbname_ninliers = sort_dict_by_value(data=dbname_ninliers, reverse=True)

                for idx, item in enumerate(sorted_dbname_ninliers):
                    if item[1] == 0:
                        continue

                    dbname = item[0]
                    db_img = cv2.imread(osp.join(image_dir, db_img_prefix, dbname))

                    matched_mkpq = dbname_matches[dbname]['mkpq']
                    matched_mkdb = dbname_matches[dbname]['mkpdb']
                    matched_p3d = dbname_matches[dbname]['mp3d']
                    reproj_mkpq = reproject(points3D=matched_p3d, rvec=ret['qvec'], tvec=ret['tvec'], camera=cfg)
                    proj_error = (matched_mkpq - reproj_mkpq) ** 2
                    proj_error = np.sqrt(proj_error[:, 0] + proj_error[:, 1])
                    min_proj_error = np.min(proj_error)
                    max_proj_error = np.max(proj_error)
                    med_proj_error = np.median(proj_error)
                    img_proj = plot_reprojpoint2D(img=q_img, points2D=matched_mkpq, reproj_points2D=reproj_mkpq)
                    img_proj = cv2.resize(img_proj, None, fx=0.5, fy=0.5)
                    img_proj = cv2.putText(img_proj, 'green p2D/red-proj', (20, 30),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 255, 0), 2)
                    img_proj = cv2.putText(img_proj,
                                           'mn/md/mx:{:.1f}/{:.1f}/{:.1f}'.format(min_proj_error, med_proj_error,
                                                                                  max_proj_error),
                                           (20, 60),
                                           cv2.FONT_HERSHEY_SIMPLEX, 1,
                                           (0, 0, 255), 2)

                    img_match = plot_matches(img1=q_img, img2=db_img, pts1=matched_mkpq, pts2=matched_mkdb,
                                             inliers=np.array([True for i in range(matched_mkpq.shape[0])], np.uint8))
                    img_match = cv2.putText(img_match, 'i/o:{:d}/{:d}'.format(matched_mkpq.shape[0], idx + 1),
                                            (50, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1,
                                            (0, 0, 255), 2
                                            )
                    img_match = cv2.resize(img_match, None, fx=0.5, fy=0.5)
                    img_proj = resize_img(img_proj, nh=img_match.shape[0])
                    img_match = np.hstack([img_match, img_proj])

                    # cv2.imshow('match_ref', img_match)
                    key = cv2.waitKey(5)
                    if vis_dir is not None:
                        id_str = '{:03d}'.format(idx + 1)
                        cv2.imwrite(osp.join(vis_dir.as_posix(),
                                             (qname.replace('/', '-') + '_' + opt_type + id_str + '_' + dbname.replace(
                                                 '/', '-'))),
                                    img_match)

        # localization succeed
        qvec = ret['qvec']
        tvec = ret['tvec']
        ret['cfg'] = cfg
        num_inliers = ret['num_inliers']

        # return qvec, tvec, num_inliers, {**best_results, **{'log_info': log_info}}
        return qvec, tvec, 0, {**best_results, **{'log_info': log_info}}

    closest = db_images[db_ids[0][0]]
    # closest = db_images[db_ids[0][0]]
    print_text = 'Localize {:s} failed, but use the pose of {:s} as approximation'.format(qname, closest.name)
    print(print_text)
    if log_info is not None:
        log_info += (print_text + '\n')
    return closest.qvec, closest.tvec, -1, {**best_results, **{'log_info': log_info}}

# opencv-contrib-python         3.4.2.16
# opencv-python                 3.4.2.16
