# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   Hierarchical-Localization -> common
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   05/07/2022 10:24
=================================================='''

import numpy as np
import cv2
import torch
from copy import copy
from scipy.spatial.transform import Rotation as sciR


def sort_dict_by_value(data, reverse=False):
    return sorted(data.items(), key=lambda d: d[1], reverse=reverse)


def resize_img(img, nh=-1, nw=-1, mode=cv2.INTER_NEAREST):
    assert nh > 0 or nw > 0
    if nh == -1:
        return cv2.resize(img, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)), interpolation=mode)
    if nw == -1:
        return cv2.resize(img, dsize=(int(img.shape[1] / img.shape[0] * nh), nh), interpolation=mode)
    return cv2.resize(img, dsize=(nw, nh), interpolation=mode)


def plot_keypoint(img_path, pts, scores=None, tag=None, save_path=None):
    if type(img_path) == str:
        img = cv2.imread(img_path)
    else:
        img = img_path.copy()

    img_out = img.copy()
    print(img.shape)
    r = 3
    for i in range(pts.shape[0]):
        pt = pts[i]
        # img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), int(r * s), (0, 0, 255), 4)
        img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 2)

    if save_path is not None:
        cv2.imwrite(save_path, img_out)
    return img_out


def sort_dict_by_value(data, reverse=False):
    return sorted(data.items(), key=lambda d: d[1], reverse=reverse)


def read_retrieval_results(path):
    output = {}
    with open(path, "r") as f:
        lines = f.readlines()
        for p in lines:
            p = p.strip("\n").split(" ")

            if p[1] == "no_match":
                continue
            if p[0] in output.keys():
                output[p[0]].append(p[1])
            else:
                output[p[0]] = [p[1]]
    return output


def nn_k(query_gps, db_gps, k=20):
    q = torch.from_numpy(query_gps)  # [N 2]
    db = torch.from_numpy(db_gps)  # [M, 2]
    # print (q.shape, db.shape)
    dist = q.unsqueeze(2) - db.t().unsqueeze(0)
    dist = dist[:, 0, :] ** 2 + dist[:, 1, :] ** 2
    print("dist: ", dist.shape)
    topk = torch.topk(dist, dim=1, k=k, largest=False)[1]
    return topk


def plot_matches(img1, img2, pts1, pts2, inliers, horizon=False, plot_outlier=False, confs=None, plot_match=True):
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

        if not plot_match:
            return cv2.resize(img_out, None, fx=0.5, fy=0.5)
        # for idx, pt in enumerate(pts1):
        #     img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 2)
        # for idx, pt in enumerate(pts2):
        #     img_out = cv2.circle(img_out, (int(pt[0] + cols1), int(pt[1])), r, (0, 0, 255), 2)
        for idx in range(inliers.shape[0]):
            # if idx % 10 > 0:
            #     continue
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)
            pt1 = pts1[idx]
            pt2 = pts2[idx]

            if confs is not None:
                nr = int(r * confs[idx])
            else:
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

        if not plot_match:
            return cv2.resize(img_out, None, fx=0.5, fy=0.5)
        # for idx, pt in enumerate(pts1):
        #     img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1])), r, (0, 0, 255), 2)
        # for idx, pt in enumerate(pts2):
        #     img_out = cv2.circle(img_out, (int(pt[0]), int(pt[1] + rows1)), r, (0, 0, 255), 2)
        for idx in range(inliers.shape[0]):
            # print("idx: ", inliers[idx])
            # if idx % 10 > 0:
            #     continue
            if inliers[idx]:
                color = (0, 255, 0)
            else:
                if not plot_outlier:
                    continue
                color = (0, 0, 255)

            if confs is not None:
                nr = int(r * confs[idx])
            else:
                nr = r

            pt1 = pts1[idx]
            pt2 = pts2[idx]
            img_out = cv2.circle(img_out, (int(pt1[0]), int(pt1[1])), r, color, 2)

            img_out = cv2.circle(img_out, (int(pt2[0]), int(pt2[1]) + rows1), r, color, 2)

            img_out = cv2.line(img_out, (int(pt1[0]), int(pt1[1])), (int(pt2[0]), int(pt2[1]) + rows1), color,
                               2)

    img_rs = cv2.resize(img_out, None, fx=0.5, fy=0.5)

    # img_rs = cv2.putText(img_rs, 'matches: {:d}'.format(len(inliers.shape[0])), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2,
    #                      (0, 0, 255), 2)

    # if save_fn is not None:
    #     cv2.imwrite(save_fn, img_rs)
    # cv2.imshow("match", img_rs)
    # cv2.waitKey(0)
    return img_rs


def plot_reprojpoint2D(img, points2D, reproj_points2D, confs=None):
    img_out = copy(img)
    r = 5
    for i in range(points2D.shape[0]):
        p = points2D[i]
        rp = reproj_points2D[i]

        if confs is not None:
            nr = int(r * confs[i])
        else:
            nr = r

        if nr >= 50:
            nr = 50
        # img_out = cv2.circle(img_out, (int(p[0]), int(p[1])), nr, color=(0, 255, 0), thickness=2)
        img_out = cv2.circle(img_out, (int(rp[0]), int(rp[1])), nr, color=(0, 0, 255), thickness=3)
        img_out = cv2.circle(img_out, (int(rp[0]), int(rp[1])), 2, color=(0, 0, 255), thickness=3)
        # img_out = cv2.line(img_out, pt1=(int(p[0]), int(p[1])), pt2=(int(rp[0]), int(rp[1])), color=(0, 0, 255),
        #                    thickness=2)

    return img_out


def reproject_fromR(points3D, rot, tvec, camera):
    proj_2d = rot @ points3D.transpose() + tvec.reshape(3, 1)

    if camera['model'] == 'SIMPLE_RADIAL':
        f = camera['params'][0]
        cx = camera['params'][1]
        cy = camera['params'][2]
        k = camera['params'][3]

    proj_2d = proj_2d[0:2, :] / proj_2d[2, :]
    r2 = proj_2d[0, :] ** 2 + proj_2d[1, :] ** 2
    radial = r2 * k
    du = proj_2d[0, :] * radial
    dv = proj_2d[1, :] * radial

    u = proj_2d[0, :] + du
    v = proj_2d[1, :] + dv
    u = u * f + cx
    v = v * f + cy
    uvs = np.vstack([u, v]).transpose()

    return uvs


def calc_depth(points3D, rvec, tvec, camera):
    rot = sciR.from_quat(quat=[rvec[1], rvec[2], rvec[3], rvec[0]]).as_dcm()
    # print('p3d: ', points3D.shape, rot.shape, rot)
    proj_2d = rot @ points3D.transpose() + tvec.reshape(3, 1)

    return proj_2d.transpose()[:, 2]


def reproject(points3D, rvec, tvec, camera):
    '''
    Args:
        points3D: [N, 3]
        rvec: [w, x, y, z]
        tvec: [x, y, z]
    Returns:
    '''
    # print('camera: ', camera)
    # print('p3d: ', points3D.shape)
    rot = sciR.from_quat(quat=[rvec[1], rvec[2], rvec[3], rvec[0]]).as_dcm()
    # print('p3d: ', points3D.shape, rot.shape, rot)
    proj_2d = rot @ points3D.transpose() + tvec.reshape(3, 1)

    if camera['model'] == 'SIMPLE_RADIAL':
        fx = camera['params'][0]
        fy = camera['params'][0]
        cx = camera['params'][1]
        cy = camera['params'][2]
        k = camera['params'][3]

        proj_2d = proj_2d[0:2, :] / proj_2d[2, :]
        r2 = proj_2d[0, :] ** 2 + proj_2d[1, :] ** 2
        radial = r2 * k
        du = proj_2d[0, :] * radial
        dv = proj_2d[1, :] * radial

        u = proj_2d[0, :] + du
        v = proj_2d[1, :] + dv
        u = u * fx + cx
        v = v * fy + cy

        uvs = np.vstack([u, v]).transpose()


    elif camera['model'] == 'OPENCV':
        fx = camera['params'][0]
        fy = camera['params'][1]
        cx = camera['params'][2]
        cy = camera['params'][3]
        distC = camera['params'][4:]
        K = np.eye(3, dtype=float)
        K[0, 0] = fx
        K[1, 1] = fy
        K[0, 2] = cx
        K[1, 2] = cy
        uvs, b, = cv2.projectPoints(objectPoints=proj_2d, rvec=np.zeros((3,)), tvec=np.zeros((3,)), cameraMatrix=K,
                                    distCoeffs=distC)
        # print(uvs.shape, b.shape, proj_2d.shape)
        # print(uvs)
        # uvs = np.vstack(uvs).transpose()
        uvs = uvs[:, 0, :]
    return uvs


def quaternion_angular_error(q1, q2):
    """
    angular error between two quaternions
    :param q1: (4, )
    :param q2: (4, )
    :return:
    """
    d = abs(np.dot(q1, q2))
    d = min(1.0, max(-1.0, d))
    theta = 2 * np.arccos(d) * 180 / np.pi
    return theta


def ColmapQ2R(qvec):
    rot = sciR.from_quat(quat=[qvec[1], qvec[2], qvec[3], qvec[0]]).as_dcm()
    return rot


def compute_pose_error(pred_qcw, pred_tcw, gt_qcw, gt_tcw):
    pred_Rcw = sciR.from_quat(quat=[pred_qcw[1], pred_qcw[2], pred_qcw[3], pred_qcw[0]]).as_dcm()
    pred_tcw = np.array(pred_tcw, float).reshape(3, 1)
    pred_Rwc = pred_Rcw.transpose()
    pred_twc = -pred_Rcw.transpose() @ pred_tcw

    gt_Rcw = sciR.from_quat(quat=[gt_qcw[1], gt_qcw[2], gt_qcw[3], gt_qcw[0]]).as_dcm()
    gt_tcw = np.array(gt_tcw, float).reshape(3, 1)
    gt_Rwc = gt_Rcw.transpose()
    gt_twc = -gt_Rcw.transpose() @ gt_tcw

    t_error_xyz = pred_twc - gt_twc
    t_error = np.sqrt(np.sum(t_error_xyz ** 2))

    pred_qwc = sciR.from_quat(quat=[pred_qcw[1], pred_qcw[2], pred_qcw[3], pred_qcw[0]]).as_quat()
    gt_qwc = sciR.from_quat(quat=[gt_qcw[1], gt_qcw[2], gt_qcw[3], gt_qcw[0]]).as_quat()

    q_error = quaternion_angular_error(q1=pred_qwc, q2=gt_qwc)

    return q_error, t_error, (t_error_xyz[0, 0], t_error_xyz[1, 0], t_error_xyz[2, 0])
