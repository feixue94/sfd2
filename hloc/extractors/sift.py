# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   Hierarchical-Localization -> sift
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   21/06/2021 09:40
=================================================='''
import cv2
import numpy as np
import torch
from ..utils.base_model import BaseModel


class Sift(BaseModel):
    conf = {
        'nms_radius': 4,
        'keypoint_threshold': 0.005,
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    required_inputs = ['image']

    def _init(self, conf):
        # self.sift = cv2.xfeatures2d.SIFT_create(conf["max_keypoints"])
        # self.sift = cv2.xfeatures2d.SIFT_create()
        self.sift = cv2.xfeatures2d.SIFT_create(conf['max_keypoints'], 6, 0.03, 6, 1.6)
        # self.net.setNonmaxSuppression(conf['nms_radius'])

    def _forward(self, data):
        image = data['image']
        # print(image.shape)
        image = image.cpu().numpy()[0, 0] * 255
        image = np.uint8(image)
        keypoints, descriptors = self.sift.detectAndCompute(image, None)
        scores = np.array([kp.response for kp in keypoints], np.float32)
        keypoints = np.array([[kp.pt[0], kp.pt[1]] for kp in keypoints])

        if self.conf["max_keypoints"]:
            topK = self.conf["max_keypoints"]
            if topK < keypoints.shape[0]:
                idxs = (-scores).argsort()[:topK]
                keypoints = keypoints[idxs]
                descriptors = descriptors[idxs]
                scores = scores[idxs]
        #
        # print("feats: ", keypoints.shape, descriptors.shape, scores.shape)
        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }
