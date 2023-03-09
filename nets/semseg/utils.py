# -*- coding: utf-8 -*-
# @Time    : 2020/7/24 下午3:14
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : utils.py

import torch
import numpy as np
import csv
import matplotlib.pyplot as plt
from tools.viz import show_flow
import os


def get_semantic_dict():
    """
    merge semantics to coarse categories e.g. [invalid, stable, short-term, dynamic]
    """
    # names = {}
    maps = {}  # "/home/xuefei/Data/fei/Research/Code/deeplearning/feature/r2d2
    with open(os.path.join(os.getcwd(), "nets/semseg/object150_info_ext.csv"), "r") as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            # names[int(row[0])] = row[5].split(";")[0]
            maps[int(row[0])] = int(row[-1])

    return maps


def get_conf_dict():
    """
    map semantic_dict to confidence
    """
    # maps = {
    #     0: 0.1,  # invalid
    #     1: 1.,  # stable
    #     2: 0.1,  # dynamic
    #     3: 0.5  # short-term
    # }

    maps = {
        0: 0.1,  # invalid
        1: 1.,  # stable
        2: 0.1,  # dynamic
        3: 0.5  # short-term
    }

    return maps


def semantic_to_stability(seg_map, semantic_dict):
    out = torch.zeros_like(seg_map).float()
    # step1: semantics to categories
    for key in semantic_dict.keys():
        out[seg_map == key] = semantic_dict[key]

    return out


def semantic_to_stability_np(seg_map, semantic_dict):
    out = np.zeros_like(seg_map)
    # step1: semantics to categories
    for key in semantic_dict.keys():
        out[seg_map == key] = semantic_dict[key]

    return out


def segmantic_to_confidence(seg_map, conf_dict, semantic_dict):
    """
    convert semantic map to confidence for other usage
    """

    out = torch.zeros_like(seg_map).float()

    # step1: semantics to categories
    for key in semantic_dict.keys():
        out[seg_map == key] = semantic_dict[key]

    # step2: categories to confidence
    for key in conf_dict.keys():
        out[out == key] = conf_dict[key]
    return out


def segmantic_to_confidence_np(seg_map, conf_dict, semantic_dict):
    """
    convert semantic map to confidence for other usage
    """

    out = np.zeros_like(seg_map)

    # step1: semantics to categories
    for key in semantic_dict.keys():
        out[seg_map == key] = semantic_dict[key]

    # step2: categories to confidence
    for key in conf_dict.keys():
        out[out == key] = conf_dict[key]
    return out


def confidence_to_rgb(conf_map,
                      colors=np.array([[0, 0, 0],
                                       [0, 255, 0],
                                       [255, 0, 0],
                                       [0, 0, 255]], np.uint8)):
    conf_color = np.zeros(shape=(conf_map.shape[0], conf_map.shape[1], 3), dtype=np.uint8)

    for i in range(conf_map.shape[0]):
        for j in range(conf_map.shape[1]):
            conf_color[i, j] = colors[conf_map[i, j]]
    return conf_color


def show_seg_pair(seg_map1, seg_map2, aflow):
    print("show_seg_pair-seg_map1: ", seg_map1.shape)
    print("show_seg_pair-seg_map2: ", seg_map2.shape)
    print("show_seg_pair-aflow: ", aflow.shape)
    H, W = aflow.shape[-2:]
    flow = (aflow - np.mgrid[:H, :W][::-1]).transpose(1, 2, 0)

    semantic_dict = get_semantic_dict()
    seg_stb1 = semantic_to_stability_np(seg_map=seg_map1, semantic_dict=semantic_dict)
    seg_stb2 = semantic_to_stability_np(seg_map=seg_map2, semantic_dict=semantic_dict)

    seg_conf_rgb1 = confidence_to_rgb(conf_map=seg_stb1)
    seg_conf_rgb2 = confidence_to_rgb(conf_map=seg_stb2)

    show_flow(img0=seg_conf_rgb1, img1=seg_conf_rgb2, flow=flow)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import cv2

    conf_dict = get_conf_dict()
    semantic_dict = get_semantic_dict()

    print("conf_dict: ", conf_dict)
    print("semantic_dict: ", semantic_dict)

    seg_map = np.random.randint(low=0, high=3, size=(512, 512))
    # seg_map = torch.from_numpy(seg_map).int()
    # conf = segmantic_to_confidence(seg_map=seg_map, conf_dict=conf_dict, semantic_dict=semantic_dict)
    conf_map = confidence_to_rgb(conf_map=seg_map)
    # conf_map = np.transpose(conf_map, (2, 0, 1))
    print("conf_map: ", conf_map.shape)

    cv2.imshow("conf_map", conf_map)
    cv2.waitKey(0)
    # print("conf: ", conf)
