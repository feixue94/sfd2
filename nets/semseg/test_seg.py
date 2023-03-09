# -*- coding: utf-8 -*-
# @Time    : 2020/6/18 上午10:12
# @Author  : Fei Xue
# @Email   : feixue@pku.edu.cn
# @File    : test_seg.py

import torch
# from nets.semseg import encoding
from mmseg.apis import inference_segmentor, init_segmentor
import mmcv

import os
import os.path as osp
import cv2
import numpy as np
from tqdm import tqdm
from glob import glob


def get_recursive_file_list(root_dir, sub_dir="", patterns=[]):
    current_files = os.listdir(osp.join(root_dir, sub_dir))
    all_files = []
    for file_name in current_files:
        full_file_name = os.path.join(root_dir, sub_dir, file_name)
        # print(file_name)

        if file_name.split('.')[-1] in patterns:
            all_files.append(osp.join(sub_dir, file_name))

        if os.path.isdir(full_file_name):
            next_level_files = get_recursive_file_list(root_dir, sub_dir=osp.join(sub_dir, file_name),
                                                       patterns=patterns)
            all_files.extend(next_level_files)

    return all_files


seg_method = 'DeepLab_ResNeSt50_ADE'
# model = encoding.models.get_model('EncNet_ResNet50s_ADE', pretrained=True).cuda().eval()
# model = encoding.models.get_model('EncNet_ResNet50s_ADE', pretrained=True).cuda().eval()
# model = encoding.models.get_model('encnet_resnet50s_pcontext', pretrained=True).cuda().eval()
# model = encoding.models.get_model('Encnet_ResNet50s_PContext', pretrained=True).cuda().eval()
# model = encoding.models.get_model('FCN_ResNeSt50_ADE', pretrained=True).cuda().eval()
# model = encoding.models.get_model(seg_method, pretrained=True).cuda().eval()

# torch.save(model.state_dict(), "enconet_resnet50s_ade")
# exit(0)

# model.load_state_dict(torch.load("enconet_resnet50s_ade"))

# img_dir = "/home/xuefei/fei/reloc/2016-02-12-02-10-44-122_L1_11/stereo_left"
# img_fn = "1455214480.303000_00003724_227.png"

# img_dir = "/home/xuefei/fei/hpatches_sequences/hpatches-sequences-release/i_londonbridge"
# img_dir = "/home/xuefei/fei/r2d2/data/aachen/images_upright/db"
# img_fn = "958.jpg"

from nets.semseg.segnet import SegNet


# def seg():
#     dataset_dir = "/home/xuefei/fei/localdataset/Aachen"
#     img_list_file = osp.join(dataset_dir, "imglists.txt")
#
#     save_dir = osp.join("/home/xuefei/fei/localdataset/semantic", seg_method)
#     save_dir = osp.join('/scratch2/fx221/exp/r2d2/Aachen_v11')
#     if not os.path.exists(save_dir):
#         os.mkdir(save_dir)
#     with open(img_list_file, "r") as f:
#         lines = f.readlines()
#
#     for line in tqdm(lines, total=len(lines)):
#         path = line.strip()
#         img_fn = osp.join(dataset_dir, path)
#         raw_img = cv2.imread(img_fn)
#         print(raw_img.shape)
#
#         sub_dir = path.strip().split('/')
#         seg_path = save_dir
#         for p in sub_dir[:-2]:
#             seg_path = osp.join(seg_path, p)
#         if not osp.exists(seg_path):
#             os.makedirs(seg_path)
#         seg_path = osp.join(seg_path, sub_dir[-1])
#         # print(seg_path)
#         # exit(0)
#         # raw_img = cv2.resize(raw_img, dsize=(512, 512))
#
#         img = np.transpose(raw_img, [2, 0, 1])
#         img = img.astype(np.float) / 255.
#         img = torch.from_numpy(img).float().cuda()
#         img = img.unsqueeze(0)
#         # out = model(img)
#
#         with torch.no_grad():
#             out = model.evaluate(img)
#             predict = torch.max(out, 1)[1].cpu().numpy() + 1
#
#         mask = encoding.utils.get_mask_pallete(predict, "ade20k").convert("RGB")
#         # mask = encoding.utils.get_mask_pallete(predict, "pascal_voc")
#         # print(mask)
#         # exit(0)
#         print(seg_path)
#         mask.save(seg_path)
#         mask.show(title="mask")
#         #
#         # cv2.imshow("img", raw_img)
#         # cv2.waitKey(10)

def seg_aachen():
    dataset_dir = '/scratch2/fx221/localization/aachen_v1_1/images/images_upright'
    test_imgs_db = get_recursive_file_list(root_dir=dataset_dir, sub_dir='db', patterns=['png', 'jpg'])
    test_imgs_query = get_recursive_file_list(root_dir=dataset_dir, sub_dir='query', patterns=['png', 'jpg'])
    test_imgs_seq = get_recursive_file_list(root_dir=dataset_dir, sub_dir='sequences', patterns=['png', 'jpg'])
    # print(len(test_imgs_seq), len(test_imgs_query), len(test_imgs_db))
    # test_imgs = test_imgs.extend(glob(osp.join(dataset_dir, '/*/*.jpg'), recursive=True))
    # print(test_imgs, len(test_imgs))
    test_imgs = test_imgs_db + test_imgs_seq + test_imgs_query

    save_dir_mask = osp.join('/scratch2/fx221/exp/r2d2/segmentations/Aachen_v11/mask')
    save_dir_seg = osp.join('/scratch2/fx221/exp/r2d2/segmentations/Aachen_v11')
    if not os.path.exists(save_dir_seg):
        os.mkdir(save_dir_seg)
    if not os.path.exists(save_dir_mask):
        os.mkdir(save_dir_mask)

    # config_file = 'nets/semseg/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    config_file = 'nets/semseg/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
    checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    for path in tqdm(test_imgs, total=len(test_imgs)):
        img = osp.join(dataset_dir,
                       osp.join(dataset_dir, path))  # or img = mmcv.imread(img), which will only load it once
        img = cv2.imread(img)
        img = cv2.resize(img, dsize=(512, 512))
        # print('img: ', img.shape)
        result = inference_segmentor(model, img)
        seg_mask = result[0].astype(np.uint8) + 1
        # print('result: ', seg_mask.shape, seg_mask)

        save_path_mask = osp.join(save_dir_mask, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_mask)):
            os.makedirs(osp.dirname(save_path_mask))
        cv2.imwrite(save_path_mask, seg_mask)

        # visualize the results in a new window
        save_path_seg = osp.join(save_dir_seg, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_seg)):
            os.makedirs(osp.dirname(save_path_seg))
        seg = model.show_result(img, result, out_file=save_path_seg, show=True, wait_time=1)
        # cv2.imwrite(save_path_seg, seg)
        # exit(0)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    # model.show_result(img, result, out_file='result.png', opacity=1.0)


def seg_robotcar():
    dataset_dir = '/scratch2/fx221/localization/RobotCar-Seasons/images'
    test_imgs = []
    # for cat in ['dawn', 'dusk', 'night', 'night-rain', 'overcast-summer', 'overcast-winter', 'rain', 'snow', 'sun']:
    for cat in ['overcast-reference']:
        cat_imgs = get_recursive_file_list(root_dir=dataset_dir, sub_dir=cat, patterns=['png', 'jpg'])
        test_imgs = test_imgs + cat_imgs
        print(cat, len(cat_imgs))

    # exit(0)

    save_dir_mask = osp.join('/scratch2/fx221/exp/r2d2/segmentations/RobotCar-Seasons/mask')
    save_dir_seg = osp.join('/scratch2/fx221/exp/r2d2/segmentations/RobotCar-Seasons')
    if not os.path.exists(save_dir_seg):
        os.mkdir(save_dir_seg)
    if not os.path.exists(save_dir_mask):
        os.mkdir(save_dir_mask)

    # config_file = 'nets/semseg/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    config_file = 'nets/semseg/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
    checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    for path in tqdm(test_imgs, total=len(test_imgs)):
        img = osp.join(dataset_dir,
                       osp.join(dataset_dir, path))  # or img = mmcv.imread(img), which will only load it once
        img = cv2.imread(img)
        img = cv2.resize(img, dsize=(512, 512))
        # print('img: ', img.shape)
        result = inference_segmentor(model, img)
        seg_mask = result[0].astype(np.uint8) + 1
        # print('result: ', seg_mask.shape, seg_mask)

        save_path_mask = osp.join(save_dir_mask, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_mask)):
            os.makedirs(osp.dirname(save_path_mask))
        cv2.imwrite(save_path_mask, seg_mask)

        # visualize the results in a new window
        save_path_seg = osp.join(save_dir_seg, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_seg)):
            os.makedirs(osp.dirname(save_path_seg))
        seg = model.show_result(img, result, out_file=save_path_seg, show=True, wait_time=1)
        # cv2.imwrite(save_path_seg, seg)
        # exit(0)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    # model.show_result(img, result, out_file='result.png', opacity=1.0)


def seg_inloc():
    dataset_dir = '/scratches/flyer_3/fx221/dataset/InLoc'
    test_imgs = []
    # for cat in ['dawn', 'dusk', 'night', 'night-rain', 'overcast-summer', 'overcast-winter', 'rain', 'snow', 'sun']:
    for cat in ['iphone7', 'cutouts']:
        cat_imgs = get_recursive_file_list(root_dir=dataset_dir, sub_dir=cat, patterns=['png', 'jpg', 'JPG'])
        test_imgs = test_imgs + cat_imgs
        print(cat, len(cat_imgs))

    # exit(0)

    save_dir_mask = osp.join('/scratches/flyer_3/fx221/dataset/InLoc/segmentations/mask')
    save_dir_seg = osp.join('/scratches/flyer_3/fx221/dataset/InLoc/segmentations')
    if not os.path.exists(save_dir_seg):
        os.mkdir(save_dir_seg)
    if not os.path.exists(save_dir_mask):
        os.mkdir(save_dir_mask)

    # config_file = 'nets/semseg/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    config_file = 'nets/semseg/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
    checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    for path in tqdm(test_imgs, total=len(test_imgs)):
        img = osp.join(dataset_dir,
                       osp.join(dataset_dir, path))  # or img = mmcv.imread(img), which will only load it once
        img = cv2.imread(img)
        img = cv2.resize(img, dsize=(512, 512))
        # print('img: ', img.shape)
        result = inference_segmentor(model, img)
        seg_mask = result[0].astype(np.uint8) + 1
        # print('result: ', seg_mask.shape, seg_mask)

        save_path_mask = osp.join(save_dir_mask, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_mask)):
            os.makedirs(osp.dirname(save_path_mask))
        cv2.imwrite(save_path_mask, seg_mask)

        # visualize the results in a new window
        save_path_seg = osp.join(save_dir_seg, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_seg)):
            os.makedirs(osp.dirname(save_path_seg))
        seg = model.show_result(img, result, out_file=save_path_seg, show=True, wait_time=1)
        # seg = model.show_result(img, result, show=True, wait_time=-1)
        # cv2.imwrite(save_path_seg, seg)
        # exit(0)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    # model.show_result(img, result, out_file='result.png', opacity=1.0)


def seg_7scenes():
    dataset_dir = '/scratches/flyer_3/fx221/dataset/7scenes'
    test_imgs = []
    # for cat in ['dawn', 'dusk', 'night', 'night-rain', 'overcast-summer', 'overcast-winter', 'rain', 'snow', 'sun']:
    for cat in ['chess', 'stairs']:
        cat_imgs = get_recursive_file_list(root_dir=dataset_dir, sub_dir=cat, patterns=['png', 'jpg', 'JPG'])
        test_imgs = test_imgs + cat_imgs
        print(cat, len(cat_imgs))

    # exit(0)

    save_dir_mask = osp.join(dataset_dir, 'segmentations/mask')
    save_dir_seg = osp.join(dataset_dir, 'segmentations')
    if not os.path.exists(save_dir_seg):
        os.mkdir(save_dir_seg)
    if not os.path.exists(save_dir_mask):
        os.mkdir(save_dir_mask)

    # config_file = 'nets/semseg/configs/pspnet/pspnet_r50-d8_512x1024_40k_cityscapes.py'
    config_file = 'nets/semseg/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
    checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth'
    # build the model from a config file and a checkpoint file
    model = init_segmentor(config_file, checkpoint_file, device='cuda:0')

    # test a single image and show the results
    for path in tqdm(test_imgs, total=len(test_imgs)):
        if path.find('depth') >= 0:
            continue

        img = osp.join(dataset_dir,
                       osp.join(dataset_dir, path))  # or img = mmcv.imread(img), which will only load it once
        img = cv2.imread(img)
        img = cv2.resize(img, dsize=(512, 512))
        # print('img: ', img.shape)
        result = inference_segmentor(model, img)
        seg_mask = result[0].astype(np.uint8) + 1
        # print('result: ', seg_mask.shape, seg_mask)

        save_path_mask = osp.join(save_dir_mask, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_mask)):
            os.makedirs(osp.dirname(save_path_mask))
        cv2.imwrite(save_path_mask, seg_mask)

        # visualize the results in a new window
        save_path_seg = osp.join(save_dir_seg, path.replace('jpg', 'png'))
        if not osp.exists(osp.dirname(save_path_seg)):
            os.makedirs(osp.dirname(save_path_seg))
        seg = model.show_result(img, result, out_file=save_path_seg, show=True, wait_time=1)
        # seg = model.show_result(img, result, show=True, wait_time=-1)
        # cv2.imwrite(save_path_seg, seg)
        # exit(0)
    # or save the visualization results to image files
    # you can change the opacity of the painted segmentation map in (0, 1].
    # model.show_result(img, result, out_file='result.png', opacity=1.0)


if __name__ == '__main__':
    # seg_aachen()
    # seg_robotcar()
    # seg_inloc()
    seg_7scenes()
