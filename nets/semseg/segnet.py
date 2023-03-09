# -*- coding: utf-8 -*-
"""
@Time ： 2021/2/7 下午2:02
@Auth ： Fei Xue
@File ： segnet.py
@Email： xuefei@sensetime.com
"""
import os
import os.path as osp
from mmseg.apis import inference_segmentor, init_segmentor


# build the model from a config file and a checkpoint file

class SegNet:
    def __init__(self, model_name="deeplabv3plus", device='cuda:0'):
        # abs_path = "/home/Code/Deeplearning/r2d2/nets/semseg"
        seg_path = os.getcwd()
        if model_name == "deeplabv3":
            config_file = 'nets/semseg/configs/deeplabv3/deeplabv3_r50-d8_512x512_80k_ade20k.py'
            checkpoint_file = 'nets/semseg/checkpoints/deeplabv3_r50-d8_512x512_80k_ade20k_20200614_185028-0bb3f844.pth'
        elif model_name == "deeplabv3plus":
            config_file = 'nets/semseg/configs/deeplabv3plus/deeplabv3plus_r50-d8_512x512_160k_ade20k.py'
            # checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_80k_ade20k_20200614_185028-bf1400d8.pth'
            checkpoint_file = 'nets/semseg/checkpoints/deeplabv3plus_r50-d8_512x512_160k_ade20k_20200615_124504-6135c7e0.pth'

        elif model_name == 'convxts-base-ade20k':
            config_file = 'nets/semseg/configs/convnext/upernet_convnext_base_fp16_512x512_160k_ade20k.py'
            checkpoint_file = '/scratches/flyer_2/fx221/Research/Code/thirdparty/mmsegmentation/checkpoints/upernet_convnext_base_fp16_512x512_160k_ade20k_20220227_181227-02a24fc6.pth'

        self.model = init_segmentor(osp.join(seg_path, config_file), osp.join(seg_path, checkpoint_file), device=device)

    def evaluate(self, img):
        result = inference_segmentor(model=self.model, img=img)
        return result

