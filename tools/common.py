# Copyright 2019-present NAVER Corp.
# CC BY-NC-SA 3.0
# Available only for non-commercial use

import os, pdb  # , shutil
import numpy as np
import torch
import json
import cv2


def resize_img(img, nh=-1, nw=-1, mode=cv2.INTER_NEAREST):
    assert nh > 0 or nw > 0
    if nh == -1:
        return cv2.resize(img, dsize=(nw, int(img.shape[0] / img.shape[1] * nw)), interpolation=mode)
    if nw == -1:
        return cv2.resize(img, dsize=(int(img.shape[1] / img.shape[0] * nh), nh), interpolation=mode)
    return cv2.resize(img, dsize=(nw, nh), interpolation=mode)

def mkdir_for(file_path):
    os.makedirs(os.path.split(file_path)[0], exist_ok=True)


def model_size(model):
    ''' Computes the number of parameters of the model 
    '''
    size = 0
    for weights in model.state_dict().values():
        size += np.prod(weights.shape)
    return size


def torch_set_gpu(gpus):
    if type(gpus) is int:
        gpus = [gpus]

    cuda = all(gpu >= 0 for gpu in gpus)

    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(gpu) for gpu in gpus])
        assert cuda and torch.cuda.is_available(), "%s has GPUs %s unavailable" % (
            os.environ['HOSTNAME'], os.environ['CUDA_VISIBLE_DEVICES'])
        torch.backends.cudnn.benchmark = True  # speed-up cudnn
        torch.backends.cudnn.fastest = True  # even more speed-up?
        print('Launching on GPUs ' + os.environ['CUDA_VISIBLE_DEVICES'])

    else:
        print('Launching on CPU')

    return cuda


def save_args(args, save_path):
    with open(save_path, 'w') as f:
        json.dump(args.__dict__, f, indent=2)


def load_args(args, save_path):
    with open(save_path, "r") as f:
        args.__dict__ = json.load(f)
