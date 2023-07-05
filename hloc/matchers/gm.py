# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   r2d2 -> gm
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   25/05/2023 10:09
=================================================='''
import torch
import sys
from pathlib import Path
from hloc.utils.base_model import BaseModel
from nets.gm import GM as GMatcher


class GM(BaseModel):
    default_config = {
        'descriptor_dim': 256,
        'hidden_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,  # [self, cross, self, cross, ...] 9 in total
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
        'with_pose': False,
        'n_layers': 9,
        'n_min_tokens': 256,
        'with_sinkhorn': True,

        'ac_fn': 'relu',
        'norm_fn': 'bn',
        'weight_path': None,
    }

    required_inputs = [
        'image0', 'keypoints0', 'scores0', 'descriptors0',
        'image1', 'keypoints1', 'scores1', 'descriptors1',
    ]

    def _init(self, conf):
        self.net = GMatcher(config=conf).eval()
        state_dict = torch.load(conf['weight_path'], map_location='cpu')['model']
        self.net.load_state_dict(state_dict, strict=True)

    def _forward(self, data):
        with torch.no_grad():
            return self.net(data)
