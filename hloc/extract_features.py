import argparse
import torch
from pathlib import Path
import h5py
import logging
from types import SimpleNamespace
import cv2
import numpy as np
from tqdm import tqdm
import pprint
import os
import os.path as osp
import torch.utils.data as Data

from . import extractors
from .utils.base_model import dynamic_load
from .utils.tools import map_tensor

'''
A set of standard configurations that can be directly selected from the command
line using their name. Each is a dictionary with the following entries:
    - output: the name of the feature file that will be generated.
    - model: the model configuration, as passed to a feature extractor.
    - preprocessing: how to preprocess the images read from disk.
'''
confs = {
    # 'superpoint_aachen': {
    'superpoint-n4096-r1024-semantic': {
        'output': 'feats-superpoint-n4096-r1024-semantic',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
        'semantic': True,
    },

    'superpoint-n4096-r1024': {
        'output': 'feats-superpoint-n4096-r1024',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'superpoint-n4096-r1600': {
        'output': 'feats-superpoint-n4096-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    'superpoint-n3000-r1600': {
        'output': 'feats-superpoint-n3000-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 3000,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'superpoint-n2000-r1600': {
        'output': 'feats-superpoint-n2000-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 2000,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'superpoint-n1000-r1600': {
        'output': 'feats-superpoint-n1000-r1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 1000,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    # Resize images to 1600px even if they are originally smaller.
    # Improves the keypoint localization if the images are of good quality.
    'superpoint-n4096-rmax1600': {
        'output': 'feats-superpoint-n4096-rmax1600',
        'model': {
            'name': 'superpoint',
            'nms_radius': 3,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    # 'superpoint_inloc': {
    'superpoint-inloc': {
        'output': 'feats-superpoint-inloc',
        'model': {
            'name': 'superpoint',
            'nms_radius': 4,
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
        },
    },
    'd2net-ss': {
        'output': 'feats-d2net-ss',
        'model': {
            'name': 'd2net',
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'd2net-ss-n4096-r1024': {
        'output': 'feats-d2net-ss-n4096-r1024',
        'model': {
            'name': 'd2net',
            'max_keypoints': 4096,
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },

    'd2net-ss-n4096-r1600': {
        'output': 'feats-d2net-ss-n4096-r1600',
        'model': {
            'name': 'd2net',
            'max_keypoints': 4096,
            'multiscale': False,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'd2net-ms-n4096-r1024': {
        'output': 'feats-d2net-ms-n4096-r1024',
        'model': {
            'name': 'd2net',
            'max_keypoints': 4096,
            'multiscale': True,
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },

    'r2d2-ss-n4096-r1600': {
        'output': 'feats-r2d2-ss-n4096-r1600',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 4096,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': False,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'r2d2-ms-n4096-r1600': {
        'output': 'feats-r2d2-ms-n4096-r1600',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 4096,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': True,

        },
    },

    'r2d2-ms-n3000-r1600': {
        'output': 'feats-r2d2-ms-n3000-r1600',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 3000,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': True,

        },
    },

    'r2d2-ms-n2000-r1600': {
        'output': 'feats-r2d2-ms-n2000-r1600',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 2000,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': True,

        },
    },

    'r2d2-ms-n1000-r1600': {
        'output': 'feats-r2d2-ms-n1000-r1600',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 1000,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': True,

        },
    },

    'r2d2-rmax1600-10k': {
        'output': 'feats-r2d2-rmax1600-10k',
        'model': {
            'name': 'r2d2',
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'max_keypoints': 10000,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': True,
        },
    },

    'sift-ms-n4096-r1024': {
        'output': 'feats-sift-ms-n4096-r1024',
        'model': {
            'name': 'sift',
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1024,
        },
    },
    'sift-ms-n4096-r1600': {
        'output': 'feats-sift-ms-n4096-r1600',
        'model': {
            'name': 'sift',
            'max_keypoints': 4096,
        },
        'preprocessing': {
            'grayscale': True,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    'caps-sift-ms-n4096-r1024': {
        'output': 'feats-caps-sift-ms-n4096-r1024',
        'model': {
            'name': 'caps',
            'max_keypoints': 4096,
            'model_fn': osp.join(os.getcwd(), 'models/caps-pretrained.pth'),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'caps-sift-ms-n4096-r1600': {
        'output': 'feats-caps-sift-ms-n4096-r1600',
        'model': {
            'name': 'caps',
            'max_keypoints': 4096,
            'model_fn': osp.join(os.getcwd(), 'models/caps-pretrained.pth'),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
            'resize_force': True,
        },
    },
    'r2d2-ms-n4096-r1024': {
        'output': 'feats-r2d2-ms-n4096-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 4096,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },

    'r2d2-ms-n2048-r1024': {
        'output': 'feats-r2d2-ms-n2048-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 2048,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },

    'r2d2-ms-n1024-r1024': {
        'output': 'feats-r2d2-ms-n1024-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 1024,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },

    'r2d2-ms-n4096-r1024-d64': {
        'output': 'feats-r2d2-ms-n4096-r1024-d64',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 4096,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_B64_D6_29.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'r2d2-ms-n4096-r1024-d32': {
        'output': 'feats-r2d2-ms-n4096-r1024-d32',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 4096,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_B32_D6_29.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'r2d2-ss-n4096-r1024': {
        'output': 'feats-r2d2-ms-n4096-r1024',
        'model': {
            'name': 'r2d2',
            'max_keypoints': 4096,
            'rel_th': 0.7,
            'rep_th': 0.7,
            'multiscale': False,
            'model_fn': osp.join(os.getcwd(), "models/r2d2_WASF_N16.pt"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'sgd2-sdss-ms-n4096-r1600': {
        'output': 'feats-sgd2-sdss-wasf-0.05-n4096-r1600',
        'model': {
            'name': 'sgd2',
            'max_keypoints': 4096,
            'conf_th': 0.05,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/20210326_sdss_wtr.pth"),
            # 'model_fn': osp.join(os.getcwd(), "models/20210322_sdss.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'sgd2-sdss-ss-n4096-r1600': {
        'output': 'feats-sgd2-sdss-ss-n4096-r1600',
        'model': {
            'name': 'sgd2',
            'max_keypoints': 4096,
            'conf_th': 0.05,
            'multiscale': False,
            'model_fn': osp.join(os.getcwd(), "models/20210326_sdss_wtr.pth"),
            # 'model_fn': osp.join(os.getcwd(), "models/20210322_sdss.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1600,
        },
    },
    'sgd2-sdss-ms-n4096-r1024-0005': {
        'output': 'feats-sgd2-sdss-ms-n4096-r1024-0005',
        'model': {
            'name': 'sgd2',
            'max_keypoints': 4096,
            'conf_th': 0.005,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/20210326_sdss_wtr.pth"),
            # 'model_fn': osp.join(os.getcwd(), "models/20210322_sdss.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'sgd2-wtr-sdss-ms-n4096-r1024-0001': {
        'output': 'feats-sgd2-wtr-sdss-ms-n4096-r1024-0001',
        'model': {
            'name': 'sgd2',
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/20210326_sdss_wtr.pth"),
            # 'model_fn': osp.join(os.getcwd(), "models/20210322_sdss.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'sgd2-sdss-ss-n4096-r1024': {
        'output': 'feats-sgd2-sdss-ss-n4096-r1024',
        'model': {
            'name': 'sgd2',
            'max_keypoints': 4096,
            'conf_th': 0.05,
            'multiscale': False,
            'model_fn': osp.join(os.getcwd(), "models/20210326_sdss_wtr.pth"),
            # 'model_fn': osp.join(os.getcwd(), "models/20210322_sdss.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },
    'sgd2nb-20210719003340-sdsssfiu-ms-n4096-r1024-0001': {
        'output': 'feats-sgd2nb-20210719003340-sdsssfiu-ms-n4096-r1024-0001',
        'model': {
            'name': 'sgd2',
            'max_keypoints': 4096,
            'conf_th': 0.001,
            'multiscale': True,
            'model_fn': osp.join(os.getcwd(), "models/20210719003340_L2SegNetNB_ap1_sd_ss_sf_iu.pth"),
        },
        'preprocessing': {
            'grayscale': False,
            'resize_max': 1024,
        },
    },

    'dir': {
        'output': 'global-feats-dir',
        'model': {
            'name': 'dir',
        },
        'preprocessing': {
            'resize_max': None,
        },
    },
}


# class ImageDataset(torch.utils.data.Dataset):
class ImageDataset(Data.Dataset):
    default_conf = {
        'globs': ['*.jpg', '*.png', '*.jpeg', '*.JPG', '*.PNG'],
        'grayscale': False,
        'resize_max': None,
        'resize_force': False,
        # 'semantic': False,
    }

    def __init__(self, root, conf):
        self.conf = conf = SimpleNamespace(**{**self.default_conf, **conf})
        self.root = root

        self.paths = []
        for g in conf.globs:
            self.paths += list(Path(root).glob('**/' + g))
        if len(self.paths) == 0:
            raise ValueError(f'Could not find any image in root: {root}.')
        self.paths = [i.relative_to(root) for i in self.paths]
        logging.info(f'Found {len(self.paths)} images in root {root}.')

        # self.use_sem = conf['semantic']

    def __getitem__(self, idx):
        path = self.paths[idx]
        if self.conf.grayscale:
            mode = cv2.IMREAD_GRAYSCALE
        else:
            mode = cv2.IMREAD_COLOR
        image = cv2.imread(str(self.root / path), mode)
        if not self.conf.grayscale:
            image = image[:, :, ::-1]  # BGR to RGB
        if image is None:
            raise ValueError(f'Cannot read image {str(path)}.')
        image = image.astype(np.float32)
        size = image.shape[:2][::-1]
        w, h = size

        if self.conf.resize_max and (self.conf.resize_force
                                     or max(w, h) > self.conf.resize_max):
            scale = self.conf.resize_max / max(h, w)
            h_new, w_new = int(round(h * scale)), int(round(w * scale))
            image = cv2.resize(
                image, (w_new, h_new), interpolation=cv2.INTER_LINEAR)

        if self.conf.grayscale:
            image = image[None]
        else:
            image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
        image = image / 255.

        data = {
            'name': str(path),
            'image': image,
            'original_size': np.array(size),
            'use_sem': True,  # for rebuttal
        }
        return data

    def __len__(self):
        return len(self.paths)


@torch.no_grad()
def main(conf, image_dir, export_dir, as_half=False):
    logging.info('Extracting local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    Model = dynamic_load(extractors, conf['model']['name'])
    model = Model(conf['model']).eval().to(device)

    loader = ImageDataset(image_dir, conf['preprocessing'])
    loader = torch.utils.data.DataLoader(loader, num_workers=4)

    feature_path = Path(export_dir, conf['output'] + '.h5')

    # Do not extract again
    # if os.path.exists(feature_path):
    #     logging.info('Feature exists...')
    #     return feature_path

    feature_path.parent.mkdir(exist_ok=True, parents=True)
    feature_file = h5py.File(str(feature_path), 'a')

    for data in tqdm(loader):
        if data['name'][0] in feature_file:
            continue

        pred = model(map_tensor(data, lambda x: x.to(device)))
        pred = {k: v[0].cpu().numpy() for k, v in pred.items()}

        pred['image_size'] = original_size = data['original_size'][0].numpy()
        if 'keypoints' in pred:
            size = np.array(data['image'].shape[-2:][::-1])
            scales = (original_size / size).astype(np.float32)
            pred['keypoints'] = (pred['keypoints'] + .5) * scales[None] - .5

        if as_half:
            for k in pred:
                dt = pred[k].dtype
                if (dt == np.float32) and (dt != np.float16):
                    pred[k] = pred[k].astype(np.float16)

        grp = feature_file.create_group(data['name'][0])
        for k, v in pred.items():
            grp.create_dataset(k, data=v)

        del pred

    feature_file.close()
    logging.info('Finished exporting features.')

    return feature_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=Path, required=True)
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superpoint_aachen',
                        choices=list(confs.keys()))
    parser.add_argument('--as_half', action='store_true')
    args = parser.parse_args()
    main(confs[args.conf], args.image_dir, args.export_dir, args.as_half)
