# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   shloc -> matchers
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   2021-05-13 10:48
=================================================='''
import argparse
import torch
import numpy as np
from pathlib import Path
import h5py
import logging
from tqdm import tqdm
import pprint
import os
import os.path as osp


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))


confs = {
    'superglue': {
        'output': 'matches-superglue',
        'model': {
            'name': 'superglue',
            'weights': 'outdoor',
            'sinkhorn_iterations': 50,
        },
    },
    'NNM': {
        'output': 'NNM',
        'model': {
            'name': 'nnm',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },
    'NNML': {
        'output': 'NNML',
        'model': {
            'name': 'nnml',
            'do_mutual_check': True,
            'distance_threshold': None,
        },
    },

    'ONN': {
        'output': 'ONN',
        'model': {
            'name': 'nn',
            'do_mutual_check': False,
            'distance_threshold': None,
        },
    },
    'NNR': {
        'output': 'NNR',
        'model': {
            'name': 'nnr',
            'do_mutual_check': True,
            'distance_threshold': 0.9,
        },
    },
    'superflue': {
        'output': 'superglue',
        'model': {
            'name': 'superglue',
            'model_fn': osp.join(os.getcwd(),
                                 "models/superglue_outdoor.pth"),
            'sinkhorn_iterations': 100,
            'match_threshold': 0.2,
        },
        'descriptor_dim': 256,
        'weights': 'indoor',
        'keypoint_encoder': [32, 64, 128, 256],
        'GNN_layers': ['self', 'cross'] * 9,
        'sinkhorn_iterations': 20,
        'match_threshold': 0.2,
    }
}


class Matcher(torch.nn.Module):
    def __init__(self, conf):
        super(Matcher, self).__init__()
        self.conf = conf
        self.mode = conf['model']['name']

    def forward(self, data):
        # print(data.keys())
        desc1 = torch.from_numpy(data['descriptors0']).cuda()
        desc2 = torch.from_numpy(data['descriptors1']).cuda()

        # print(desc1.shape, desc2.shape)
        if self.mode == "nnm":
            matches = self.mutual_nn_matcher(descriptors1=desc1,
                                             descriptors2=desc2)
        elif self.mode == "nnr":
            matches = self.mutual_nn_ratio_matcher(descriptors1=desc1,
                                                   descriptors2=desc2,
                                                   ratio=self.conf['model']['distance_threshold'])
        elif self.mode == "nnml":
            matches = self.matcher_with_label(descriptors1=desc1,
                                              labels1=data['labels0'],
                                              descriptors2=desc2,
                                              labels2=data['labels1'],
                                              )

        all_matches = np.ones((desc1.shape[0],), dtype=int) * -1
        # print("desc1/2", desc1.shape, desc2.shape)
        scores = torch.topk(desc1 @ desc2.t(), dim=1, k=1)[0]
        for i in range(matches.shape[0]):
            all_matches[matches[i, 0]] = matches[i, 1]
        return {
            'matches0': all_matches,
            'matching_scores0': scores.squeeze().cpu().numpy(),
        }

    # Mutual nearest neighbors matcher for L2 normalized descriptors.
    def mutual_nn_matcher(self, descriptors1, descriptors2):
        device = descriptors1.device
        sim = descriptors1 @ descriptors2.t()
        nn12 = torch.max(sim, dim=1)[1]
        nn21 = torch.max(sim, dim=0)[1]
        ids1 = torch.arange(0, sim.shape[0], device=device)
        mask = ids1 == nn21[nn12]
        matches = torch.stack([ids1[mask], nn12[mask]]).t()
        return matches.data.cpu().numpy()

    # Symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    def ratio_matcher(self, descriptors1, descriptors2, ratio=0.9):
        device = descriptors1.device
        sim = descriptors1 @ descriptors2.t()
        # print("sim: ", sim.shape)

        # Retrieve top 2 nearest neighbors 1->2.
        nns_sim, nns = torch.topk(sim, 2, dim=1)
        nns_dist = torch.sqrt(2 - 2 * nns_sim)
        # Compute Lowe's ratio.
        ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
        # Save first NN.
        nn12 = nns[:, 0]

        # Retrieve top 2 nearest neighbors 1->2.
        nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
        nns_dist = torch.sqrt(2 - 2 * nns_sim)
        # Compute Lowe's ratio.
        ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
        # Save first NN.
        nn21 = nns[:, 0]

        # Symmetric ratio test.
        ids1 = torch.arange(0, sim.shape[0], device=device)
        mask = torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio)

        # Final matches.
        matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

        # return matches
        return matches.data.cpu().numpy()

    # Mutual NN + symmetric Lowe's ratio test matcher for L2 normalized descriptors.
    def mutual_nn_ratio_matcher(self, descriptors1, descriptors2, ratio=0.9):
        device = descriptors1.device
        sim = descriptors1 @ descriptors2.t()

        # Retrieve top 2 nearest neighbors 1->2.
        nns_sim, nns = torch.topk(sim, 2, dim=1)
        nns_dist = torch.sqrt(2 - 2 * nns_sim)
        # Compute Lowe's ratio.
        ratios12 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
        # Save first NN and match similarity.
        nn12 = nns[:, 0]

        # Retrieve top 2 nearest neighbors 1->2.
        nns_sim, nns = torch.topk(sim.t(), 2, dim=1)
        nns_dist = torch.sqrt(2 - 2 * nns_sim)
        # Compute Lowe's ratio.
        ratios21 = nns_dist[:, 0] / (nns_dist[:, 1] + 1e-8)
        # Save first NN.
        nn21 = nns[:, 0]

        # Mutual NN + symmetric ratio test.
        ids1 = torch.arange(0, sim.shape[0], device=device)
        mask = torch.min(ids1 == nn21[nn12], torch.min(ratios12 <= ratio, ratios21[nn12] <= ratio))

        # Final matches.
        matches = torch.stack([ids1[mask], nn12[mask]], dim=-1)

        # return matches

        return matches.data.cpu().numpy()

    def matcher_cross_label(self, pts1, descs1, pts2, descs2, labels1, labels2, global_ids=None):
        def sel_pts_with_labels(pts, descs, labels):
            sel_pts = []
            sel_descs = []
            nums = [0]
            n = 0
            for gid in global_ids:
                if gid == 0:
                    continue
                for idx, p in enumerate(pts):
                    id = labels[int(p[1]), int(p[0])]
                    if gid == id:
                        sel_pts.append(p)
                        sel_descs.append(descs[idx])
                        n += 1
                nums.append(n)
            return np.array(sel_pts, pts.dtype), np.array(sel_descs, descs.dtype), nums

        if global_ids is None:
            global_ids = np.unique(labels1).tolist()

        sel_pts1, sel_descs1, nums1 = sel_pts_with_labels(pts=pts1, descs=descs1, labels=labels1)
        sel_pts2, sel_descs2, nums2 = sel_pts_with_labels(pts=pts2, descs=descs2, labels=labels2)

        # print(nums1)
        # print(nums2)

        matches = []
        for n in range(1, len(nums1)):
            s1, e1 = nums1[n - 1], nums1[n]
            s2, e2 = nums2[n - 1], nums2[n]
            if s1 == e1 or s2 == e2:
                continue
            # n_matches = mutual_nn_matcher(descriptors1=torch.FloatTensor(sel_descs1[s1:e1]).cuda(),
            #                               descriptors2=torch.FloatTensor(sel_descs2[s2:e2]).cuda())
            n_matches = self.ratio_matcher(descriptors1=torch.FloatTensor(sel_descs1[s1:e1]).cuda(),
                                           descriptors2=torch.FloatTensor(sel_descs2[s2:e2]).cuda())
            for m in n_matches:
                matches.append([m[0] + s1, m[1] + s2])

        matches = np.array(matches, np.int)
        return sel_pts1, sel_descs1, sel_pts2, sel_descs2, matches

    def matcher_with_label(self, descriptors1, labels1, descriptors2, labels2, matcher='mnn'):
        uids1 = np.unique(labels1).tolist()
        uids2 = np.unique(labels2).tolist()
        uids = [v for v in uids1 if v in uids2]
        valid_uids = [v for v in uids if v > 0]

        all_matches = []
        # matching between features with the same labels
        matched_ids1 = []
        matched_ids2 = []
        for id in valid_uids:
            idx1 = np.where(labels1 == id)[0]
            idx2 = np.where(labels2 == id)[0]
            # print(id)
            # print(idx1.shape)
            # print(idx2.shape)
            desc1 = descriptors1[idx1]
            desc2 = descriptors2[idx2]
            matches = self.mutual_nn_matcher(descriptors1=desc1, descriptors2=desc2)
            # print(matches.shape)
            for i in range(matches.shape[0]):
                # print("m: ", matches[i])
                all_matches.append(([idx1[matches[i, 0]], idx2[matches[i, 1]]]))
                matched_ids1.append(idx1[matches[i, 0]])
                matched_ids2.append(idx2[matches[i, 1]])

        # matching between features with unmatched labels
        desc1 = []
        desc2 = []
        idx1 = []
        idx2 = []
        for i in range(descriptors1.shape[0]):
            # l = labels1[i]
            # if l in valid_uids:
            if i in matched_ids1:
                continue
            desc1.append(descriptors1[i])
            idx1.append(i)
        for i in range(descriptors2.shape[0]):
            # l = labels2[i]
            # if l in valid_uids:
            if i in matched_ids2:
                continue
            desc2.append(descriptors2[i])
            idx2.append(i)

        if len(desc1) == 0 or len(desc2) == 0:
            all_matches = np.array(all_matches, np.int)
            return all_matches
        desc1 = torch.stack(desc1)
        # print(descriptors1.shape, desc1.shape)
        desc2 = torch.stack(desc2)
        matches = self.mutual_nn_matcher(descriptors1=desc1, descriptors2=desc2)
        for i in range(matches.shape[0]):
            all_matches.append((idx1[matches[i, 0]], idx2[matches[i, 1]]))

        # return torch.stack(all_matches)
        all_matches = np.array(all_matches, np.int)
        return all_matches


@torch.no_grad()
def main(conf, pairs, features, export_dir):
    logging.info('Matching local features with configuration:'
                 f'\n{pprint.pformat(conf)}')

    feature_path = Path(export_dir, features + '.h5')
    assert feature_path.exists(), feature_path
    feature_file = h5py.File(str(feature_path), 'r')
    pairs_name = pairs.stem
    assert pairs.exists(), pairs
    with open(pairs, 'r') as f:
        pair_list = f.read().rstrip('\n').split('\n')

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    if conf['model']['name'] == 'superglue':
        model = SuperGlue(config=conf).cuda()
        # model.load_state_dict(torch.load(conf['model']['model_fn']))
        model.eval()
    else:
        model = Matcher(conf=conf).cuda()
        model.eval()

    match_name = f'{features}-{conf["output"]}-{pairs_name}'
    match_path = Path(export_dir, match_name + '.h5')

    # if os.path.exists(match_path):
    #     logging.info('Matching file exists.')
    #     return match_path
    match_file = h5py.File(str(match_path), 'a')

    matched = set()
    for pair in tqdm(pair_list, smoothing=.1):
        name0, name1 = pair.split(' ')
        pair = names_to_pair(name0, name1)

        # Avoid to recompute duplicates to save time
        if len({(name0, name1), (name1, name0)} & matched) \
                or pair in match_file:
            continue

        data = {}
        feats0, feats1 = feature_file[name0], feature_file[name1]

        if conf['model']['name'] == 'superglue':
            for k, v in feats1.items():
                data[k + '0'] = torch.from_numpy(v.__array__()).float().to(device)
                data['image0'] = torch.empty((1,) + tuple(feats0['image_size'])[::-1])
            for k, v in feats1.items():
                data[k + '1'] = torch.from_numpy(v.__array__()).float().to(device)
                data['image1'] = torch.empty((1,) + tuple(feats1['image_size'])[::-1])
            data = {k: v[None] for k, v in data.items()}
            pred = model(data)
            matches = pred['matches0'][0].cpu().short().numpy()

            if 'matching_scores0' in pred:
                scores = pred['matching_scores0'][0].cpu().half().numpy()

        else:
            for k in feats1.keys():
                data[k + '0'] = feats0[k].__array__().transpose()  # [N, 128]
            for k in feats1.keys():
                data[k + '1'] = feats1[k].__array__().transpose()  # [N, 128]
            # data = {k: torch.from_numpy(v)[None].float().to(device)
            #         for k, v in data.items()}

            # some matchers might expect an image but only use its size
            data['image0'] = torch.empty((1, 1,) + tuple(feats0['image_size'])[::-1])
            data['image1'] = torch.empty((1, 1,) + tuple(feats1['image_size'])[::-1])

            pred = model(data)
            matches = pred['matches0']  # [0].cpu().short().numpy()
            if 'matching_scores0' in pred:
                scores = pred['matching_scores0']

        grp = match_file.create_group(pair)
        # print("matches: ", matches.shape)
        grp.create_dataset('matches0', data=matches)
        # print("matches: ", matches)

        if 'matching_scores0' in pred:
            grp.create_dataset('matching_scores0', data=scores)
            # print("scores: ", scores.shape)

        matched |= {(name0, name1), (name1, name0)}

        # plot matches
        """
        img1 = cv2.imread(osp.join(img_dir, name0))
        img2 = cv2.imread(osp.join(img_dir, name1))
        valid_matches = []
        for i in range(matches.shape[0]):
            if matches[i] > 0:
                valid_matches.append([i, matches[i]])
        valid_matches = np.array(valid_matches, np.int)
        img_matches = plot_matches(img1=img1, img2=img2,
                                   pts1=data["keypoints0"],
                                   pts2=data["keypoints1"],
                                   matches=valid_matches,
                                   )
        cv2.imshow("match", img_matches)
        cv2.waitKey(0)
        """

    match_file.close()
    logging.info('Finished exporting matches.')

    return match_path


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--export_dir', type=Path, required=True)
    parser.add_argument('--features', type=str,
                        default='feats-superpoint-n4096-r1024')
    parser.add_argument('--pairs', type=Path, required=True)
    parser.add_argument('--conf', type=str, default='superglue',
                        choices=list(confs.keys()))
    args = parser.parse_args()

    img_dir = "/home/mifs/fx221/fx221/localization/aachen_v1_1/images/images_upright"
    main(confs[args.conf], args.pairs, args.features, args.export_dir)
