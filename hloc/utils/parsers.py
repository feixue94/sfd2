from pathlib import Path
import logging
import numpy as np
from collections import defaultdict


def parse_image_lists_with_intrinsics(paths):
    results = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0

    for lfile in files:
        with open(lfile, 'r') as f:
            raw_data = f.readlines()

        logging.info(f'Importing {len(raw_data)} queries in {lfile.name}')
        for data in raw_data:
            data = data.strip('\n').split(' ')
            name, camera_model, width, height = data[:4]
            params = np.array(data[4:], float)
            info = (camera_model, int(width), int(height), params)
            results.append((name, info))

    assert len(results) > 0
    return results


def parse_img_lists_for_extended_cmu_seaons(paths):
    Ks = {
        "c0": "OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571",
        "c1": "OPENCV 1024 768 868.993378 866.063001 525.942323 420.042529 -0.399431 0.188924 0.000153 0.000571"
    }

    results = []
    files = list(Path(paths.parent).glob(paths.name))
    assert len(files) > 0

    for lfile in files:
        with open(lfile, 'r') as f:
            raw_data = f.readlines()

            logging.info(f'Importing {len(raw_data)} queries in {lfile.name}')
            for name in raw_data:
                name = name.strip('\n')
                camera = name.split('_')[2]
                K = Ks[camera].split(' ')
                camera_model, width, height = K[:3]
                params = np.array(K[3:], float)
                # print("camera: ", camera_model, width, height, params)
                info = (camera_model, int(width), int(height), params)
                results.append((name, info))

        assert len(results) > 0
        return results


def parse_retrieval(path):
    retrieval = defaultdict(list)
    with open(path, 'r') as f:
        for p in f.read().rstrip('\n').split('\n'):
            q, r = p.split(' ')
            retrieval[q].append(r)
    return dict(retrieval)


def names_to_pair_new(name0, name1, separator="/"):
    return separator.join((name0.replace("/", "-"), name1.replace("/", "-")))


def names_to_pair(name0, name1):
    return '_'.join((name0.replace('/', '-'), name1.replace('/', '-')))
