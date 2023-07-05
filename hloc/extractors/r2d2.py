import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as tvf

from ..utils.base_model import BaseModel


def load_network(model_fn):
    checkpoint = torch.load(model_fn)
    # print("\n>> Creating net = " + checkpoint['net'])
    print("\n Loaded R2D2 model")
    net = eval(checkpoint['net'])
    # nb_of_weights = common.model_size(net)
    # print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")
    # print(" (Model size: {:.0f}K parameters )".format(nb_of_weights / 1000))

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.', ''): v for k, v in weights.items()})
    return net.eval()


class BaseNet(nn.Module):
    """ Takes a list of images as input, and returns for each image:
        - a pixelwise descriptor
        - a pixelwise confidence
    """

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors=F.normalize(x, p=2, dim=1),
                    repeatability=self.softmax(urepeatability),
                    reliability=self.softmax(ureliability))

    def forward_one(self, x):
        raise NotImplementedError()

    def forward(self, imgs, **kw):
        res = [self.forward_one(img) for img in imgs]
        # merge all dictionaries into one
        res = {k: [r[k] for r in res if k in r] for k in {k for r in res for k in r}}
        return dict(res, imgs=imgs, **kw)


class PatchNet(BaseNet):
    """ Helper class to construct a fully-convolutional network that
        extract a l2-normalized patch descriptor.
    """

    def __init__(self, inchan=3, dilated=True, dilation=1, bn=True, bn_affine=False):
        BaseNet.__init__(self)
        self.inchan = inchan
        self.curchan = inchan
        self.dilated = dilated
        self.dilation = dilation
        self.bn = bn
        self.bn_affine = bn_affine
        self.ops = nn.ModuleList([])

    def _make_bn(self, outd):
        return nn.BatchNorm2d(outd, affine=self.bn_affine)

    def _add_conv(self, outd, k=3, stride=1, dilation=1, bn=True, relu=True):
        d = self.dilation * dilation
        if self.dilated:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=1)
            self.dilation *= stride
        else:
            conv_params = dict(padding=((k - 1) * d) // 2, dilation=d, stride=stride)
        self.ops.append(nn.Conv2d(self.curchan, outd, kernel_size=k, **conv_params))
        if bn and self.bn: self.ops.append(self._make_bn(outd))
        if relu: self.ops.append(nn.ReLU(inplace=True))
        self.curchan = outd

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for n, op in enumerate(self.ops):
            x = op(x)
        return self.normalize(x)


class L2_Net(PatchNet):
    """ Compute a 128D descriptor for all overlapping 32x32 patches.
        From the L2Net paper (CVPR'17).
    """

    def __init__(self, dim=128, **kw):
        PatchNet.__init__(self, **kw)
        add_conv = lambda n, **kw: self._add_conv((n * dim) // 128, **kw)
        add_conv(32)
        add_conv(32)
        add_conv(64, stride=2)
        add_conv(64)
        add_conv(128, stride=2)
        add_conv(128)
        add_conv(128, k=7, stride=8, bn=False, relu=False)
        self.out_dim = dim


class Quad_L2Net(PatchNet):
    """ Same than L2_Net, but replace the final 8x8 conv by 3 successive 2x2 convs.
    """

    def __init__(self, dim=128, mchan=4, relu22=False, **kw):
        PatchNet.__init__(self, **kw)
        self._add_conv(8 * mchan)
        self._add_conv(8 * mchan)
        self._add_conv(16 * mchan, stride=2)
        self._add_conv(16 * mchan)
        self._add_conv(32 * mchan, stride=2)
        self._add_conv(32 * mchan)
        # replace last 8x8 convolution with 3 2x2 convolutions
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(32 * mchan, k=2, stride=2, relu=relu22)
        self._add_conv(dim, k=2, stride=2, bn=False, relu=False)
        self.out_dim = dim


class Quad_L2Net_ConfCFS(Quad_L2Net):
    """ Same than Quad_L2Net, with 2 confidence maps for repeatability and reliability.
    """

    def __init__(self, **kw):
        Quad_L2Net.__init__(self, **kw)
        # reliability classifier
        self.clf = nn.Conv2d(self.out_dim, 2, kernel_size=1)
        # repeatability classifier: for some reasons it's a softplus, not a softmax!
        # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
        self.sal = nn.Conv2d(self.out_dim, 1, kernel_size=1)

    def forward_one(self, x):
        assert self.ops, "You need to add convolutions first"
        for op in self.ops:
            x = op(x)
        # compute the confidence maps
        ureliability = self.clf(x ** 2)
        urepeatability = self.sal(x ** 2)
        return self.normalize(x, ureliability, urepeatability)


class NonMaxSuppression(torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr

    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def process_singlescale(net, img, detector):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    with torch.no_grad():
        res = net(imgs=[img])

    # get output and reliability map
    descriptors = res['descriptors'][0]
    reliability = res['reliability'][0]
    repeatability = res['repeatability'][0]

    # normalize the reliability for nms
    # extract maxima and descs
    with torch.no_grad():
        y, x = detector(**res)  # nms
    c = reliability[0, 0, y, x]
    q = repeatability[0, 0, y, x]
    d = descriptors[0, :, y, x].t()
    n = d.shape[0]
    X, Y, S, C, Q, D = [], [], [], [], [], []

    X.append(x.float())
    Y.append(y.float())
    C.append(c)
    Q.append(q)
    D.append(d)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    scores = torch.cat(C) * torch.cat(Q)

    XYS = torch.stack([X, Y], dim=-1)
    D = torch.cat(D)

    return XYS.cpu().numpy(), D.cpu().numpy(), scores.cpu().numpy()


def process_multiscale(net, img, detector, scale_f=2 ** 0.25,
                       min_scale=0.0, max_scale=1,
                       min_size=256, max_size=1024,
                       verbose=False,
                       scales=[1.0, 0.86, 0.72, 0.6, 0.5]):
    old_bm = torch.backends.cudnn.benchmark
    torch.backends.cudnn.benchmark = False  # speedup

    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"

    # assert max_scale <= 1
    # s = 1.0  # current scale factor

    X, Y, S, C, Q, D = [], [], [], [], [], []
    pts_list = []
    # while s + 0.001 >= max(min_scale, min_size / max(H, W)):
    #     if s - 0.001 <= min(max_scale, max_size / max(H, W)):
    for s in scales:
        # nh, nw = img.shape[2:]
        nh, nw = round(H * s), round(W * s)
        img = F.interpolate(img, (nh, nw), mode='bilinear', align_corners=False)
        if verbose:
            # print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d}")
            print("extracting at scale x{:.02f} = {:4d}x{:3d}".format(s, nw, nh))
            # extract descriptors
        with torch.no_grad():
            res = net(imgs=[img])
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            with torch.no_grad():
                y, x = detector(**res)  # nms
            c = reliability[0, 0, y, x]
            q = repeatability[0, 0, y, x]
            d = descriptors[0, :, y, x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W / nw)
            Y.append(y.float() * H / nh)
            S.append((32 / s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)

            pts_list.append(torch.stack([x.float() * W / nw, y.float() * H / nh], dim=-1))

        # s /= scale_f
        # down-scale the image for next iteration

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    # print("Y: ", len(Y))
    # print("X: ", len(X))
    # print("S: ", len(S))
    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S)  # scale
    scores = torch.cat(C) * torch.cat(Q)  # scores = reliability * repeatability
    XYS = torch.stack([X, Y], dim=-1)
    D = torch.cat(D)

    XYS = XYS.cpu().numpy()
    D = D.cpu().numpy()
    scores = scores.cpu().numpy()
    return XYS, D, scores


class R2D2(BaseModel):
    default_conf = {
        'max_keypoints': -1,
        'remove_borders': 4,
    }

    required_inputs = ['image']

    def _init(self, conf):
        print(conf)
        self.dector = NonMaxSuppression(rel_thr=conf["rel_th"], rep_thr=conf["rep_th"]).cuda().eval()
        self.net = load_network(model_fn=conf["model_fn"])

    def _forward(self, data):
        RGB_mean = [0.485, 0.456, 0.406]
        RGB_std = [0.229, 0.224, 0.225]

        norm_RGB = tvf.Compose([tvf.Normalize(mean=RGB_mean, std=RGB_std)])
        image = data['image']
        image = norm_RGB(image)
        if self.conf['multiscale']:
            keypoints, descriptors, scores = process_multiscale(net=self.net, detector=self.dector, img=image,
                                                                max_size=9999)
        else:
            keypoints, descriptors, scores = process_singlescale(net=self.net, img=image, detector=self.dector)

        if self.conf["max_keypoints"]:
            topK = self.conf["max_keypoints"]
            if topK < keypoints.shape[0]:
                idxs = (-scores).argsort()[:topK]
                keypoints = keypoints[idxs]
                descriptors = descriptors[idxs]
                scores = scores[idxs]
        return {
            'keypoints': torch.from_numpy(keypoints)[None],
            'scores': torch.from_numpy(scores)[None],
            'descriptors': torch.from_numpy(descriptors.T)[None],
        }
