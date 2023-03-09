# -*- coding: utf-8 -*-
"""
@Time ： 2021/3/17 上午10:07
@Auth ： Fei Xue
@File ： mobilenet.py
@Email： feixue@pku.edu.cn
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from collections import OrderedDict
import os
from mmcv.cnn import ConvModule, constant_init, kaiming_init
from mmcv.runner import load_checkpoint
from torch.nn.modules.batchnorm import _BatchNorm
from torch.utils import checkpoint as cp
import logging


def _make_divisible(v, divisor, min_value=None):
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def preprocess_mobilenet_shared_weights(weights: OrderedDict):
    # extract for shared weights
    processed_weights_dict = OrderedDict()
    for key, val in weights.items():
        if key.startswith('features.') and 7 > int(key.split('.')[1]) > 1:
            # print(key)
            key = key[9:]
            processed_weights_dict[key] = val

    return processed_weights_dict


def conv_3x3_bn(inp, oup, stride):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidualV2(nn.Module):
    """InvertedResidual block for MobileNetV2.
    Args:
        in_channels (int): The input channels of the InvertedResidual block.
        out_channels (int): The output channels of the InvertedResidual block.
        stride (int): Stride of the middle (first) 3x3 convolution.
        expand_ratio (int): Adjusts number of channels of the hidden layer
            in InvertedResidual by this amount.
        dilation (int): Dilation rate of depthwise conv. Default: 1
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    Returns:
        Tensor: The output tensor.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 stride,
                 expand_ratio,
                 dilation=1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 with_cp=False):
        super(InvertedResidualV2, self).__init__()
        self.stride = stride
        assert stride in [1, 2], f'stride must in [1, 2]. ' \
                                 f'But received {stride}.'
        self.with_cp = with_cp
        self.use_res_connect = self.stride == 1 and in_channels == out_channels
        hidden_dim = int(round(in_channels * expand_ratio))

        layers = []
        if expand_ratio != 1:
            layers.append(
                ConvModule(
                    in_channels=in_channels,
                    out_channels=hidden_dim,
                    kernel_size=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    act_cfg=act_cfg))
        layers.extend([
            ConvModule(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                stride=stride,
                padding=dilation,
                dilation=dilation,
                groups=hidden_dim,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg),
            ConvModule(
                in_channels=hidden_dim,
                out_channels=out_channels,
                kernel_size=1,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=None)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):

        def _inner_forward(x):
            if self.use_res_connect:
                return x + self.conv(x)
            else:
                return self.conv(x)

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]

        hidden_dim = round(inp * expand_ratio)
        self.identity = stride == 1 and inp == oup

        if expand_ratio == 1:
            self.conv = nn.Sequential(
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )
        else:
            self.conv = nn.Sequential(
                # pw
                nn.Conv2d(inp, hidden_dim, 1, 1, 0, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # dw
                nn.Conv2d(hidden_dim, hidden_dim, 3, stride, 1, groups=hidden_dim, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True),
                # pw-linear
                nn.Conv2d(hidden_dim, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
            )

    def forward(self, x):
        if self.identity:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    """MobileNetV2 backbone.
    Args:
        widen_factor (float): Width multiplier, multiply number of
            channels in each layer by this amount. Default: 1.0.
        strides (Sequence[int], optional): Strides of the first block of each
            layer. If not specified, default config in ``arch_setting`` will
            be used.
        dilations (Sequence[int]): Dilation of each layer.
        out_indices (None or Sequence[int]): Output from which stages.
            Default: (7, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Default: -1, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Default: None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='ReLU6').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Default: False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Default: False.
    """

    # Parameters to build layers. 3 parameters are needed to construct a
    # layer, from left to right: expand_ratio, channel, num_blocks.
    arch_settings = [[1, 16, 1], [6, 24, 2], [6, 32, 3], [6, 64, 4],
                     [6, 96, 3], [6, 160, 3], [6, 320, 1]]

    def __init__(self,
                 widen_factor=1.,
                 strides=(1, 2, 2, 2, 1, 2, 1),
                 dilations=(1, 1, 1, 1, 1, 1, 1),
                 out_indices=(1, 2, 4, 6),
                 frozen_stages=-1,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU6'),
                 norm_eval=False,
                 with_cp=False):
        super(MobileNetV2, self).__init__()
        self.widen_factor = widen_factor
        self.strides = strides
        self.dilations = dilations
        assert len(strides) == len(dilations) == len(self.arch_settings)
        self.out_indices = out_indices
        for index in out_indices:
            if index not in range(0, 7):
                raise ValueError('the item in out_indices must in '
                                 f'range(0, 8). But received {index}')

        if frozen_stages not in range(-1, 7):
            raise ValueError('frozen_stages must be in range(-1, 7). '
                             f'But received {frozen_stages}')
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp

        self.in_channels = _make_divisible(32 * widen_factor, 8)

        self.conv1 = ConvModule(
            in_channels=3,
            out_channels=self.in_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

        self.layers = []

        for i, layer_cfg in enumerate(self.arch_settings):
            expand_ratio, channel, num_blocks = layer_cfg
            stride = self.strides[i]
            dilation = self.dilations[i]
            out_channels = _make_divisible(channel * widen_factor, 8)
            inverted_res_layer = self.make_layer(
                out_channels=out_channels,
                num_blocks=num_blocks,
                stride=stride,
                dilation=dilation,
                expand_ratio=expand_ratio)
            layer_name = f'layer{i + 1}'
            self.add_module(layer_name, inverted_res_layer)
            self.layers.append(layer_name)

    def make_layer(self, out_channels, num_blocks, stride, dilation,
                   expand_ratio):
        """Stack InvertedResidual blocks to build a layer for MobileNetV2.
        Args:
            out_channels (int): out_channels of block.
            num_blocks (int): Number of blocks.
            stride (int): Stride of the first block.
            dilation (int): Dilation of the first block.
            expand_ratio (int): Expand the number of channels of the
                hidden layer in InvertedResidual by this ratio.
        """
        layers = []
        for i in range(num_blocks):
            layers.append(
                InvertedResidualV2(
                    self.in_channels,
                    out_channels,
                    stride if i == 0 else 1,
                    expand_ratio=expand_ratio,
                    dilation=dilation if i == 0 else 1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg,
                    with_cp=self.with_cp))
            self.in_channels = out_channels

        return nn.Sequential(*layers)

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')

    def forward(self, x):
        x = self.conv1(x)

        outs = []
        for i, layer_name in enumerate(self.layers):
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)

        if len(outs) == 1:
            return outs[0]
        else:
            return tuple(outs)

    def extract(self, x, required_indices=[0, 1, 2]):
        print("x: ", x.shape)
        x = self.conv1(x)
        max_id = max(required_indices)
        outs = []
        for i, layer_name in enumerate(self.layers):
            if i > max_id:
                break
            layer = getattr(self, layer_name)
            x = layer(x)
            if i in required_indices:
                outs.append(x)
        return tuple(outs)

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            for param in self.conv1.parameters():
                param.requires_grad = False
        for i in range(1, self.frozen_stages + 1):
            layer = getattr(self, f'layer{i}')
            layer.eval()
            for param in layer.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(MobileNetV2, self).train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()


class MobileNetM(nn.Module):
    def __init__(self, middim=64, outdim=128, width_mult=1., freeze_encoder=False, require_feature=False,
                 seg_branch=False):
        super(MobileNetM, self).__init__()
        block = InvertedResidual
        input_channel = 16
        self.require_feature = require_feature
        self.seg_branch = seg_branch
        self.cfgs_encoder = [
            # t, c, n, s
            [1, 16, 1, 2],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            # [6, 64, 4, 2],
        ]

        layers_encoder = []
        layers_encoder.append(conv_3x3_bn(3, input_channel, 1))
        # add mentioned in cfgs_share
        for t, c, n, s in self.cfgs_encoder:
            output_channel = _make_divisible(c * width_mult, 4 if width_mult == 0.1 else 8)
            for i in range(n):
                layers_encoder.append(block(input_channel, output_channel, s if i == 0 else 1, t))
                input_channel = output_channel
        self.features_encoder = nn.Sequential(*layers_encoder)

        self._initialize_weights()

        if freeze_encoder:
            self.freeze_shared_weights()

        if self.seg_branch:
            self.segconv1 = conv_1x1_bn(inp=16, oup=16)
            self.segconv2 = conv_1x1_bn(inp=24, oup=24)
            self.segconv3 = conv_1x1_bn(inp=32, oup=32)

            iconv_incs = [32 * 2 + 24, 16 + 24 + middim, middim + 16 + 16]
        else:
            iconv_incs = [32 + 24, 16 + middim, middim + 16]

        self.iconv3 = conv_3x3_bn(inp=iconv_incs[0], oup=middim, stride=1)  # H/4
        self.iconv2 = conv_3x3_bn(inp=iconv_incs[1], oup=middim, stride=1)  # H/2
        self.iconv1 = conv_3x3_bn(inp=iconv_incs[2], oup=middim, stride=1)  # H
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=middim, out_channels=middim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(middim),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=middim, out_channels=outdim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outdim, affine=False, track_running_stats=True),
            nn.ReLU(inplace=True),
        )
        self.convDb = nn.Conv2d(in_channels=outdim, out_channels=outdim, kernel_size=1, stride=1, padding=0)
        self.convPb = nn.Conv2d(outdim, 1, kernel_size=1, padding=0)

    def freeze_shared_weights(self):
        for param in self.features_encoder.parameters():
            param.requires_grad = False

    def det(self, x):
        out1 = self.features_encoder[0](x)  # 16, H
        # out1_pool = self.max_pooling(out1)
        out2 = self.features_encoder[1:2](out1)  # 16, H/2
        out3 = self.features_encoder[2:4](out2)  # 24, H/4
        out4 = self.features_encoder[4:](out3)  # 32, H/8

        if self.seg_branch:
            seg_out2 = self.segconv1(out2)
            seg_out3 = self.segconv2(out3)
            seg_out4 = self.segconv3(out4)

            # out4 = torch.cat([out4, seg_out4], dim=1)
            # out3 = torch.cat([out3, seg_out3], dim=1)
            # out2 = torch.cat([out2, seg_out2], dim=1)

        # print("out1: ", out1.shape)
        # print("out2: ", out2.shape)
        # print("out3: ", out3.shape)
        # print("out4: ", out4.shape)
        if self.seg_branch:
            out4c = torch.cat([out4, seg_out4], dim=1)
        else:
            out4c = out4
        out4_up = F.interpolate(out4c, size=(out3.shape[2], out3.shape[3]), mode="bilinear", align_corners=True)
        out3_iv = self.iconv3(torch.cat([out4_up, out3], dim=1))

        if self.seg_branch:
            out3c = torch.cat([out3_iv, seg_out3], dim=1)
        else:
            out3c = out3_iv
        out3_up = F.interpolate(out3c, size=(out2.shape[2], out2.shape[3]), mode="bilinear", align_corners=True)
        out2_iv = self.iconv2(torch.cat([out3_up, out2], dim=1))

        if self.seg_branch:
            out2c = torch.cat([out2_iv, seg_out2], dim=1)
        else:
            out2c = out2_iv
        out2_up = F.interpolate(out2c, size=(out1.shape[2], out1.shape[3]), mode="bilinear", align_corners=True)
        out1_iv = self.iconv1(torch.cat([out2_up, out1], dim=1))

        out = self.conv(out1_iv)
        score = self.convPb(out).squeeze()
        score = torch.sigmoid(score)
        desc = self.convDb(out)
        desc = F.normalize(desc, dim=1)

        if self.require_feature:
            if self.seg_branch:
                return score, desc, (seg_out2, seg_out3, seg_out4)
            else:
                return score, desc, (out2, out3, out4)
        return score, desc

    def forward(self, batch):
        b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        score, desc, feats = self.det(x)

        return {
            "score": score,
            "desc": desc,
            "pred_feats": feats
        }

        desc1 = desc[:b, :, :, :]
        desc2 = desc[b:, :, :, :]

        score1 = score[:b, :, :]
        score2 = score[b:, :, :]

        return {
            'dense_features1': desc1,
            'scores1': score1,
            'dense_features2': desc2,
            'scores2': score2,
        }

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

        # init self.features_ext_share weights by imagenet pretrained model
        try:
            # TODO: preprocessing pretrained weights
            weight_path = os.path.join(os.getcwd(), 'weights/mobilenetv2_1.0-f2a8633.pth')
            shared_ext_state_dict = torch.load(weight_path,
                                               map_location='cpu')
            # print(shared_ext_state_dict.keys())
            state_dict = preprocess_mobilenet_shared_weights(shared_ext_state_dict)
            # print(state_dict.keys())
            self.features_encoder.load_state_dict(state_dict,
                                                  strict=False)
            print("Load pretained weights for mobilenetv2.")
        except FileNotFoundError:
            print(
                '{} is not found.\ninit by default method.'.format('./pretrained_weights/mobilenetv2_1.0-f2a8633.pth'))
        # print('init mobilenet')


if __name__ == '__main__':
    import datetime
    import time

    weight_path = "semseg/checkpoints/deeplabv3plus_m-v2-d8_512x512_160k_ade20k_20200825_223255-465a01d4.pth"

    with torch.no_grad():
        img = torch.rand((1, 3, 720, 1280)).cuda()
        # net = MobileNetM(require_feature=True, seg_branch=True).cuda().eval()
        net = MobileNetV2().cuda().eval()
        print(net)

        a = net.extract(x=img, required_indices=[0, 1, 2, 3, 4, 5])
        for v in a:
            print(v.shape)
        exit(0)
        # print(net)

        start_t = time.time()
        n = 1
        for i in range(n):
            ouput = net.det(img)
        end_time = time.time()
        print((end_time - start_t) / n)
