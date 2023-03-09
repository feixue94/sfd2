# -*- coding: UTF-8 -*-
'''=================================================
@Project -> File   swd2 -> swinnet
@IDE    PyCharm
@Author fx221@cam.ac.uk
@Date   03/07/2021 09:50
=================================================='''
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from nets.swin_transformer import swin_base
from nets.swin_transformer import PatchEmbed, BasicLayer, PatchMerging
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from nets.checkpoint import load_checkpoint
from mmseg.utils import get_root_logger


class PatchUpsampling(nn.Module):
    """ Patch Upsampleing Layer
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super(PatchUpsampling, self).__init__()
        self.dim = dim
        self.upsample = nn.Linear(4 * dim, 8 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x, H, W):
        """ Forward function.

        Args:
            x: Input feature, tensor size (B, H*W, C).
            H, W: Spatial resolution of the input feature.
        """
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"
        x = x.view(B, H, W, C)


# class SwinNet(nn.Module):
#     def __init__(self,
#                  pretrain_img_size=224,
#                  patch_size=4,
#                  in_chans=3,
#                  embed_dim=96,
#                  # depths=[2, 2, 6, 2],
#                  depths=[2, 2, 2, 2],
#                  # num_heads=[3, 6, 12, 24],
#                  num_heads=[3, 3, 3, 3],
#                  window_size=7,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  drop_path_rate=0.2,
#                  norm_layer=nn.LayerNorm,
#                  ape=False,
#                  patch_norm=True,
#                  out_indices=(0, 1, 2, 3),
#                  frozen_stages=-1,
#                  use_checkpoint=False
#                  ):
#         super(SwinNet, self).__init__()
#
#         self.pretrain_img_size = pretrain_img_size
#         self.num_layers = len(depths)
#         self.embed_dim = embed_dim
#         self.ape = ape
#         self.patch_norm = patch_norm
#         self.out_indices = out_indices
#         self.frozen_stages = frozen_stages
#
#         # split image into non-overlapping patches
#         self.patch_embed = PatchEmbed(
#             patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
#             norm_layer=norm_layer if self.patch_norm else None)
#
#         # absolute position embedding
#         if self.ape:
#             pretrain_img_size = to_2tuple(pretrain_img_size)
#             patch_size = to_2tuple(patch_size)
#             patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]
#
#             self.absolute_pos_embed = nn.Parameter(
#                 torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
#             trunc_normal_(self.absolute_pos_embed, std=.02)
#
#         self.pos_drop = nn.Dropout(p=drop_rate)
#
#         # stochastic depth
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
#
#         # build layers
#
#         self.layer1 = BasicLayer(
#             dim=int(embed_dim),
#             # dim=128,
#             depth=depths[0],
#             num_heads=num_heads[0],
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             drop=drop_rate,
#             attn_drop=attn_drop_rate,
#             drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
#             # downsample=PatchMerging,
#             downsample=None,
#             use_checkpoint=use_checkpoint
#         )
#
#         self.layer2 = BasicLayer(
#             dim=int(embed_dim),
#             # dim=128,
#             depth=depths[1],
#             num_heads=num_heads[1],
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             drop=drop_rate,
#             attn_drop=attn_drop_rate,
#             drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
#             downsample=None,
#             use_checkpoint=use_checkpoint
#         )
#
#         self.layer3 = BasicLayer(
#             dim=int(embed_dim),
#             # dim=256,
#             depth=depths[2],
#             num_heads=num_heads[2],
#             window_size=window_size,
#             mlp_ratio=mlp_ratio,
#             qkv_bias=qkv_bias,
#             qk_scale=qk_scale,
#             drop=drop_rate,
#             attn_drop=attn_drop_rate,
#             drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
#             downsample=None,
#             use_checkpoint=use_checkpoint
#         )
#
#         # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
#         self.num_features = [128, 128, 128]
#         self.norm = norm_layer(96)
#
#         # add a norm layer for each output
#         # for i_layer in out_indices:
#         #     layer = norm_layer(num_features[i_layer])
#         #     layer_name = f'norm{i_layer}'
#         #     self.add_module(layer_name, layer)
#
#         self.deconv1 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1,
#                                output_padding=1),
#             nn.BatchNorm2d(128, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128, affine=True),
#             nn.ReLU(inplace=True),
#         )
#
#         self.deconv2 = nn.Sequential(
#             nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
#                                output_padding=1),
#             nn.BatchNorm2d(128, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128, affine=True),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(128, affine=True),
#             nn.ReLU(inplace=True),
#
#         )
#
#         self.conv = nn.Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
#             nn.BatchNorm2d(128, affine=False, track_running_stats=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
#             nn.BatchNorm2d(128, affine=False, track_running_stats=True),
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
#         )
#
#         self.clf = nn.Conv2d(128, 2, kernel_size=1)
#         # repeatability classifier: for some reasons it's a softplus, not a softmax!
#         # Why? I guess it's a mistake that was left unnoticed in the code for a long time...
#         self.sal = nn.Conv2d(128, 1, kernel_size=1)
#
#     def forward_one(self, x):
#         """forward"""
#         x = self.patch_embed(x)
#         Wh, Ww = x.size(2), x.size(3)
#         if self.ape:
#             # interpolate the position embedding to the corresponding size
#             absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
#             x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
#         else:
#             x = x.flatten(2).transpose(1, 2)
#         x = self.pos_drop(x)
#
#         x_out, H, W, x, Wh, Ww = self.layer1(x, Wh, Ww)
#         # print('after layer1: ', x_out.shape)
#
#         x_out, H, W, x, Wh, Ww = self.layer2(x, Wh, Ww)
#         # print('after layer2: ', x_out.shape)
#
#         x_out, H, W, x, Wh, Ww = self.layer3(x, Wh, Ww)
#         # print('after layer3: ', x_out.shape)
#
#         x_out = self.norm(x_out)
#         # print(x_out.shape)
#         out = x_out.view(-1, H, W, 96).permute(0, 3, 1, 2).contiguous()
#         # print(out.shape)
#         # exit(0)
#         out = self.deconv1(out)
#         out = self.deconv2(out)
#         out = self.conv(out)
#
#         ureliability = self.clf(out ** 2)
#         urepeatability = self.sal(out ** 2)
#
#
#         return self.normalize(x=out,
#                               ureliability=ureliability,
#                               urepeatability=urepeatability)
#
#     def forward(self, imgs, **kwargs):
#         res = [self.forward_one(img) for img in imgs]
#         # merge all dictionaries into one
#         res = {k: [r[k] for r in res if k in r] for k in {k for r in res for k in r}}
#         return dict(res, imgs=imgs, **kwargs)
#
#     def _freeze_stages(self):
#         if self.frozen_stages >= 0:
#             self.patch_embed.eval()
#             for param in self.patch_embed.parameters():
#                 param.requires_grad = False
#
#         if self.frozen_stages >= 1 and self.ape:
#             self.absolute_pos_embed.requires_grad = False
#
#         if self.frozen_stages >= 2:
#             self.pos_drop.eval()
#             for i in range(0, self.frozen_stages - 1):
#                 m = self.layers[i]
#                 m.eval()
#                 for param in m.parameters():
#                     param.requires_grad = False
#
#     def init_weights(self, pretrained=None):
#         """Initialize the weights in backbone.
#
#         Args:
#             pretrained (str, optional): Path to pre-trained weights.
#                 Defaults to None.
#         """
#
#         def _init_weights(m):
#             if isinstance(m, nn.Linear):
#                 trunc_normal_(m.weight, std=.02)
#                 if isinstance(m, nn.Linear) and m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.LayerNorm):
#                 nn.init.constant_(m.bias, 0)
#                 nn.init.constant_(m.weight, 1.0)
#
#         if isinstance(pretrained, str):
#             self.apply(_init_weights)
#             logger = get_root_logger()
#             load_checkpoint(self, pretrained, strict=False, logger=logger)
#         elif pretrained is None:
#             self.apply(_init_weights)
#         else:
#             raise TypeError('pretrained must be a str or None')
#
#     def normalize(self, x, ureliability, urepeatability):
#         return dict(descriptors=F.normalize(x, p=2, dim=1),
#                     repeatability=self.softmax(urepeatability),
#                     reliability=self.softmax(ureliability))
#
#     def softmax(self, ux):
#         if ux.shape[1] == 1:
#             x = F.softplus(ux)
#             return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
#         elif ux.shape[1] == 2:
#             return F.softmax(ux, dim=1)[:, 1:2]


class SwinNet(nn.Module):
    def __init__(self,
                 pretrain_img_size=224,
                 patch_size=4,
                 in_chans=3,
                 embed_dim=96,
                 # depths=[2, 2, 6, 2],
                 depths=[2, 2, 2, 2],
                 # num_heads=[3, 6, 12, 24],
                 num_heads=[3, 3, 3, 3],
                 window_size=7,
                 mlp_ratio=4.,
                 qkv_bias=True,
                 qk_scale=None,
                 drop_rate=0.,
                 attn_drop_rate=0.,
                 drop_path_rate=0.2,
                 norm_layer=nn.LayerNorm,
                 ape=False,
                 patch_norm=True,
                 out_indices=(0, 1, 2, 3),
                 frozen_stages=-1,
                 use_checkpoint=False
                 ):
        super(SwinNet, self).__init__()

        self.pretrain_img_size = pretrain_img_size
        self.num_layers = len(depths)
        self.embed_dim = embed_dim
        self.ape = ape
        self.patch_norm = patch_norm
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(
            patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim,
            norm_layer=norm_layer if self.patch_norm else None)

        # absolute position embedding
        if self.ape:
            pretrain_img_size = to_2tuple(pretrain_img_size)
            patch_size = to_2tuple(patch_size)
            patches_resolution = [pretrain_img_size[0] // patch_size[0], pretrain_img_size[1] // patch_size[1]]

            self.absolute_pos_embed = nn.Parameter(
                torch.zeros(1, embed_dim, patches_resolution[0], patches_resolution[1]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule

        # build layers

        self.layer1 = BasicLayer(
            dim=int(embed_dim),
            # dim=128,
            depth=depths[0],
            num_heads=num_heads[0],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:0]):sum(depths[:0 + 1])],
            # downsample=PatchMerging,
            downsample=None,
            use_checkpoint=use_checkpoint
        )

        self.layer2 = BasicLayer(
            dim=int(embed_dim),
            # dim=128,
            depth=depths[1],
            num_heads=num_heads[1],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:1]):sum(depths[:1 + 1])],
            downsample=None,
            use_checkpoint=use_checkpoint
        )

        self.layer3 = BasicLayer(
            dim=int(embed_dim),
            # dim=256,
            depth=depths[2],
            num_heads=num_heads[2],
            window_size=window_size,
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            drop=drop_rate,
            attn_drop=attn_drop_rate,
            drop_path=dpr[sum(depths[:2]):sum(depths[:2 + 1])],
            downsample=None,
            use_checkpoint=use_checkpoint
        )

        # num_features = [int(embed_dim * 2 ** i) for i in range(self.num_layers)]
        self.num_features = [128, 128, 128]
        self.norm = norm_layer(96)

        # add a norm layer for each output
        # for i_layer in out_indices:
        #     layer = norm_layer(num_features[i_layer])
        #     layer_name = f'norm{i_layer}'
        #     self.add_module(layer_name, layer)

        self.deconv1 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=96, out_channels=128, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
        )

        self.deconv2 = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=3, stride=2, padding=1,
                               output_padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, affine=True),
            nn.ReLU(inplace=True),

        )

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=2, dilation=4),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=4, dilation=8),
            nn.BatchNorm2d(128, affine=False, track_running_stats=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=2, stride=1, padding=8, dilation=16),
        )

        self.convDb = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0)
        self.convPb = nn.Conv2d(128, 1, kernel_size=1, padding=0)

        self.init_weights(pretrained=None)

    def det(self, x):
        """forward"""
        x = self.patch_embed(x)
        Wh, Ww = x.size(2), x.size(3)
        if self.ape:
            # interpolate the position embedding to the corresponding size
            absolute_pos_embed = F.interpolate(self.absolute_pos_embed, size=(Wh, Ww), mode='bicubic')
            x = (x + absolute_pos_embed).flatten(2).transpose(1, 2)  # B Wh*Ww C
        else:
            x = x.flatten(2).transpose(1, 2)
        x = self.pos_drop(x)

        x_out, H, W, x, Wh, Ww = self.layer1(x, Wh, Ww)
        # print('after layer1: ', x_out.shape)

        x_out, H, W, x, Wh, Ww = self.layer2(x, Wh, Ww)
        # print('after layer2: ', x_out.shape)

        x_out, H, W, x, Wh, Ww = self.layer3(x, Wh, Ww)
        # print('after layer3: ', x_out.shape)

        x_out = self.norm(x_out)
        # print(x_out.shape)
        out = x_out.view(-1, H, W, 96).permute(0, 3, 1, 2).contiguous()
        # print(out.shape)
        # exit(0)
        out = self.deconv1(out)
        out = self.deconv2(out)
        out = self.conv(out)

        score = self.convPb(out).squeeze()
        score = torch.sigmoid(score)

        desc = self.convDb(out)
        desc = F.normalize(desc, dim=1)

        return score, desc

    def forward(self, batch):
        # b = batch['image1'].size(0)
        x = torch.cat([batch['image1'], batch['image2']], dim=0)

        score, desc = self.det(x)
        return {
            "score": score,
            "desc": desc,
        }

        # if self.require_feature:
        #     score, desc, seg_feats = self.det(x)
        #     return {
        #         "score": score,
        #         "desc": desc,
        #         "pred_feats": seg_feats,
        #     }
        #
        # else:
        #     score, desc = self.det(x)
        #     return {
        #         "score": score,
        #         "desc": desc,
        #     }
        #
        # desc1 = desc[:b, :, :, :]
        # desc2 = desc[b:, :, :, :]
        #
        # score1 = score[:b, :, :]
        # score2 = score[b:, :, :]
        #
        # return {
        #     'dense_features1': desc1,
        #     'scores1': score1,
        #     'dense_features2': desc2,
        #     'scores2': score2,
        # }

    def init_weights(self, pretrained=None):
        """Initialize the weights in backbone.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
                Defaults to None.
        """

        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=.02)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)

        if isinstance(pretrained, str):
            self.apply(_init_weights)
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            self.apply(_init_weights)
        else:
            raise TypeError('pretrained must be a str or None')

    def normalize(self, x, ureliability, urepeatability):
        return dict(descriptors=F.normalize(x, p=2, dim=1),
                    repeatability=self.softmax(urepeatability),
                    reliability=self.softmax(ureliability))

    def softmax(self, ux):
        if ux.shape[1] == 1:
            x = F.softplus(ux)
            return x / (1 + x)  # for sure in [0,1], much less plateaus than softmax
        elif ux.shape[1] == 2:
            return F.softmax(ux, dim=1)[:, 1:2]


if __name__ == '__main__':
    img = torch.ones((4, 3, 256, 256)).cuda()
    # img = torch.ones((4, 3, 224, 224)).cuda()  # .cuda()
    net = SwinNet().cuda()
    out = net(img)
    for k in out:
        print(out[k].shape)
