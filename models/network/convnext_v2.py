# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from torch import distributed as dist
from timm.models.layers import trunc_normal_, DropPath
from itertools import accumulate


class LayerNorm(nn.Module):
    """ LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    """

    def __init__(self, dim):
        super().__init__()
        self.gamma = nn.Parameter(torch.zeros(1, 1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, 1, dim))

    def forward(self, x):
        Gx = torch.norm(x, p=2, dim=(1, 2), keepdim=True)
        Nx = Gx / (Gx.mean(dim=-1, keepdim=True) + 1e-6)
        return self.gamma * (x * Nx) + self.beta + x


class Block(nn.Module):
    """ ConvNeXtV2 Block.

    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
    """

    def __init__(self, dim, drop_path=0.):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)  # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)  # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.grn = GRN(4 * dim)
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.grn(x)
        x = self.pwconv2(x)
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class ConvNeXtV2(nn.Module):
    """ ConvNeXt V2

    Args:
        in_chans (int): Number of input image channels. Default: 3
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (tuple(int)): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_rate_range (tuple(float)): Stochastic depth rate range.
    """

    def __init__(self,
                 in_chans=3,
                 depths=[3, 3, 9, 3],
                 dims=[96, 192, 384, 768],
                 drop_rate_range=(0.1, 0.3),
                 use_ckpt=False,
                 freeze_stages=(),
                 pre_weights=None,
                 ):
        super().__init__()
        self.depths = depths
        self.use_ckpt = use_ckpt
        self.freeze_stages = freeze_stages

        dp_rates = [x.item() for x in torch.linspace(*drop_rate_range, sum(depths))]

        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.downsample_dropouts = nn.ModuleList([
            nn.Dropout(dp_rates[i]) if dp_rates[i] > 0. else nn.Identity()
            for i in accumulate([0] + depths[:-1])
        ])

        self.stages = nn.ModuleList()  # 4 feature resolution stages, each consisting of multiple residual blocks
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[Block(dims[i], dp_rates[cur + j]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self._init_weights(pre_weights)
        self._freeze_stages()

    def _init_weights(self, pre_weights):
        if pre_weights:
            u, w = self.load_state_dict(torch.load(pre_weights, map_location='cpu')['model'], strict=False)

            initialized = False
            if dist.is_available():
                initialized = dist.is_initialized()
            if initialized:
                rank = dist.get_rank()
            else:
                rank = 0

            if rank == 0:
                print('ConvNext v2: misaligned params during the loading of parameters: {} {}'.format(u, w))

        else:
            for m in self.modules():
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                   trunc_normal_(m.weight, std=.02)
                   nn.init.constant_(m.bias, 0)

    def _freeze_stages(self):
        """Freeze parameters."""
        for idx in self.freeze_stages:
            for block in self.stages[idx][1::2]:
                block.eval()
                for param in block.parameters():
                    param.requires_grad = False

    def train(self, mode=True):
        """Convert the model into training mode."""
        super().train(mode)
        self._freeze_stages()

    def forward(self, x):
        xs = list()
        for i in range(4):
            x =self.downsample_dropouts[i](self.downsample_layers[i](x))
            x = self.stages[i](x)
            xs.append(x)

        return xs

    def stem_forward(self, x):
        return x

    def stage_forward(self, x, stage_id):
        x = self.downsample_dropouts[stage_id](self.downsample_layers[stage_id](x))
        if self.use_ckpt:
            x = checkpoint.checkpoint(self.stages[stage_id], x)
        else:
            x = self.stages[stage_id](x)

        return x
