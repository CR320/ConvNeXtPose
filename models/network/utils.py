import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BatchNorm2d

try:
    from mmcv.ops import DeformConv2d
    has_mmcv_full = True
except (ImportError, ModuleNotFoundError):
    has_mmcv_full = False
assert has_mmcv_full == True


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.norm_shape = (num_channels,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # u = x.mean(1, keepdim=True)
        # s = (x - u).pow(2).mean(1, keepdim=True)
        # x = (x - u) / torch.sqrt(s + self.eps)
        # x = self.weight[:, None, None] * x + self.bias[:, None, None]
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.norm_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class BasicBlock(nn.Module):
    """BasicBlock for ResNet.

    Args:
        in_channels (int): Input channels of this block.
        out_channels (int): Output channels of this block.
        expansion (int): The ratio of ``out_channels/mid_channels`` where
            ``mid_channels`` is the output channels of conv1. This is a
            reserved argument in BasicBlock and should always be 1. Default: 1.
        stride (int): stride of the block. Default: 1
        dilation (int): dilation of convolution. Default: 1
        downsample (nn.Module): downsample operation on identity branch.
            Default: None.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='BN')
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=1,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert self.expansion == 1
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation

        self.norm1 = eval(norm)(self.mid_channels)
        self.norm2 = eval(norm)(self.mid_channels)

        self.conv1 = nn.Conv2d(
            in_channels,
            self.mid_channels,
            3,
            stride=stride,
            padding=dilation,
            dilation=dilation,
            bias=False)
        self.conv2 = nn.Conv2d(
            self.mid_channels,
            out_channels,
            3,
            padding=1,
            bias=False)

        self.act = eval(act)(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.act(out + identity)

        return out

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)


class AdaptiveActivationBlock(nn.Module):
    """Adaptive activation convolution block. "Bottom-up human pose estimation
    via disentangled keypoint regression", CVPR'2021.

    Args:
        in_channels (int): Number of input channels
        out_channels (int): Number of output channels
        groups (int): Number of groups. Generally equal to the
            number of joints.
        norm_cfg (dict): Config for normalization layers.
        act_cfg (dict): Config for activation layers.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 groups=1,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):

        super(AdaptiveActivationBlock, self).__init__()

        assert in_channels % groups == 0 and out_channels % groups == 0
        self.groups = groups

        regular_matrix = torch.tensor([[-1, -1, -1, 0, 0, 0, 1, 1, 1],
                                       [-1, 0, 1, -1, 0, 1, -1, 0, 1],
                                       [1, 1, 1, 1, 1, 1, 1, 1, 1]])
        self.register_buffer('regular_matrix', regular_matrix.float())

        self.transform_matrix_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=6 * groups,
            kernel_size=3,
            padding=1,
            groups=groups,
            bias=True)

        if has_mmcv_full:
            self.adapt_conv = DeformConv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
                groups=groups,
                deform_groups=groups)
        else:
            raise ImportError('Please install the full version of mmcv '
                              'to use `DeformConv2d`.')

        self.norm = eval(norm)(out_channels)
        self.act = eval(act)(inplace=True)

    def forward(self, x):
        B, _, H, W = x.size()
        residual = x

        affine_matrix = self.transform_matrix_conv(x)
        affine_matrix = affine_matrix.permute(0, 2, 3, 1).contiguous()
        affine_matrix = affine_matrix.view(B, H, W, self.groups, 2, 3)
        offset = torch.matmul(affine_matrix, self.regular_matrix)
        offset = offset.transpose(4, 5).reshape(B, H, W, self.groups * 18)
        offset = offset.permute(0, 3, 1, 2).contiguous()

        x = self.adapt_conv(x, offset)
        x = self.norm(x)
        x = self.act(x + residual)

        return x

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                nn.init.constant_(m.transform_matrix_conv.bias, 0)
