import math
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.network.utils import LayerNorm2d, BasicBlock, AdaptiveActivationBlock
from models.utils import get_locations


class ConfidencesHead(nn.Module):
    """Heatmaps for joints detection.
    Args:
        in_channels (list): Number of input channels.
        num_joints (int): Number of joint keys.
    """

    def __init__(self,
                 in_channels,
                 mid_channels=32,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):
        super().__init__()
        self.trans_blocks = nn.ModuleList([
            nn.Sequential(nn.Conv2d(in_channels=ch,
                                    out_channels=mid_channels,
                                    kernel_size=1,
                                    bias=False),
                          eval(norm)(mid_channels),
                          eval(act)(inplace=True))
            for ch in in_channels]
        )
        self.filter_blocks = nn.ModuleList([
            BasicBlock(mid_channels, mid_channels, norm=norm, act=act)
            for _ in in_channels]
        )
        self.cls_layer = nn.Conv2d(in_channels=mid_channels, out_channels=1, kernel_size=1)

        self.init_weights()

    def forward(self, xs):
        """Forward function."""
        assert isinstance(xs, list)

        conf_list = list()
        for i, x in enumerate(xs):
            x = self.trans_blocks[i](x)
            x = self.filter_blocks[i](x)
            confidences = self.cls_layer(x)
            conf_list.append(confidences)

        return conf_list

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.cls_layer.bias, -1 * math.log((1 - 0.01) / 0.01))


class JointsHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_joints,
                 num_filters_per_joint=12,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):
        super().__init__()
        num_offset_filters = num_joints * num_filters_per_joint
        self.trans_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(ch, num_offset_filters, kernel_size=1, bias=False),
                eval(norm)(num_offset_filters),
                eval(act)(inplace=True)
            )
            for ch in in_channels]
        )
        self.filter_blocks = nn.ModuleList([
            nn.Sequential(
                AdaptiveActivationBlock(num_offset_filters, num_offset_filters, num_joints, norm, act),
                AdaptiveActivationBlock(num_offset_filters, num_offset_filters, num_joints, norm, act)
            )
            for _ in in_channels]
        )
        self.reg_layer = nn.Conv2d(num_offset_filters, num_joints * 2, kernel_size=1, groups=num_joints)

        self.num_joints = num_joints
        self.init_weights()

    def forward(self, xs):
        """Forward function."""
        assert isinstance(xs, list)

        # multi-scale joint offsets prediction
        joints_offsets_list = list()
        for i, x in enumerate(xs):
            # long offsets regression
            x = self.trans_blocks[i](x)
            x = self.filter_blocks[i](x)
            offsets = self.reg_layer(x)

            B, _, h, w = offsets.shape
            offsets = offsets.reshape(B, -1, 2, h, w)
            joints_offsets_list.append(offsets)

        return joints_offsets_list

    def init_weights(self):
        """Initialize model weights."""
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                nn.init.constant_(m.transform_matrix_conv.bias, 0)


class RefineHead(nn.Module):
    def __init__(self,
                 in_channels,
                 num_joints,
                 num_filters_per_joint=8,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):
        super().__init__()
        num_offset_filters = num_joints * num_filters_per_joint
        self.trans_block = nn.Sequential(
            nn.Conv2d(in_channels, num_offset_filters, kernel_size=1, bias=False),
            eval(norm)(num_offset_filters),
            eval(act)(inplace=True)
        )
        self.filter_block = nn.Sequential(
            AdaptiveActivationBlock(num_offset_filters, num_offset_filters, num_joints, norm, act),
            AdaptiveActivationBlock(num_offset_filters, num_offset_filters, num_joints, norm, act)
        )
        self.reg_layer = nn.Conv2d(num_offset_filters, num_joints * 2, kernel_size=1, groups=num_joints)
        self.init_weights()

    def forward(self, x):
        x = self.trans_block(x)
        x = self.filter_block(x)
        short_offsets = self.reg_layer(x)

        B, _, H, W = short_offsets.shape
        short_offsets = short_offsets.reshape(B, -1, 2, H, W)

        return short_offsets

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)

        for m in self.modules():
            if hasattr(m, 'transform_matrix_conv'):
                nn.init.constant_(m.transform_matrix_conv.weight, 0)
                nn.init.constant_(m.transform_matrix_conv.bias, 0)


class HeatmapsHead(nn.Module):
    """Heatmaps for joints detection.
    Args:
        in_channels (int): Number of input channels.
        num_joints (int): Number of joint keys.
    """

    def __init__(self,
                 in_channels,
                 num_joints,
                 mid_channels=32,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):
        super().__init__()
        self.trans_block = nn.Sequential(
            nn.Conv2d(in_channels=in_channels,
                      out_channels=mid_channels,
                      kernel_size=1,
                      bias=False),
            eval(norm)(mid_channels),
            eval(act)(inplace=True)
        )
        self.filter_block = BasicBlock(mid_channels, mid_channels, norm=norm, act=act)
        self.heatmap_layer = nn.Conv2d(in_channels=mid_channels, out_channels=num_joints, kernel_size=1)
        self.init_weights()

    def forward(self, x):
        """Forward function."""
        x = self.trans_block(x)
        x = self.filter_block(x)
        heatmaps = self.heatmap_layer(x).sigmoid()

        return heatmaps

    def init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, LayerNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, val=1)
                nn.init.constant_(m.bias, 0)

        nn.init.constant_(self.heatmap_layer.bias, -1 * math.log((1 - 0.01) / 0.01))


class MSPose(nn.Module):
    def __init__(self,
                 type,
                 in_channels,
                 num_joints,
                 norm,
                 act,
                 backbone,
                 neck=None,
                 flip_mode=False,
                 flip_index=None,
                 pre_weights=None):
        super().__init__()
        self.type = type
        self.flip_mode = flip_mode
        self.flip_index = flip_index

        self.backbone = backbone
        self.neck = neck
        self.confidences_head = ConfidencesHead(in_channels[1:], norm=norm, act=act)
        self.joints_head = JointsHead(in_channels[1:], num_joints, norm=norm, act=act)
        self.heatmaps_head = HeatmapsHead(in_channels[0], num_joints, norm=norm, act=act)
        self.refine_head = RefineHead(in_channels[0], num_joints, norm=norm, act=act)
        self.init_weights(pre_weights)

    def train(self, mode=True):
        super(MSPose, self).train(mode)

        if self.flip_mode and mode:
            # freeze BN layer
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.SyncBatchNorm):
                    m.eval()

            # freeze blocks
            freeze_module_list = [
                self.backbone, self.neck, self.confidences_head, self.joints_head
            ]
            for module in freeze_module_list:
                if module is None:
                    continue
                module.eval()

    def init_weights(self, pre_weights):
        if pre_weights is not None:
            state_dict = torch.load(pre_weights, map_location='cpu')
            self.load_state_dict(state_dict['model'], strict=True)
            print('Init pose net from: {}'.format(pre_weights))

        if self.flip_mode:
            # freeze BN layer
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.weight.requires_grad = False
                    m.bias.requires_grad = False

            # freeze blocks
            freeze_module_list = [
                self.backbone, self.neck, self.confidences_head, self.joints_head
            ]
            for module in freeze_module_list:
                if module is None:
                    continue
                for param in module.parameters():
                    param.requires_grad = False

    @staticmethod
    def encode_joints(joints_offsets_list, short_offsets):
        # short offsets shape
        B, K, _, H, W = short_offsets.shape

        joints_list, joints_refine_list = list(), list()
        for offsets in joints_offsets_list:
            h, w = offsets.shape[-2:]

            # joint locations encoding
            grids = get_locations(h, w, offsets.device).T.reshape(2, h, w)
            joints = grids[None, None, :, :, :] + offsets

            # scale joints
            joints[:, :, 0] = joints[:, :, 0] * (W / w)
            joints[:, :, 1] = joints[:, :, 1] * (H / h)
            joints_list.append(einops.rearrange(joints, 'b k n h w -> b h w k n'))

            # refine joints
            joints = joints.detach()
            joints_refine = torch.zeros_like(joints)

            # warp & add short-range offsets
            joints_loc_T = einops.rearrange(joints, 'b k n h w -> b k h w n')
            norm_joints_loc = torch.stack([joints_loc_T[..., 0] / W, joints_loc_T[..., 1] / H], dim=-1)
            norm_joints_loc = norm_joints_loc * 2 - 1

            for k in range(K):
                short_offsets_k = short_offsets[:, k]
                short_offsets_warp = F.grid_sample(short_offsets_k,
                                                   norm_joints_loc[:, k],
                                                   padding_mode="border",
                                                   align_corners=False)
                joints_refine[:, k] = joints[:, k] + short_offsets_warp

            joints_refine_list.append(einops.rearrange(joints_refine, 'b k n h w -> b h w k n'))

        return joints_list, joints_refine_list

    def forward(self, input, phase='inference'):
        if self.flip_mode:
            input_flip = torch.flip(input, [3])
            input = torch.cat([input, input_flip], dim=0)

        # backbone inference
        fea_list = self.backbone(input)

        if self.neck:
            fea_list = self.neck(fea_list)

        # calculate heat-maps
        heatmaps = self.heatmaps_head(fea_list[0])

        # calculate refine offsets
        short_offsets = self.refine_head(fea_list[0])

        # calculate joint offsets
        ms_joint_offsets = self.joints_head(fea_list[1:])

        # calculate confidences and multi-scale offsets & weights
        ms_confidences = self.confidences_head(fea_list[1:])

        if self.flip_mode:
            num_batches = input.shape[0] // 2

            flipped_hms = torch.flip(heatmaps[num_batches:], [3])
            flipped_hms = flipped_hms[:, self.flip_index]
            heatmaps = (heatmaps[:num_batches] + flipped_hms) * 0.5

            flipped_offsets = torch.flip(short_offsets[num_batches:], [4])
            flipped_offsets[:, :, 0] = -flipped_offsets[:, :, 0]
            flipped_offsets = flipped_offsets[:, self.flip_index]
            short_offsets = (short_offsets[:num_batches] + flipped_offsets) * 0.5

            for i, (confs, offsets) in enumerate(zip(ms_confidences, ms_joint_offsets)):
                ms_confidences[i] = confs[:num_batches]
                ms_joint_offsets[i] = offsets[:num_batches]

        ms_joints, ms_joints_ref = self.encode_joints(ms_joint_offsets, short_offsets)

        if phase == 'inference':
            return dict(ms_confidences=[x.sigmoid() for x in ms_confidences],
                        ms_joints_ref=ms_joints_ref,
                        heatmaps=heatmaps)

        elif phase == 'train':
            return dict(ms_confidences=ms_confidences,
                        ms_joints=ms_joints,
                        ms_joints_ref=ms_joints_ref,
                        heatmaps=heatmaps)
        else:
            raise NotImplementedError
