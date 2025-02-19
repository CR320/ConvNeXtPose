import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributed as dist
from timm.models.layers import DropPath
from collections import OrderedDict


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps
        self.norm_shape = (num_channels,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1).contiguous()
        x = F.layer_norm(x, self.norm_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2).contiguous()


class BasicBlock(nn.Module):
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
        assert expansion == 1
        assert out_channels % expansion == 0
        mid_channels = out_channels // expansion

        self.bn1 = eval(norm)(mid_channels)
        self.bn2 = eval(norm)(out_channels)

        self.conv1 = nn.Conv2d(in_channels, mid_channels, 3, stride, dilation, dilation=dilation, bias=False)
        self.conv2 = nn.Conv2d(mid_channels, out_channels, 3, padding=1, bias=False)

        self.act_layer = eval(act)(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_layer(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.act_layer(out)

        return out


class Bottleneck(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 expansion=4,
                 stride=1,
                 dilation=1,
                 downsample=None,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expansion = expansion
        assert out_channels % expansion == 0
        self.mid_channels = out_channels // expansion
        self.stride = stride
        self.dilation = dilation

        self.bn1 = eval(norm)(self.mid_channels)
        self.bn2 = eval(norm)(self.mid_channels)
        self.bn3 = eval(norm)(out_channels)

        self.conv1 = nn.Conv2d(in_channels, self.mid_channels, 1, bias=False)
        self.conv2 = nn.Conv2d(self.mid_channels, self.mid_channels, 3, stride, dilation, dilation=dilation, bias=False)
        self.conv3 = nn.Conv2d(self.mid_channels, out_channels, kernel_size=1, bias=False)

        self.act_layer = eval(act)(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        """Forward function."""

        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act_layer(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act_layer(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        out = self.act_layer(out)

        return out


class HRModule(nn.Module):
    """High-Resolution Module for HRNet.

    In this module, every branch has 4 BasicBlocks/Bottlenecks. Fusion/Exchange
    is in this module.
    """

    def __init__(self,
                 num_branches,
                 block,
                 num_blocks,
                 num_channels,
                 multiscale_output=True,
                 drop_rate=0.,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU',
                 upsample_cfg=dict(mode='bilinear', align_corners=None)):

        # Protect mutable default arguments
        super().__init__()
        self.multiscale_output = multiscale_output
        self.num_branches = num_branches
        self.num_channels = num_channels
        self.act = act
        self.norm = norm
        self.upsample_cfg = upsample_cfg

        self.branches = self._make_branches(num_branches, block, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.act_layer = eval(act)(inplace=True)

        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0. else nn.Identity()

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels):
        """Make one branch."""

        layers = list()
        layers.append(block(num_channels[branch_index],
                            num_channels[branch_index],
                            norm=self.norm,
                            act=self.act))

        for _ in range(1, num_blocks[branch_index]):
            layers.append(block(num_channels[branch_index],
                                num_channels[branch_index],
                                norm=self.norm,
                                act=self.act))

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        """Make branches."""
        branches = []

        for i in range(num_branches):
            branches.append(self._make_one_branch(i, block, num_blocks, num_channels))

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        """Make fuse layer."""
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        in_channels = self.num_channels
        fuse_layers = []
        num_out_branches = num_branches if self.multiscale_output else 1

        for i in range(num_out_branches):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(in_channels[j], in_channels[i], 1, 1, 0, bias=False),
                            eval(self.norm)(in_channels[i]),
                            nn.Upsample(scale_factor=2**(j - i), **self.upsample_cfg))
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv_downsamples = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], in_channels[i], 3, 2, 1, bias=False),
                                    eval(self.norm)(in_channels[i]))
                            )
                        else:
                            conv_downsamples.append(
                                nn.Sequential(
                                    nn.Conv2d(in_channels[j], in_channels[j], 3, 2, 1,bias=False),
                                    eval(self.norm)(in_channels[j]),
                                    eval(self.act)(inplace=True))
                            )
                    fuse_layer.append(nn.Sequential(*conv_downsamples))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def forward(self, x):
        """Forward function."""
        if self.num_branches == 1:
            return [self.drop_path(self.branches[0](x[0]))]

        for i in range(self.num_branches):
            x[i] = self.drop_path(self.branches[i](x[i]))

        x_fuse = []
        for i in range(len(self.fuse_layers)):
            y = 0
            for j in range(self.num_branches):
                if i == j:
                    y += x[j]
                else:
                    y += self.fuse_layers[i][j](x[j])
            x_fuse.append(self.dropout(self.act_layer(y)))

        return x_fuse


class TransitionModule(nn.Module):
    def __init__(self,
                 base_channels,
                 hr_channels,
                 ratio,
                 type,
                 norm='nn.BatchNorm2d',
                 act='nn.ReLU'):
        super().__init__()
        assert type in ['b2h', 'h2b']
        self.type = type
        self.ratio = ratio
        if type == 'b2h':
            self.conv = nn.Sequential(
                nn.Conv2d(int(base_channels / (ratio ** 2)), hr_channels, 1, 1, 0, bias=False),
                eval(norm)(hr_channels),
            )
        else:
            self.conv = nn.Sequential(
                nn.Conv2d(hr_channels, int(base_channels / (ratio ** 2)), 1, 1, 0, bias=False),
                eval(norm)(int(base_channels / (ratio ** 2))),
            )
        self.act = eval(act)(inplace=True)

    @staticmethod
    def pixel_shuffle(x, ratio):
        B, C, H, W = x.shape
        if ratio > 1:
            interval = ratio
            assert interval % 2 == 0
            x = x.reshape(B, C // (interval * interval), interval, interval, H, W)
            x = einops.rearrange(x, ' b c m n h w -> b c h m w n').contiguous()
            x = x.reshape(B, -1, H * interval, W * interval)
        elif ratio < 1:
            interval = int(1 / ratio)
            assert interval % 2 == 0
            x = x.reshape(B, C, H // interval, interval, W // interval, interval)
            x = einops.rearrange(x, 'b c h m w n -> b c m n h w').contiguous()
            x = x.reshape(B, -1, H // interval, W // interval)
        return x

    def forward(self, x_list, x_base):
        if self.type == 'b2h':
            x_b2h = self.pixel_shuffle(x_base, self.ratio)
            x_b2h = self.conv(x_b2h)
            x_list.append(self.act(x_b2h))
        else:
            x_h2b = self.conv(x_list[-1])
            x_h2b = self.pixel_shuffle(x_h2b, 1 / self.ratio)
            x_base = self.act(x_h2b) + x_base

        return x_list, x_base


class HRAdapter(nn.Module):
    def __init__(self, transitions, hr_modules, norm, act, drop_rate=0., pre_weights=None):
        super().__init__()
        self.trans_b2h, self.trans_h2b = nn.ModuleList(), nn.ModuleList()
        for i, item in enumerate(transitions):
            self.trans_b2h.append(TransitionModule(**transitions[item], type='b2h', norm=norm, act=act))
            if i < len(transitions) - 1:
                self.trans_h2b.append(TransitionModule(**transitions[item], type='h2b', norm=norm, act=act))

        for i, item in enumerate(hr_modules):
            # if i == 0:
            #     out_channels = hr_modules[item]['num_channels'][0]
            #     self.stem = nn.Sequential(
            #         nn.Conv2d(3, out_channels // 2, 3, 2, 1, bias=False),
            #         eval(norm)(out_channels // 2),
            #         eval(act)(inplace=True),
            #         nn.Conv2d(out_channels // 2, out_channels, 3, 2, 1, bias=False),
            #         eval(norm)(out_channels),
            #         eval(act)(inplace=True)
            #     )
            #     param_dict = list(transitions.values())
            #     self.stem_trans = TransitionModule(**param_dict[i], type='h2b', norm=norm, act=act)
            self.add_module(
                name=item,
                module=self._make_stage(hr_modules[item], norm=norm, act=act, drop_rate=drop_rate)
            )

        in_channels = hr_modules['stage1']['num_channels'][0]
        output_channels = hr_modules['stage2']['num_channels'][0]
        self.stage1_trans = nn.Sequential(
            nn.Conv2d(in_channels, output_channels, 1, 1, 0, bias=False),
            eval(norm)(output_channels),
            eval(act)(inplace=True)
        )

        self.init_weights(pre_weights)

    def _make_stage(self, layer_config, norm, act, drop_rate):
        """Make stage."""
        num_modules = layer_config['num_modules']
        num_branches = layer_config['num_branches']
        num_blocks = layer_config['num_blocks']
        num_channels = layer_config['num_channels']

        hr_modules = []
        if num_branches == 1:
            input_channels = num_channels[0]
            output_channels = num_channels[0] * 4
            downsample = nn.Sequential(
                nn.Conv2d(input_channels, output_channels, 1, 1, 0, bias=False),
                eval(norm)(output_channels)
            )
            hr_modules.append(Bottleneck(input_channels, output_channels, downsample=downsample, norm=norm, act=act))
            for _ in range(1, num_blocks[0]):
                hr_modules.append(Bottleneck(output_channels, output_channels, norm=norm, act=act))
            hr_modules.append(nn.Sequential(
                nn.Conv2d(output_channels, input_channels, 1, 1, 0, bias=False),
                eval(norm)(input_channels),
                eval(act)(inplace=True)
            ))
        else:
            for i in range(num_modules):
                hr_modules.append(
                    HRModule(num_branches, BasicBlock, num_blocks, num_channels, norm=norm, act=act, drop_rate=drop_rate)
                )

        return nn.Sequential(*hr_modules)

    def init_weights(self, pre_weights):
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

        if pre_weights is not None:
            state_dict = torch.load(pre_weights, map_location='cpu')
            new_state_dict = OrderedDict()
            for k in state_dict:
                if k.startswith('layer1'):
                    new_k = k.replace('layer1.', 'stage1.')
                    if new_k in self.state_dict():
                        new_state_dict[new_k] = state_dict[k]

                elif k in self.state_dict():
                     new_state_dict[k] = state_dict[k]

            u, w = self.load_state_dict(new_state_dict, strict=False)

            initialized = False
            if dist.is_available():
                initialized = dist.is_initialized()
            if initialized:
                rank = dist.get_rank()
            else:
                rank = 0

            if rank == 0:
                print('HRAdapter: misaligned params during the loading of parameters: {} {}'.format(u, w))

    def forward(self):
        pass

    def stage_forward(self, x_list, stage_id):
        module = getattr(self, 'stage{}'.format(stage_id + 1))
        if stage_id == 0:
            x_list[0] = module(x_list[0])
        else:
            x_list = module(x_list)

        return x_list


class HRCoMer(nn.Module):
    def __init__(self, base_net, hr_adapter, num_stages=4):
        # Protect mutable default arguments
        super(HRCoMer, self).__init__()
        self.base_net = base_net
        self.hr_adapter = hr_adapter
        self.num_stages = num_stages

    def forward(self, x):
        # stem for base net
        x_base = self.base_net.stem_forward(x)

        # stages
        x_list = list()
        for i in range(self.num_stages):
            x_base = self.base_net.stage_forward(x_base, stage_id=i)

            x_list, x_base = self.hr_adapter.trans_b2h[i](x_list, x_base)

            x_list = self.hr_adapter.stage_forward(x_list, i)

            if i < self.num_stages - 1:
                x_list, x_base = self.hr_adapter.trans_h2b[i](x_list, x_base)

            if i == 0:
                x_list[0] = self.hr_adapter.stage1_trans(x_list[0])

        return x_list
