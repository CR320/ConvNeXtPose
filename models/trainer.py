import torch
import torch.nn as nn
import torch.nn.functional as F


class Trainer(nn.Module):
    def __init__(self,
                 model,
                 criterion,
                 matcher=None,
                 sigmas=None,
                 limbs_table=None):
        super().__init__()
        self.net = model
        self.matcher = matcher
        self.criterion = criterion
        self.sigmas = torch.from_numpy(sigmas) if sigmas is not None else None
        self.limbs_table = limbs_table

    def forward(self, data):
        outputs = self.net(data['image'], phase='train')

        # reshape predictions & masks
        masks = data['masks']
        re_confs, re_joints, re_joints_ref, re_masks = list(), list(), list(), list()
        pred_shapes = list()
        for confs_i, joints_i, joints_ref_i in zip(outputs['ms_confidences'],
                                                   outputs['ms_joints'],
                                                   outputs['ms_joints_ref']):
            B, H, W, K, _ = joints_i.shape
            pred_shapes.append((H, W))
            re_confs.append(confs_i.reshape(B, H * W, 1))
            re_joints.append(joints_i.reshape(B, H * W, K, 2))
            re_joints_ref.append(joints_ref_i.reshape(B, H * W, K, 2))
            re_masks.append(F.interpolate(masks.unsqueeze(1),
                                          size=(H, W),
                                          mode='bilinear',
                                          align_corners=False).reshape(B, H * W))
        re_confs = torch.cat(re_confs, dim=1)
        re_joints = torch.cat(re_joints, dim=1)
        re_joints_ref = torch.cat(re_joints_ref, dim=1)
        re_masks = torch.cat(re_masks, dim=1)

        # match
        predictions = dict(
            joints=re_joints,
            joints_refine=re_joints_ref,
            confidences=re_confs,
            heatmaps=outputs['heatmaps'],
            shapes=pred_shapes
        )
        targets = dict(
            joints=data['target_joints'][..., 0:2],
            visible_flags=data['target_joints'][..., 2] > 0,
            areas=data['target_areas'],
            sizes=data['target_sizes'],
            heatmaps=data['target_heatmaps'],
            masks=re_masks,
            hms_masks=data['masks'],
            sigmas=self.sigmas.type_as(data['masks']),
            limbs_table=self.limbs_table
        )
        src_ids, tgt_ids = self.matcher(predictions, targets)

        # calculate loss value
        losses = dict()
        for item in self.criterion:
            losses[item] = self.criterion[item](predictions, targets, src_ids=src_ids, tgt_ids=tgt_ids)

        return losses
