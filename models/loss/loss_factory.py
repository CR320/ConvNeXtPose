import torch
import torch.nn as nn
import torch.nn.functional as F


class BinaryFocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.25, weight=1.0):
        """ Create the criterion.
        Parameters:
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight

    def forward(self, predictions, targets, src_ids, tgt_ids=None):
        pred_confs = predictions['confidences']
        masks = targets['masks']
        target_confs = torch.zeros_like(pred_confs)
        target_confs[src_ids] = 1.0

        pred_confs = pred_confs[masks > 0]
        target_confs = target_confs[masks > 0]

        # binary focal loss
        _pred_confs = pred_confs.sigmoid()
        pt = (1 - _pred_confs) * target_confs + _pred_confs * (1 - target_confs)
        focal_weight = (self.alpha * target_confs + (1 - self.alpha) * (1 - target_confs)) * pt.pow(self.gamma)
        loss = F.binary_cross_entropy_with_logits(pred_confs, target_confs, reduction='none') * focal_weight
        loss = loss.mean()

        return loss * self.weight


class OksLoss(nn.Module):
    def __init__(self, use_bbox=False, weight=1.0, eps=1e-5):
        super().__init__()
        self.use_bbox = use_bbox
        self.weight = weight
        self.eps = eps

    def forward(self, predictions, targets, src_ids, tgt_ids):
        pred_joints = predictions['joints_refine']
        target_joints, target_areas, visible_flags, sigmas = \
            targets['joints'], targets['areas'], targets['visible_flags'], targets['sigmas']

        pred_joints = pred_joints[src_ids]
        target_joints = target_joints[tgt_ids]
        target_areas = target_areas[tgt_ids]
        visible_flags = visible_flags[tgt_ids]

        if len(visible_flags) > 0:
            scales = target_areas * 0.53 if self.use_bbox else target_areas
            vars = (sigmas * 2)**2
            d_square = torch.square(pred_joints - target_joints).sum(dim=-1)
            oks = torch.exp(-1 * d_square / (2 * scales[:, None] * vars[None, :] + self.eps))
            oks = (oks * visible_flags).sum(dim=-1) / (visible_flags.sum(dim=-1) + self.eps)
            loss = 1 - oks.mean()
        else:
            dummy_preds = predictions['joints_refine'][0, 0]
            dummy_targets = torch.zeros_like(dummy_preds)
            dummy_loss = torch.abs(dummy_preds - dummy_targets) * 0
            loss = dummy_loss.mean()

        return loss * self.weight


class JointL1Loss(nn.Module):
    def __init__(self, weight=1.0, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self,  predictions, targets, src_ids, tgt_ids):
        pred_joints = predictions['joints']
        target_joints, target_areas, visible_flags, sigmas = \
            targets['joints'], targets['areas'], targets['visible_flags'], targets['sigmas']

        pred_joints = pred_joints[src_ids]
        target_joints = target_joints[tgt_ids]
        target_areas = target_areas[tgt_ids]
        visible_flags = visible_flags[tgt_ids]

        if len(visible_flags) > 0:
            loss = torch.abs(pred_joints - target_joints).sum(dim=-1)
            loss = loss / (target_areas.sqrt()[:, None] + self.eps)
            loss = (loss * visible_flags).sum() / (visible_flags.sum() + self.eps)
        else:
            dummy_preds = predictions['joints'][0, 0]
            dummy_targets = torch.zeros_like(dummy_preds)
            dummy_loss = torch.abs(dummy_preds - dummy_targets) * 0
            loss = dummy_loss.mean()

        return loss * self.weight


class LimbL1Loss(nn.Module):
    def __init__(self, weight=1.0, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def forward(self, predictions, targets, src_ids, tgt_ids):
        pred_joints = predictions['joints']
        target_joints, target_areas, visible_flags, limbs_table = \
            targets['joints'], targets['areas'], targets['visible_flags'], targets['limbs_table']

        pred_joints = pred_joints[src_ids]
        target_joints = target_joints[tgt_ids]
        target_areas = target_areas[tgt_ids]
        visible_flags = visible_flags[tgt_ids]

        src_idx = limbs_table[:, 0]
        dst_idx = limbs_table[:, 1]
        pred_limbs = pred_joints[:, src_idx] - pred_joints[:, dst_idx]
        target_limbs = target_joints[:, src_idx] - target_joints[:, dst_idx]
        limb_flags = visible_flags[:, src_idx] * visible_flags[:, dst_idx]

        if len(visible_flags) > 0:
            loss = torch.abs(pred_limbs - target_limbs).sum(dim=-1)
            loss = loss / (target_areas.sqrt()[:, None] + self.eps)
            loss = (loss * limb_flags).sum() / (limb_flags.sum() + self.eps)
        else:
            dummy_preds = predictions['joints'][0, 0]
            dummy_targets = torch.zeros_like(dummy_preds)
            dummy_loss = torch.abs(dummy_preds - dummy_targets) * 0
            loss = dummy_loss.mean()

        return loss * self.weight


class HeatmapFocalLoss(nn.Module):
    """CenterFocalLoss is a variant of focal loss.

    More details can be found in the `paper
    <https://arxiv.org/abs/1808.01244>`_

    Args:
        weight (float): Loss weight of current loss.
    """

    def __init__(self, weight=1.0, alpha=2., beta=4.):
        super(HeatmapFocalLoss, self).__init__()
        self.weight = weight
        self.alpha = alpha
        self.beta = beta

    def forward(self, predictions, targets, src_ids=None, tgt_ids=None):
        pred = predictions['heatmaps']
        gt, masks = targets['heatmaps'], targets['hms_masks']

        pos_inds = gt.eq(1).float()
        if masks is None:
            neg_inds = gt.lt(1).float()
        else:
            neg_inds = gt.lt(1).float() * masks.eq(1).float()[:, None, :, :]

        neg_weights = torch.pow(1 - gt, self.beta)

        loss = 0

        pos_loss = torch.log(pred) * torch.pow(1 - pred, self.alpha) * pos_inds
        neg_loss = torch.log(1 - pred) * torch.pow(pred, self.alpha) * neg_weights * neg_inds

        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

        loss = self.weight * loss

        return loss
