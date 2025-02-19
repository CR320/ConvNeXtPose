import torch
import torch.nn as nn
import numpy as np
from scipy.optimize import linear_sum_assignment

INF = 1e5


class BinaryFocalCost:
    def __init__(self, gamma=2, alpha=0.25, weight=1.0, eps=1e-5):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.weight = weight
        self.eps = eps

    def __call__(self, srcs, tgts):
        preds = srcs['confidences'].sigmoid()
        num_gts = tgts['num_tgts']
        neg_cost = -(1 - preds + self.eps).log() * (1 - self.alpha) * preds.pow(self.gamma)
        pos_cost = -(preds + self.eps).log() * self.alpha * (1 - preds).pow(self.gamma)
        ctr_cost = (pos_cost - neg_cost).expand(-1, num_gts.sum())

        return ctr_cost * self.weight


class OksCost:
    def __init__(self, use_bbox=False, weight=1.0, eps=1e-5):
        super().__init__()
        self.use_bbox = use_bbox
        self.weight = weight
        self.eps = eps

    def __call__(self, srcs, tgts):
        pred_joints = srcs['joints']
        target_joints, target_areas, visible_flags, sigmas = \
            tgts['joints'], tgts['areas'], tgts['visible_flags'], tgts['sigmas']

        scales = target_areas * 0.53 if self.use_bbox else target_areas
        vars = (sigmas * 2) ** 2
        d_square = torch.square(pred_joints[:, None, :, :] - target_joints[None, :, :, :]).sum(dim=-1)
        d_oks = torch.exp(-1 * d_square / (2 * scales[None, :, None] * vars[None, None, :] + self.eps))
        d_oks = d_oks * visible_flags[None, :, :]
        d_oks = d_oks.sum(dim=-1) / (visible_flags.sum(dim=-1)[None, :] + self.eps)

        return -1 * d_oks * self.weight


class JointL1Cost:
    def __init__(self, weight=1.0, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, srcs, tgts):
        pred_joints = srcs['joints']
        target_joints, target_areas, visible_flags = \
            tgts['joints'], tgts['areas'], tgts['visible_flags']

        d_l1 = torch.abs(pred_joints[:, None] - target_joints[None, :]).sum(dim=-1)
        d_l1 = d_l1 / (target_areas[None, :, None].sqrt() + self.eps)
        d_l1 = (d_l1 * visible_flags[None, :, :]).sum(dim=-1) / (visible_flags.sum(dim=-1)[None, :] + self.eps)

        return d_l1 * self.weight


class LimbL1Cost:
    def __init__(self, weight=1.0, eps=1e-5):
        super().__init__()
        self.weight = weight
        self.eps = eps

    def __call__(self, srcs, tgts):
        pred_joints = srcs['joints']
        target_joints, target_areas, visible_flags, limbs_table = \
            tgts['joints'], tgts['areas'], tgts['visible_flags'], tgts['limbs_table']

        src_idx = limbs_table[:, 0]
        dst_idx = limbs_table[:, 1]
        pred_limbs = pred_joints[:, src_idx] - pred_joints[:, dst_idx]
        target_limbs = target_joints[:, src_idx] - target_joints[:, dst_idx]
        limb_flags = visible_flags[:, src_idx] * visible_flags[:, dst_idx]

        d_l1 = torch.abs(pred_limbs[:, None] - target_limbs[None, :]).sum(dim=-1)
        d_l1 = d_l1 / (target_areas[None, :, None].sqrt() + self.eps)
        d_l1 = (d_l1 * limb_flags[None, :, :]).sum(dim=-1) / (limb_flags.sum(dim=-1)[None, :] + self.eps)

        return d_l1 * self.weight


class ScaleSelection(nn.Module):
    def __init__(self, output_size, num_scale):
        super().__init__()
        bounds = [output_size / 2 ** (num_scale - 1 - n) for n in range(num_scale)]
        self.register_buffer('bounds', torch.tensor(bounds, dtype=torch.float32, requires_grad=False))

    def __call__(self, cost_matrix, shapes, target_sizes):
        d = target_sizes[None, :] - self.bounds[:, None]

        ind = 0
        for i, shape in enumerate(shapes):
            step = shape[0] * shape[1]
            cost_matrix[:, ind:ind+step, d[i] > 0] = INF
            ind += step

        return cost_matrix


class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost, output_size=-1, num_scales=-1, with_constrain=True):
        """Creates the matcher

        Params:
            cost: This is the relative weights in the matching cost
        """
        super().__init__()
        self.cost_functions = cost
        if with_constrain and output_size > 0 and num_scales > 0:
            self.with_constrain = True
            self.scale_selection = ScaleSelection(output_size, num_scales)
        else:
            self.with_constrain = False

    def _get_src_permutation_ids(self, indices):
        # permute predictions following indices
        batch_ids = np.concatenate([np.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_ids = np.concatenate([src for (src, _) in indices])
        return batch_ids, src_ids

    def _get_tgt_permutation_ids(self, indices):
        # permute targets following indices
        batch_ids = np.concatenate([np.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_ids = np.concatenate([tgt for (_, tgt) in indices])
        return batch_ids, tgt_ids

    @torch.no_grad()
    def forward(self, predictions, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "centerness": Tensor of dim [batch_size, num_queries, 1] with the classification confidences
                 "offsets": Tensor of dim [batch_size, num_queries, 2] with the predicted offsets of center coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "centers": Tensor of dim [batch, max_number, 3] containing the target centers coordinates and flag

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        pred_confs, pred_joints = predictions['confidences'], predictions['joints']
        target_joints, target_areas, target_sizes, visible_flags, masks = \
            targets['joints'], targets['areas'], targets['sizes'], targets['visible_flags'], targets['masks']

        valid_flags = (visible_flags.sum(-1)) > 0
        num_tgts = valid_flags.sum(-1)

        # feed data dictionary
        batch_size, num_queries, num_joints, _ = pred_joints.shape
        srcs = dict(
            confidences=pred_confs.reshape(batch_size * num_queries, 1),
            joints=pred_joints.reshape(batch_size * num_queries, num_joints, 2),
        )
        tgts = dict(
            joints=torch.cat([target_joints[b][valid_flags[b]] for b in range(batch_size)]),
            areas=torch.cat([target_areas[b][valid_flags[b]] for b in range(batch_size)]),
            sizes=torch.cat([target_sizes[b][valid_flags[b]] for b in range(batch_size)]),
            visible_flags=torch.cat([visible_flags[b][valid_flags[b]] for b in range(batch_size)]),
            num_tgts=num_tgts,
            sigmas=targets['sigmas'],
            limbs_table=targets['limbs_table']
        )

        # compute cost matrix
        costs = dict()
        for item in self.cost_functions:
            costs[item] = self.cost_functions[item](srcs, tgts)

        num_gts = num_tgts.sum().item()
        C = sum(costs.values())

        # select scale
        if self.with_constrain:
            pred_shapes = predictions['shapes']
            C = self.scale_selection(C.reshape(batch_size, num_queries, num_gts),
                                     pred_shapes,
                                     tgts['sizes'])
            C = C.reshape(batch_size * num_queries, num_gts)

        # mask invalid region
        masks = masks.flatten()
        C[masks == 0] = INF
        C = C.reshape(batch_size, num_queries, num_gts)

        # hungarian match
        if num_gts == 0:
            return (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)), \
                   (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
        else:
            try:
                sizes = [int(num_tgts[b]) for b in range(batch_size)]
                indices = [linear_sum_assignment(c[i].cpu()) for i, c in enumerate(C.split(sizes, -1))]
                src_ids = self._get_src_permutation_ids(indices)
                tgt_ids = self._get_tgt_permutation_ids(indices)

                return src_ids, tgt_ids

            except ValueError:
                for item in costs:
                    is_nan = torch.isnan(costs[item])
                    if is_nan.sum() > 0:
                        inds = torch.where(is_nan)
                        print(item)
                        print(inds)
                exit(1)

            except Exception as r:
                print('Unknown exception: {}'.format(r))
                exit(1)
