import torch
import torch.nn.functional as F
import numpy as np
from datasets.pipeline.utils import get_affine_transform, warp_affine_joints


def get_locations(output_h, output_w, device):
    shifts_x = torch.arange(
        0.5, output_w + 0.5, step=1,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0.5, output_h + 0.5, step=1,
        dtype=torch.float32, device=device
    )
    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = shift_x.reshape(-1)
    shift_y = shift_y.reshape(-1)
    locations = torch.stack((shift_x, shift_y), dim=1)

    return locations


def aggregate_multi_heatmaps(heatmaps_list):
    H, W = heatmaps_list[0].shape[-2:]
    new_hms_list = [heatmaps_list[0]]
    for hms in heatmaps_list[1:]:
        new_hms_list.append(
            F.interpolate(hms, (H, W), mode='bilinear', align_corners=False)
        )

    agg_heatmaps = sum(new_hms_list) / len(new_hms_list)

    agg_heatmaps_list = list()
    for i, hms in enumerate(heatmaps_list):
        if i == 0:
            agg_heatmaps_list.append(agg_heatmaps)
        else:
            H, W = hms.shape[-2:]
            agg_heatmaps_list.append(F.interpolate(agg_heatmaps, (H, W), mode='bilinear', align_corners=False))

    return agg_heatmaps_list


def look_up_poses(confidences, joints, val_th, max_num_people):
    # reshape predictions
    re_confs, re_joints = list(), list()
    for confs_i, joints_i in zip(confidences, joints):
        B, H, W, K, _ = joints_i.shape
        re_confs.append(confs_i.reshape(B, H * W, 1))
        re_joints.append(joints_i.reshape(B, H * W, K, 2))
    confidences = torch.cat(re_confs, dim=1)
    joints = torch.cat(re_joints, dim=1)

    # look up topK confidences
    confidences = confidences.flatten()
    ctr_scores, ctr_indices = confidences.topk(max_num_people)

    # filter low-score candidates with threshold
    is_valid = ctr_scores > val_th
    ctr_indices = ctr_indices[is_valid]
    ctr_scores = ctr_scores[is_valid]

    # look up poses
    joints = joints.view(len(confidences), -1, 2)
    poses = joints[ctr_indices]

    return poses, ctr_scores


def look_up_joint_scores(poses, heatmaps):
    _, K, H, W = heatmaps.shape
    x_inds = torch.clamp(torch.floor(poses[..., 0]), 0, W-1).long()
    y_inds = torch.clamp(torch.floor(poses[..., 1]), 0, H-1).long()
    reshaped_hms = heatmaps.view(K, H * W).transpose(0, 1)
    pose_scores = torch.gather(reshaped_hms, 0, y_inds * W + x_inds)

    return pose_scores


def cal_area_2_torch(v):
    w = torch.max(v[:, :, 0], -1)[0] - torch.min(v[:, :, 0], -1)[0]
    h = torch.max(v[:, :, 1], -1)[0] - torch.min(v[:, :, 1], -1)[0]
    return w * w + h * h


def pose_nms(pose_coords, joint_scores, nms_th=0.05, nms_num_th=8):
    num_people, num_joints, _ = pose_coords.shape
    pose_area = cal_area_2_torch(pose_coords)[:,None].repeat(1, num_people*num_joints)
    pose_area = pose_area.reshape(num_people, num_people, num_joints)
    mean_scores = joint_scores.mean(dim=-1)

    pose_diff = pose_coords[:, None, :, :] - pose_coords
    pose_diff.pow_(2)
    pose_dist = pose_diff.sum(3)
    pose_dist.sqrt_()
    pose_thre = nms_th * torch.sqrt(pose_area)
    pose_dist = (pose_dist < pose_thre).sum(2)
    nms_pose = pose_dist > nms_num_th

    ignored_pose_inds = []
    keep_pose_inds = []
    for i in range(nms_pose.shape[0]):
        if i in ignored_pose_inds:
            continue
        keep_inds = nms_pose[i].nonzero().cpu().numpy()
        keep_inds = [list(kind)[0] for kind in keep_inds]
        keep_scores = mean_scores[keep_inds]
        ind = torch.argmax(keep_scores)
        keep_ind = keep_inds[ind]
        if keep_ind in ignored_pose_inds:
            continue
        keep_pose_inds += [keep_ind]
        ignored_pose_inds += list(set(keep_inds)-set(ignored_pose_inds))

    return keep_pose_inds


def project_to_ori_size(poses, img_center, img_scale, img_base_size):
    trans = get_affine_transform(img_center, img_scale, 0, img_base_size, inv=True)
    target_poses = np.zeros_like(poses)

    for p in range(len(poses)):
        for k in range(len(poses[p])):
            target_poses[p, k] = warp_affine_joints(poses[p, k], trans)

    return target_poses
