import torch
import numpy as np
from models.utils import look_up_poses, look_up_joint_scores, pose_nms, project_to_ori_size, aggregate_multi_heatmaps


def parse_poses(confidences, joints, heatmaps, val_th, max_num_people):
    poses, ctr_scores = look_up_poses(confidences, joints, val_th, max_num_people)
    joint_scores = look_up_joint_scores(poses, heatmaps)

    return poses, joint_scores, ctr_scores


def parse_results(outputs_list, eval_cfg):
    confidences_list, joints_list, heatmaps_list = list(), list(), list()
    for outputs in outputs_list:
        confidences_list.append(outputs['ms_confidences'])
        joints_list.append(outputs['ms_joints_ref'])
        heatmaps_list.append(outputs['heatmaps'])

    if len(heatmaps_list) > 1:
        assert eval_cfg['test_scale_factors'][0] == 1
        heatmaps_list = aggregate_multi_heatmaps(heatmaps_list)

    poses_list, pose_scores_list = list(), list()
    for i, (confidences, joints, heatmaps) in enumerate(zip(confidences_list, joints_list, heatmaps_list)):
        poses, joint_scores, ctr_scores = parse_poses(confidences,
                                                      joints,
                                                      heatmaps,
                                                      eval_cfg['val_th'],
                                                      eval_cfg['max_num_people'])

        poses = poses * eval_cfg['ratio'] / eval_cfg['test_scale_factors'][i]
        pose_scores = ctr_scores[:, None] * joint_scores

        poses_list.append(poses)
        pose_scores_list.append(pose_scores)

    if len(poses_list) == 1:
        poses = poses_list[0].cpu().numpy()
        pose_scores = pose_scores_list[0].cpu().numpy()

    else:
        poses = torch.cat(poses_list, dim=0)
        pose_scores = torch.cat(pose_scores_list, dim=0)

        # filter redundant results in multi-scale test
        keep_inds = pose_nms(poses, pose_scores) if len(poses) > 1 else[]
        poses = poses[keep_inds].cpu().numpy()
        pose_scores = pose_scores[keep_inds].cpu().numpy()

    final_poses = project_to_ori_size(poses,
                                      np.array(eval_cfg['center']),
                                      np.array(eval_cfg['scale']),
                                      eval_cfg['base_size'])
    final_poses = np.concatenate([final_poses, pose_scores[:, :, None]], axis=-1)

    results = dict(
        poses=[final_poses[i] for i in range(len(final_poses))],
        scores=pose_scores.mean(axis=-1).tolist()
    )

    return results
