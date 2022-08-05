import os
import torch
import numpy as np
from typing import Dict
from pathlib import Path
from utils import dcputil
from utils.se3 import transform
from models.correspondSlover import SVDslover
from utils.config import cfg


def to_numpy(tensor):
    """Wrapper around .detach().cpu().numpy() """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        return tensor
    else:
        raise NotImplementedError


def matching_accuracy(pmat_pred, pmat_gt, ns):
    """
    Matching Accuracy between predicted permutation matrix and ground truth permutation matrix.
    :param pmat_pred: predicted permutation matrix
    :param pmat_gt: ground truth permutation matrix
    :param ns: number of exact pairs
    :return: matching accuracy, matched num of pairs, total num of pairs
    """
    device = pmat_pred.device
    batch_num = pmat_pred.shape[0]

    pmat_gt = pmat_gt.to(device)

    assert torch.all((pmat_pred == 0) + (pmat_pred == 1)), 'pmat_pred can noly contain 0/1 elements.'
    assert torch.all((pmat_gt == 0) + (pmat_gt == 1)), 'pmat_gt should noly contain 0/1 elements.'
    assert torch.all(torch.sum(pmat_pred, dim=-1) <= 1) and torch.all(torch.sum(pmat_pred, dim=-2) <= 1)
    assert torch.all(torch.sum(pmat_gt, dim=-1) <= 1) and torch.all(torch.sum(pmat_gt, dim=-2) <= 1)

    match_num_list = []
    gt_num_list = []
    pred_num_list = []
    acc_gt = []
    acc_pred = []
    for b in range(batch_num):
        match_num = torch.sum(pmat_pred[b, :ns[b]] * pmat_gt[b, :ns[b]]) + 1e-8
        gt_num = torch.sum(pmat_gt[b, :ns[b]]) + 1e-8
        pred_num = torch.sum(pmat_pred[b, :ns[b]]) + 1e-8
        match_num_list.append(match_num.cpu().numpy())
        gt_num_list.append(gt_num.cpu().numpy())
        pred_num_list.append(pred_num.cpu().numpy())
        acc_gt.append((match_num / gt_num).cpu().numpy())
        acc_pred.append((match_num / pred_num).cpu().numpy())

    return {'acc_gt': np.array(acc_gt),
            'acc_pred': np.array(acc_pred),
            'match_num': np.array(match_num_list),
            'gt_num': np.array(gt_num_list),
            'pred_num': np.array(pred_num_list)}


def calcorrespondpc(pmat_pred, pc2_gt, ref_overlap):
    pc2 = torch.zeros_like(pc2_gt).to(pc2_gt)
    overlap = torch.zeros_like(ref_overlap).to(ref_overlap)
    pmat_pred_index = np.zeros((pc2_gt.shape[0], pc2_gt.shape[1]), dtype=int)
    for i in range(pmat_pred.shape[0]):
        pmat_predi_index1 = torch.where(pmat_pred[i])
        pmat_predi_index00 = torch.where(torch.sum(pmat_pred[i], dim=0) == 0)[0]  # n row sum->1，1024
        pmat_predi_index01 = torch.where(torch.sum(pmat_pred[i], dim=1) == 0)[0]  # n col sum->1024,1
        pc2[i, torch.cat((pmat_predi_index1[0], pmat_predi_index01))] = \
            pc2_gt[i, torch.cat((pmat_predi_index1[1], pmat_predi_index00))]
        overlap[i, torch.cat((pmat_predi_index1[0], pmat_predi_index01))] = \
            ref_overlap[i, torch.cat((pmat_predi_index1[1], pmat_predi_index00))]
        pmat_pred_index[i, pmat_predi_index1[0].cpu().numpy()] = pmat_predi_index1[1].cpu().numpy()
        pmat_pred_index[i, pmat_predi_index01.cpu().numpy()] = pmat_predi_index00.cpu().numpy()
    return pc2, overlap, pmat_pred_index


def square_distance(src, dst):
    return torch.sum((src[:, :, None, :] - dst[:, None, :, :]) ** 2, dim=-1)


def compute_transform(s_perm_mat, P1_gt, P2_gt, src_overlap, ref_overlap):
    corr_P2_gt, corr_ref_overlap, _ = calcorrespondpc(s_perm_mat, P2_gt, ref_overlap)
    weights = src_overlap * corr_ref_overlap
    R_pre, T_pre = SVDslover(P1_gt.clone(), corr_P2_gt, s_perm_mat, weights=weights)
    return R_pre, T_pre


def compute_metrics(s_perm_mat, P1_gt, P2_gt, R_gt, T_gt, src_overlap, ref_overlap):
    # compute R,t
    R_pre, T_pre = compute_transform(s_perm_mat, P1_gt, P2_gt, src_overlap, ref_overlap)

    r_pre_euler_deg = dcputil.npmat2euler(R_pre.detach().cpu().numpy(), seq='xyz')
    r_gt_euler_deg = dcputil.npmat2euler(R_gt.detach().cpu().numpy(), seq='xyz')
    r_mse = np.mean((r_gt_euler_deg - r_pre_euler_deg) ** 2, axis=1)
    r_mae = np.mean(np.abs(r_gt_euler_deg - r_pre_euler_deg), axis=1)
    t_mse = torch.mean((T_gt - T_pre) ** 2, dim=1)
    t_mae = torch.mean(torch.abs(T_gt - T_pre), dim=1)

    # Rotation, translation errors (isotropic)
    # i.e. doesn't depend on error direction, which is more representative of the actual error
    concatenated = dcputil.concatenate(dcputil.inverse(R_gt.detach().cpu().numpy(), T_gt.detach().cpu().numpy()),
                                       np.concatenate(
                                           [R_pre.detach().cpu().numpy(), T_pre.unsqueeze(-1).detach().cpu().numpy()],
                                           axis=-1))
    rot_trace = concatenated[:, 0, 0] + concatenated[:, 1, 1] + concatenated[:, 2, 2]
    residual_rotdeg = torch.acos(torch.clamp(0.5 * (rot_trace - 1), min=-1.0, max=1.0)) * 180.0 / np.pi
    residual_transmag = concatenated[:, :, 3].norm(dim=-1)

    # Chamfer distance
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_transformed, P2_gt), dim=-1)[0]
    dist_ref = torch.min(square_distance(P2_gt, P1_transformed), dim=-1)[0]
    chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # Source distance
    P1_pre_trans = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                              P1_gt.detach().cpu().numpy())).to(P1_gt)
    P1_gt_trans = torch.from_numpy(transform(torch.cat((R_gt, T_gt[:, :, None]), dim=2).detach().cpu().numpy(),
                                             P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(square_distance(P1_pre_trans, P1_gt_trans), dim=-1)[0]
    presrc_dist = torch.mean(dist_src, dim=1)

    # Clip Chamfer distance
    clip_val = torch.Tensor([0.1]).cuda()
    P1_transformed = torch.from_numpy(transform(torch.cat((R_pre, T_pre[:, :, None]), dim=2).detach().cpu().numpy(),
                                                P1_gt.detach().cpu().numpy())).to(P1_gt)
    dist_src = torch.min(torch.min(torch.sqrt(square_distance(P1_transformed, P2_gt)), dim=-1)[0], clip_val)
    dist_ref = torch.min(torch.min(torch.sqrt(square_distance(P2_gt, P1_transformed)), dim=-1)[0], clip_val)
    clip_chamfer_dist = torch.mean(dist_src, dim=1) + torch.mean(dist_ref, dim=1)

    # correspondence distance
    P2_gt_copy, _, _ = calcorrespondpc(s_perm_mat, P2_gt.detach(), ref_overlap)
    inlier_src = torch.sum(s_perm_mat, axis=-1)[:, :, None]
    P1_gt_trans_corr = P1_gt_trans.mul(inlier_src)
    P2_gt_copy_coor = P2_gt_copy.mul(inlier_src)
    correspond_dis = torch.sqrt(torch.sum((P1_gt_trans_corr - P2_gt_copy_coor) ** 2, dim=-1, keepdim=True))
    correspond_dis[inlier_src == 0] = np.nan

    metrics = {'r_mse': r_mse,
               'r_mae': r_mae,
               't_mse': to_numpy(t_mse),
               't_mae': to_numpy(t_mae),
               'err_r_deg': to_numpy(residual_rotdeg),
               'err_t': to_numpy(residual_transmag),
               'chamfer_dist': to_numpy(chamfer_dist),
               'pcab_dist': to_numpy(presrc_dist),
               'clip_chamfer_dist': to_numpy(clip_chamfer_dist),
               'pre_transform': np.concatenate((to_numpy(R_pre), to_numpy(T_pre)[:, :, None]), axis=2),
               'gt_transform': np.concatenate((to_numpy(R_gt), to_numpy(T_gt)[:, :, None]), axis=2),
               'cpd_dis_nomean': to_numpy(correspond_dis)}

    return metrics


def summarize_metrics(metrics):
    """Summaries computed metrices by taking mean over all data instances"""
    summarized = {}
    for k in metrics:
        if k.endswith('mse'):
            summarized[k[:-3] + 'rmse'] = np.sqrt(np.mean(metrics[k]))
        elif k.startswith('err'):
            summarized[k + '_mean'] = np.mean(metrics[k])
            summarized[k + '_rmse'] = np.sqrt(np.mean(metrics[k] ** 2))
        elif k.endswith('nomean'):
            summarized[k] = metrics[k]
        else:
            summarized[k] = np.mean(metrics[k])

    return summarized


def print_metrics(summary_metrics: Dict, title: str = 'Metrics'):
    """Prints out formated metrics to logger"""

    print('=' * (len(title) + 1))
    print(title + ':')

    print('DeepCP metrics: {:.4f}(rot-rmse) | {:.4f}(rot-mae) | {:.4g}(trans-rmse) | {:.4g}(trans-mae)'.
          format(summary_metrics['r_rmse'], summary_metrics['r_mae'],
                 summary_metrics['t_rmse'], summary_metrics['t_mae'],
                 ))
    print('Rotation error: {:.4f}(deg, mean) | {:.4f}(deg, rmse)'.
          format(summary_metrics['err_r_deg_mean'],
                 summary_metrics['err_r_deg_rmse']))
    print('Translation error: {:.4g}(mean) | {:.4g}(rmse)'.
          format(summary_metrics['err_t_mean'],
                 summary_metrics['err_t_rmse']))
