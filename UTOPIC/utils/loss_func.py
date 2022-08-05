import torch
import torch.nn as nn
import torch.nn.functional as F
from extensions.Chamfer3D.dist_chamfer_3D import chamfer_3DDist


class OverallLoss(nn.Module):
    def __init__(self):
        super(OverallLoss, self).__init__()
        self.permloss = PermLoss()
        self.clsloss = ClassifyLoss()
        self.cd1 = CDLoss_L1()
        self.cd2 = CDLoss_L1()
        self.cd3 = CDLoss_L1()
        self.cd4 = CDLoss_L1()
        self.cls_loss_prob = ClassifyLoss()
        self.kl_loss = nn.KLDivLoss(reduction='batchmean')

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns,
                pred_class, gt_class,
                src_gt, ref_gt, coarse_src, fine_src, coarse_ref, fine_ref, prob):
        perm_loss = self.permloss(pred_perm, gt_perm, pred_ns, gt_ns)
        cls_loss = self.clsloss(pred_class, gt_class, gt_ns)
        cd_coarse_src_loss = self.cd1(coarse_src, src_gt)
        cd_fine_src_loss = self.cd2(fine_src, src_gt)
        cd_coarse_ref_loss = self.cd3(coarse_ref, ref_gt)
        cd_fine_ref_loss = self.cd4(fine_ref, ref_gt)
        clsloss_prob = self.cls_loss_prob(prob, gt_class, gt_ns)
        klloss = self.kl_loss(torch.log(prob), gt_class) / pred_ns[0]

        loss_item = {
            'perm_loss': perm_loss,
            'overlap_loss': cls_loss,
            'c_s_cd_loss': cd_coarse_src_loss,
            'f_s_cd_loss': cd_fine_src_loss,
            'c_r_cd_loss': cd_coarse_ref_loss,
            'f_r_cd_loss': cd_fine_ref_loss,
            'overlap_prob_loss': clsloss_prob,
            'kl_loss': klloss
        }

        return loss_item


class PermLoss(nn.Module):
    """
    Cross entropy loss between two permutations.
    cal avg loss, 平均每个节点的loss
    """

    def __init__(self):
        super(PermLoss, self).__init__()

    def forward(self, pred_perm, gt_perm, pred_ns, gt_ns):
        batch_num = pred_perm.shape[0]

        pred_perm = pred_perm.to(dtype=torch.float32)

        assert torch.all((pred_perm >= 0) * (pred_perm <= 1))
        assert torch.all((gt_perm >= 0) * (gt_perm <= 1))

        loss = torch.tensor(0.).to(pred_perm.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_perm[b, :pred_ns[b], :gt_ns[b]],
                gt_perm[b, :pred_ns[b], :gt_ns[b]],
                reduction='sum')
            n_sum += pred_ns[b].to(n_sum.dtype).to(pred_perm.device)

        return loss / n_sum


class ClassifyLoss(nn.Module):
    """
    Cross entropy loss between two classify.
    cal avg loss, 平均每个节点的loss
    """

    def __init__(self):
        super(ClassifyLoss, self).__init__()

    def forward(self, pred_class, gt_class, gt_ns):
        batch_num = pred_class.shape[0]

        pred_class = pred_class.to(dtype=torch.float32)

        assert torch.all((pred_class >= 0) * (pred_class <= 1))
        assert torch.all((gt_class >= 0) * (gt_class <= 1))

        loss = torch.tensor(0.).to(pred_class.device)
        n_sum = torch.zeros_like(loss)
        for b in range(batch_num):
            loss += F.binary_cross_entropy(
                pred_class[b, :gt_ns[b], :],
                gt_class[b, :gt_ns[b], :],
                reduction='sum')
            n_sum += gt_ns[b].to(n_sum.dtype).to(pred_class.device)

        return loss / n_sum


class CDLoss_L1(nn.Module):

    def __init__(self):
        super(CDLoss_L1, self).__init__()
        self.chamfer_dist = chamfer_3DDist()

    def forward(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)
        d1 = torch.mean(torch.sqrt(d1))
        d2 = torch.mean(torch.sqrt(d2))

        return (d1 + d2) / 2


class CDLoss_L2(nn.Module):

    def __init__(self):
        super(CDLoss_L2, self).__init__()
        self.chamfer_dist = chamfer_3DDist()

    def forward(self, p1, p2):
        d1, d2, _, _ = self.chamfer_dist(p1, p2)

        return torch.mean(d1) + torch.mean(d2)
