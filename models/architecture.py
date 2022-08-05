import torch
import torch.nn as nn
import numpy as np
import itertools

from models.affinity_layer import Affinity
from models.adaptive import AdaptiveFCN
from models.transformer import TransformerModule
from models.pcn import PCN

from utils.config import cfg


def sinkhorn_rpm(log_alpha, n_iters: int = 5, slack: bool = True, eps: float = -1) -> torch.Tensor:
    """ Run sinkhorn iterations to generate a near doubly stochastic matrix, where each row or column sum to <=1

    Args:
        log_alpha: log of positive matrix to apply sinkhorn normalization (B, J, K)
        n_iters (int): Number of normalization iterations
        slack (bool): Whether to include slack row and column
        eps: eps for early termination (Used only for handcrafted RPM). Set to negative to disable.

    Returns:
        log(perm_matrix): Doubly stochastic matrix (B, J, K)

    Modified from original source taken from:
        Learning Latent Permutations with Gumbel-Sinkhorn Networks
        https://github.com/HeddaCohenIndelman/Learning-Gumbel-Sinkhorn-Permutations-w-Pytorch
    """

    # Sinkhorn iterations
    prev_alpha = None
    if slack:
        zero_pad = nn.ZeroPad2d((0, 1, 0, 1))
        log_alpha_padded = zero_pad(log_alpha[:, None, :, :])

        log_alpha_padded = torch.squeeze(log_alpha_padded, dim=1)

        for i in range(n_iters):
            # Row normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :-1, :] - (torch.logsumexp(log_alpha_padded[:, :-1, :], dim=2, keepdim=True)),
                log_alpha_padded[:, -1, None, :]),  # Don't normalize last row
                dim=1)

            # Column normalization
            log_alpha_padded = torch.cat((
                log_alpha_padded[:, :, :-1] - (torch.logsumexp(log_alpha_padded[:, :, :-1], dim=1, keepdim=True)),
                log_alpha_padded[:, :, -1, None]),  # Don't normalize last column
                dim=2)

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha_padded[:, :-1, :-1]) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha_padded[:, :-1, :-1]).clone()

        log_alpha = log_alpha_padded[:, :-1, :-1]
    else:
        for i in range(n_iters):
            # Row normalization (i.e. each row sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=2, keepdim=True))

            # Column normalization (i.e. each column sum to 1)
            log_alpha = log_alpha - (torch.logsumexp(log_alpha, dim=1, keepdim=True))

            if eps > 0:
                if prev_alpha is not None:
                    abs_dev = torch.abs(torch.exp(log_alpha) - prev_alpha)
                    if torch.max(torch.sum(abs_dev, dim=[1, 2])) < eps:
                        break
                prev_alpha = torch.exp(log_alpha).clone()

    return log_alpha


class PipeLine(nn.Module):
    def __init__(self):
        super(PipeLine, self).__init__()
        self.embed_dim = cfg.MODEL.FEATURE_EMBED_CHANNEL
        self.pointfeaturer = AdaptiveFCN(cfg.MODEL.NEIGHBORSNUM, self.embed_dim)
        self.transformer1 = TransformerModule(2*self.embed_dim, self.embed_dim, self.embed_dim // 2, 4,
                                              ['self', 'cross', 'self', 'cross', 'self', 'cross'])
        self.transformer2 = TransformerModule(self.embed_dim, self.embed_dim, self.embed_dim // 2, 4,
                                              ['self', 'cross', 'self', 'cross', 'self', 'cross'])
        self.pcn = PCN(2048, 2*self.embed_dim, 2)
        self.aff1 = Affinity(self.embed_dim)
        self.aff2 = Affinity(self.embed_dim)
        self.inst1 = nn.InstanceNorm2d(1, affine=True)
        self.inst2 = nn.InstanceNorm2d(1, affine=True)
        self.cross = nn.Linear(2*self.embed_dim, self.embed_dim)
        self.proj = nn.Linear(self.embed_dim, 1)
        self.sigmoid = nn.Sigmoid()

        self.proj_mean = nn.Linear(self.embed_dim, 1)
        self.proj_std = nn.Linear(self.embed_dim, 1)

        self.tri_k = 2

    def getTriLength(self, points, pairs):
        batch, num, num_pair, _ = pairs.shape
        tri_len = torch.zeros((batch * num, num_pair, 3), dtype=torch.float32).cuda()
        temp = (torch.arange(batch) * num).reshape(batch, 1, 1, 1).repeat(1, num, num_pair, 3).cuda()
        pairs = pairs + temp
        points = points.reshape(-1, 3)
        pairs = pairs.reshape(-1, num_pair, 3)
        tri_len[:, :, 0] = torch.sqrt(torch.sum(((points[pairs[:, :, 0], :] - points[pairs[:, :, 1], :]) ** 2), dim=-1))
        tri_len[:, :, 1] = torch.sqrt(torch.sum(((points[pairs[:, :, 1], :] - points[pairs[:, :, 2], :]) ** 2), dim=-1))
        tri_len[:, :, 2] = torch.sqrt(torch.sum(((points[pairs[:, :, 0], :] - points[pairs[:, :, 2], :]) ** 2), dim=-1))
        tri_len = tri_len.reshape(batch, num, num_pair, 3)
        tri_len, _ = torch.sort(tri_len, dim=-1, descending=False)
        return tri_len.squeeze(-2)

    def get_geometric_structure(self, points):
        with torch.no_grad():
            batch_size, num_point, _ = points.shape

            dist_map = torch.sqrt((torch.sum((points.unsqueeze(1) - points.unsqueeze(2)) ** 2, -1)))  # (B, N, N)
            dist_embedding = dist_map

            knn_indices = dist_map.topk(k=self.tri_k + 1, dim=2, largest=False)[1]  # (B, N, k)
            knn_indices = knn_indices[:, :, 1:]
            idx = knn_indices

            knn_indices = knn_indices.unsqueeze(3).expand(batch_size, num_point, self.tri_k, 3)  # (B, N, k, 3)
            expanded_points = points.unsqueeze(1).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
            knn_points = torch.gather(expanded_points, dim=2, index=knn_indices)  # (B, N, k, 3)
            ref_vectors = torch.sum(knn_points, dim=-2) - points  # (B, N, 3)
            anc_vectors = points.unsqueeze(1) - points.unsqueeze(2)  # (B, N, N, 3)
            ref_vectors = ref_vectors.unsqueeze(2).expand(batch_size, num_point, num_point, 3)  # (B, N, N, 3)
            sin_values = torch.linalg.norm(torch.cross(ref_vectors, anc_vectors, dim=-1), dim=-1)  # (B, N, N)
            cos_values = torch.sum(ref_vectors * anc_vectors, dim=-1)  # (B, N, N)
            angles = torch.atan2(sin_values, cos_values)  # (B, N, N)
            angle_embedding = angles

            index = torch.arange(num_point).reshape(1, -1, 1)
            index = index.repeat(batch_size, 1, self.tri_k * (self.tri_k - 1) // 2).unsqueeze(-1).cuda()
            pairs_index = list(itertools.combinations(torch.arange(0, self.tri_k), 2))
            idx_pairs = idx[:, :, pairs_index]
            pair = torch.cat((index, idx_pairs), dim=-1)
            tri_length = self.getTriLength(points, pair)  # (B, N, 3)
            tri_len_map = torch.sum(torch.sqrt((tri_length.unsqueeze(1) - tri_length.unsqueeze(2)) ** 2), dim=-1)
            tri_len_embedding = tri_len_map

        geo_embedding = torch.cat([dist_embedding[..., None], angle_embedding[..., None], tri_len_embedding[..., None]], dim=-1)

        return geo_embedding

    def regular_score(self, score):
        score = torch.where(torch.isnan(score), torch.zeros_like(score), score)
        score = torch.where(torch.isinf(score), torch.zeros_like(score), score)
        return score

    def resample(self, mu, logvar, k=1):
        sample_z = []
        for _ in range(k):
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            sample_z.append(eps.mul(std).add_(mu))
        sample_z = torch.cat(sample_z, dim=2)
        return sample_z

    def forward(self, points_src, points_ref, phase):
        # extract feature
        emb_src = self.pointfeaturer(points_src)
        emb_ref = self.pointfeaturer(points_ref)
        
        coarse_src, fine_src = self.pcn(emb_src)
        coarse_ref, fine_ref = self.pcn(emb_ref)

        geo_src = self.get_geometric_structure(points_src)
        geo_ref = self.get_geometric_structure(points_ref)

        emb_src, emb_ref = self.transformer1(emb_src, emb_ref, geo_src, geo_ref)

        s = self.aff1(emb_src, emb_ref)
        s = self.inst1(s[:, None, :, :]).squeeze(dim=1)
        log_s = sinkhorn_rpm(s, n_iters=20, slack=cfg.MODEL.SKADDCR)
        s = torch.exp(log_s)

        # uncertainty
        _, src_num, _ = points_src.shape
        emb_feats = torch.cat((emb_src, emb_ref), dim=1)
        mean = self.proj_mean(emb_feats)
        variance = self.proj_std(emb_feats)

        prob = self.resample(mean, variance, 1)
        prob = torch.clamp(self.sigmoid(prob), min=0, max=1)
        prob = self.regular_score(prob)

        prob_out = self.resample(mean, variance, 50)
        prob_out = self.sigmoid(prob_out)

        uncertainty = prob_out.var(dim=2, keepdim=True).detach()
        uncertainty = (uncertainty - uncertainty.min()) / (uncertainty.max() - uncertainty.min())

        emb1_new = self.cross(torch.cat((emb_src, torch.bmm(s, emb_ref)), dim=-1))
        emb2_new = self.cross(torch.cat((emb_ref, torch.bmm(s.transpose(1, 2), emb_src)), dim=-1))
        emb_src = emb1_new
        emb_ref = emb2_new

        feats = torch.cat((emb_src, emb_ref), dim=1)
        feats = feats * (1 - uncertainty)

        # random mask
        if phase == 'train':
            rand_mask = uncertainty < torch.Tensor(np.random.random(uncertainty.size())).to(uncertainty.device)
            feats = feats * rand_mask.to(torch.float32)

        emb_src = feats[:, :src_num, :]
        emb_ref = feats[:, src_num:, :]

        emb_src, emb_ref = self.transformer2(emb_src, emb_ref, geo_src, geo_ref)
        s = self.aff2(emb_src, emb_ref)
        s = self.inst2(s[:, None, :, :]).squeeze(dim=1)
        log_s = sinkhorn_rpm(s, n_iters=20, slack=cfg.MODEL.SKADDCR)
        s = torch.exp(log_s)

        src_overlap = self.proj(emb_src)
        ref_overlap = self.proj(emb_ref)
        src_overlap = torch.clamp(self.sigmoid(src_overlap), min=0, max=1)
        ref_overlap = torch.clamp(self.sigmoid(ref_overlap), min=0, max=1)
        src_overlap = self.regular_score(src_overlap)
        ref_overlap = self.regular_score(ref_overlap)

        src_row_sum = torch.sum(s, dim=-1, keepdim=True)
        ref_col_sum = torch.sum(s, dim=-2)[:, :, None]

        data_dict = {
            's_pred': s,
            'src_row_sum': src_row_sum,
            'ref_col_sum': ref_col_sum,
            'src_overlap': src_overlap,
            'ref_overlap': ref_overlap,
            'coarse_src': coarse_src,
            'fine_src': fine_src,
            'coarse_ref': coarse_ref,
            'fine_ref': fine_ref,
            'prob': prob,
            'uncertainty': uncertainty
        }

        return data_dict
