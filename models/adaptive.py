import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None, dim6=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim6 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 0:3], k=k)
        device = x.device

        idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1) * num_points

        idx = idx + idx_base

        idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2).contiguous()

    return feature, idx


class AdaptiveConv(nn.Module):
    def __init__(self, k, in_channels, feat_channels, nhiddens, out_channels):
        super(AdaptiveConv, self).__init__()
        self.in_channels = in_channels
        self.nhiddens = nhiddens
        self.out_channels = out_channels
        self.feat_channels = feat_channels
        self.k = k

        self.conv0 = nn.Conv2d(feat_channels, nhiddens, kernel_size=1, bias=False)
        self.conv1 = nn.Conv2d(nhiddens, nhiddens * in_channels, kernel_size=1, bias=False)
        self.bn0 = nn.BatchNorm2d(nhiddens)
        self.bn1 = nn.BatchNorm2d(nhiddens)
        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2)
        self.residual_layer = nn.Sequential(nn.Conv2d(feat_channels, out_channels, kernel_size=1, bias=False),
                                            nn.BatchNorm2d(out_channels),
                                            )
        self.linear = nn.Sequential(nn.Conv2d(nhiddens, out_channels, kernel_size=1, bias=False),
                                    nn.BatchNorm2d(out_channels))

    def forward(self, points, feat, idx):
        # points: (bs, in_channels, num_points), feat: (bs, feat_channels/2, num_points)
        batch_size, _, num_points = points.size()

        x, _ = get_graph_feature(points, k=self.k, idx=idx)  # (bs, in_channels, num_points, k)
        y, _ = get_graph_feature(feat, k=self.k, idx=idx)  # (bs, feat_channels, num_points, k)

        kernel = self.conv0(y)  # (bs, nhiddens, num_points, k)
        kernel = self.leaky_relu(self.bn0(kernel))
        kernel = self.conv1(kernel)  # (bs, in*nhiddens, num_points, k)
        kernel = kernel.permute(0, 2, 3, 1).view(batch_size, num_points, self.k, self.nhiddens,
                                                 self.in_channels)  # (bs, num_points, k, nhiddens, in)

        x = x.permute(0, 2, 3, 1).unsqueeze(4)  # (bs, num_points, k, in_channels, 1)
        x = torch.matmul(kernel, x).squeeze(4)  # (bs, num_points, k, nhiddens)
        x = x.permute(0, 3, 1, 2).contiguous()  # (bs, nhiddens, num_points, k)

        # nhiddens -> out_channels
        x = self.leaky_relu(self.bn1(x))
        x = self.linear(x)  # (bs, out_channels, num_points, k)
        # residual: feat_channels -> out_channels
        residual = self.residual_layer(y)
        x += residual
        x = self.leaky_relu(x)

        x = x.max(dim=-1, keepdim=False)[0]  # (bs, out_channels, num_points)

        return x


class AdaptiveFCN(nn.Module):
    def __init__(self, neighboursnum, emb_dims=512):
        super(AdaptiveFCN, self).__init__()

        self.neighboursnum = neighboursnum

        self.adp_conv1 = AdaptiveConv(neighboursnum, 6, 6, 64, 64)
        self.adp_conv2 = AdaptiveConv(neighboursnum, 6, 128, 64, 64)
        self.adp_conv3 = AdaptiveConv(neighboursnum, 6, 128, 64, 128)
        self.adp_conv4 = AdaptiveConv(neighboursnum, 6, 256, 64, 256)

        self.conv = nn.Conv2d(512, emb_dims, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(emb_dims)

    def forward(self, x):
        x = x[:, :, 0:3]
        x = x.permute(0, 2, 1).contiguous()  # (B, 3, N)

        batch_size, num_dims, num_points = x.size()

        points = x
        _, idx = get_graph_feature(points, k=self.neighboursnum)

        x1 = self.adp_conv1(points, x, idx)
        x2 = self.adp_conv2(points, x1, idx)
        x3 = self.adp_conv3(points, x2, idx)
        x4 = self.adp_conv4(points, x3, idx)

        x = torch.cat((x1, x2, x3, x4), dim=1)
        x_node = x

        x_edge = F.relu(self.bn(self.conv(x.unsqueeze(dim=-1)))).view(batch_size, -1, num_points)

        x = torch.cat((x_node, x_edge), dim=1).transpose(1, 2).contiguous()
        return x
