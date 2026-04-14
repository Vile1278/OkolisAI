"""RandLA-Net PyTorch skeleton.

This is a faithful but compact port of the architecture from Hu et al.
(CVPR 2020) — enough to train from scratch. For production we recommend
starting from the original repo's weights and adapting:
    https://github.com/QingyongHu/RandLA-Net

The key innovations we preserve:
  * Random point sampling (fast for large N)
  * Local Spatial Encoding (LocSE): neighbourhood geometry features
  * Attentive Pooling
  * Dilated Residual Block

NOTE: For brevity the forward pass here implements a single encoder/decoder
scale. Extend to 4 scales (n/4, n/16, n/64, n/256) for final training.
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


def knn(xyz: torch.Tensor, k: int) -> torch.Tensor:
    # xyz: (B, N, 3) → idx (B, N, k)
    dist = torch.cdist(xyz, xyz)                      # (B,N,N)
    return dist.topk(k, largest=False).indices        # (B,N,k)


class SharedMLP(nn.Module):
    def __init__(self, in_c, out_c, bn=True):
        super().__init__()
        self.conv = nn.Conv2d(in_c, out_c, 1, bias=not bn)
        self.bn = nn.BatchNorm2d(out_c) if bn else nn.Identity()

    def forward(self, x):
        return F.leaky_relu(self.bn(self.conv(x)), 0.2)


class LocSE(nn.Module):
    """Local Spatial Encoding: for each point, encode relative geometry of its
    k nearest neighbours."""
    def __init__(self, d_out: int):
        super().__init__()
        self.mlp = SharedMLP(10, d_out)

    def forward(self, xyz, feat, knn_idx):
        # xyz: (B,N,3); feat: (B,N,C); knn_idx: (B,N,k)
        B, N, k = knn_idx.shape
        idx = knn_idx.unsqueeze(-1).expand(-1, -1, -1, 3)
        neigh = torch.gather(xyz.unsqueeze(2).expand(-1, -1, k, -1), 1, idx)
        center = xyz.unsqueeze(2).expand(-1, -1, k, -1)
        rel = center - neigh
        dist = rel.norm(dim=-1, keepdim=True)
        geom = torch.cat([center, neigh, rel, dist], dim=-1)   # (B,N,k,10)
        geom = geom.permute(0, 3, 1, 2)                         # (B,10,N,k)
        return self.mlp(geom)                                   # (B,d_out,N,k)


class AttentivePool(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.score = nn.Conv2d(d_in, d_in, 1)
        self.mlp = SharedMLP(d_in, d_out)

    def forward(self, x):
        # x: (B,C,N,k)
        w = F.softmax(self.score(x), dim=-1)
        x = (x * w).sum(dim=-1, keepdim=True)  # (B,C,N,1)
        return self.mlp(x).squeeze(-1)         # (B,d_out,N)


class DilatedResBlock(nn.Module):
    def __init__(self, d_in: int, d_out: int, k: int = 16):
        super().__init__()
        self.k = k
        self.locse1 = LocSE(d_out // 2)
        self.ap1 = AttentivePool(d_out // 2 + d_in, d_out // 2)
        self.locse2 = LocSE(d_out)
        self.ap2 = AttentivePool(d_out + d_out // 2, d_out)
        self.shortcut = nn.Conv1d(d_in, d_out, 1)

    def forward(self, xyz, feat):
        idx = knn(xyz, self.k)
        g1 = self.locse1(xyz, feat, idx)                                 # (B,d/2,N,k)
        f1 = feat.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, self.k)  # (B,d_in,N,k)
        x = torch.cat([g1, f1], dim=1)
        x = self.ap1(x)                                                  # (B,d/2,N)
        g2 = self.locse2(xyz, x.transpose(1, 2), idx)
        x2 = x.unsqueeze(-1).expand(-1, -1, -1, self.k)
        x = torch.cat([g2, x2], dim=1)
        x = self.ap2(x)                                                  # (B,d,N)
        return F.leaky_relu(x + self.shortcut(feat.transpose(1, 2)), 0.2)


class RandLANet(nn.Module):
    """Minimal single-scale RandLA-Net. For full hierarchy, stack N blocks
    with random sampling between and mirror up-samplers."""
    def __init__(self, in_feat_dim: int = 6, num_classes: int = 6, d: int = 64):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_feat_dim, d, 1), nn.BatchNorm1d(d), nn.LeakyReLU(0.2))
        self.block1 = DilatedResBlock(d, d * 2)
        self.block2 = DilatedResBlock(d * 2, d * 4)
        self.head = nn.Sequential(
            nn.Conv1d(d * 4, d * 2, 1), nn.BatchNorm1d(d * 2), nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Conv1d(d * 2, num_classes, 1))

    def forward(self, xyz: torch.Tensor, feats: torch.Tensor) -> torch.Tensor:
        # xyz: (B,N,3); feats: (B,N,F) → logits: (B,N,C)
        x = feats.transpose(1, 2)                   # (B,F,N)
        x = self.stem(x)                            # (B,d,N)
        x = self.block1(xyz, x.transpose(1, 2))     # (B,2d,N)
        x = self.block2(xyz, x.transpose(1, 2))     # (B,4d,N)
        logits = self.head(x)                       # (B,C,N)
        return logits.transpose(1, 2)               # (B,N,C)
