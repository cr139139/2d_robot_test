import torch
from torch import nn
import math


class PointNetEncoder(nn.Module):
    def __init__(self, channel=3):
        super(PointNetEncoder, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )

    def forward(self, x):
        B, N, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # (B,  256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B,  256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B,  512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)
        feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)
        return feature_global


class LinearModel(nn.Module):
    def __init__(self, n_in, n_out, n_hidden=64):
        super(LinearModel, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_out)
        )

    def forward(self, x):
        x = self.net(x)
        return x


class AffineCoupling(nn.Module):
    def __init__(self, n_in, n_cond=0):
        super(AffineCoupling, self).__init__()
        self.net = LinearModel(n_in + n_cond, n_in * 2)
        self.scale = nn.Parameter(torch.ones(n_in, 1, 1))

    def forward(self, x, ldj, cond=None, reverse=False):
        x_change, x_id = x.chunk(2, dim=1)

        if cond is not None:
            st = self.net(torch.cat((x_id, cond), dim=1))
        else:
            st = self.net(x_id)
        s, t = st[:, 0::2, ...], st[:, 1::2, ...]
        s = self.scale * torch.tanh(s)

        if reverse:
            x_change = x_change * s.mul(-1).exp() - t
            ldj = ldj - s.flatten(1).sum(-1)
        else:
            x_change = (x_change + t) * s.exp()
            ldj = ldj + s.flatten(1).sum(-1)

        x = torch.cat((x_change, x_id), dim=1)
        return x, ldj


class NormalizingFlow(nn.Module):
    def __init__(self, n_layer, n_dim, n_cond=0):
        super(NormalizingFlow, self).__init__()
        self.n_layer = n_layer
        self.net = nn.ModuleList(
            [AffineCoupling(n_dim // 2, n_cond)] * n_layer
        )

        r = list(range(n_dim))
        permutations_temp = [r[i:] + r[:i] for i in range(n_dim)]
        self.permutations = []
        self.inv_permutations = []

        for i in range(n_layer):
            index = i % n_layer
            permutation = torch.tensor(permutations_temp[index], dtype=torch.long)
            inv_permutation = torch.sort(permutation)[1]
            self.permutations.append(permutation)
            self.inv_permutations.append(inv_permutation)

    def forward(self, x, cond=None, reverse=False):
        ldj = 0
        if not reverse:
            for i in range(self.n_layer):
                x = x[..., self.permutation[i]]
                x, ldj = self.net[i].forward(x, ldj, cond)
            return x, ldj
        else:
            for i in reversed(range(10)):
                x, ldj = self.net[i].forward(x, ldj, cond, reverse)
                x = x[..., self.inv_permutations[i]]
            return x, ldj
