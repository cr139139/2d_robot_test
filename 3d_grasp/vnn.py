import torch
import torch.nn as nn
import torch.nn.functional as F

EPS = 1e-6


class VNLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(VNLinearLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.map_to_feat = nn.Linear(in_channels, out_channels, bias=False)
        self.map_to_dir = nn.Linear(in_channels, 1, bias=False)

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, 3, N_feat, ...]
        '''
        p = self.map_to_feat(x)
        d = self.map_to_dir(x)
        dotprod = (p * d).sum(2, keepdims=True)
        d_norm_sq = (d * d).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        x_out = self.negative_slope * p + (1 - self.negative_slope) * (
                mask * p + (1 - mask) * (p - (dotprod / (d_norm_sq + EPS)) * d))
        return x_out


class VNNSVD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNNSVD, self).__init__()

        self.layer_a = VNLinearLeakyReLU(in_channels, out_channels)
        self.layer_b = VNLinearLeakyReLU(out_channels, 3)
        self.layer_c = nn.Linear(out_channels * 3, out_channels * 3, bias=True)

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, 3, N_feat, ...]
        '''

        a = self.layer_a(x)
        b = self.layer_b(a)

        ab = (a.transpose(2, -1) @ b).transpose(2, -1)
        n_batch, N_samples, _, N_feat = ab.size()

        c = nn.functional.leaky_relu(self.layer_c(ab.reshape(n_batch, N_samples, -1)))
        c = c.reshape(n_batch, N_samples, 3, N_feat)
        c = c.max(dim=1, keepdim=False)[0]

        A = a.mean(dim=1, keepdim=False)
        B = ab.mean(dim=1, keepdim=False)

        H = A @ B.transpose(1, -1)
        U, S, Vh = torch.linalg.svd(H)
        S = torch.eye(3).reshape((1, 3, 3)).repeat((n_batch, 1, 1))
        S[:, 2, 2] = torch.linalg.det(U @ Vh)
        R = U @ S @ Vh
        return R, a, ab, c


class VNNPointNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNNPointNet, self).__init__()

        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
            nn.ReLU(),
            nn.Conv1d(out_channels, out_channels, 1),
        )
        self.out_channels = out_channels

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, 3]
        '''
        B, N_samples, _ = x.size()
        x = x.transpose(1, -1)
        x = self.net(x)
        x = torch.max(x, 2, keepdim=False)[0]
        return x

if __name__ == "__main__":
    import random
    import math

    theta = random.random()
    c = math.cos(theta)
    s = math.sin(theta)

    R = torch.tensor([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1]])

    test = 3

    if test == 1:
        xo = torch.rand((1, 2, 3, 2))
        xt = R[None, None, :, :] @ xo
        layer = VNLinearLeakyReLU(2, 2)
        yo = layer.forward(xo)
        yt = layer.forward(xt)
        ytt = R[None, None, :, :] @ yo
        print(yt)
        print(ytt)
        print(yt - ytt)
    elif test == 2:
        xo = torch.rand((1, 1, 3, 2))
        xt = (R[None, None, :, :] @ xo)
        layer = VNNSVD(2, 4)
        Ro, ao, abo, co = layer.forward(xo)
        Rt, at, abt, ct = layer.forward(xt)

        # print(Ro @ Ro.transpose(1, -1))
        # print(Rt @ Rt.transpose(1, -1))
        # print(Ro.shape, ao.shape, abo.shape, co.shape)

        print(Ro @ abo)
        # print(ao)
        print(Rt @ abt)
        # print(at)

        # print(co)
        # print(ct)
    elif test == 3:
        xo = torch.rand((2, 2, 3))
        xo[0, :, :] = 1
        layer = VNNPointNet(3, 4)
        C = layer.forward(xo)


