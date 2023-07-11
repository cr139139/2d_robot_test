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


class VNTLinear(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNTLinear, self).__init__()
        self.weight = torch.nn.Parameter(torch.rand(in_channels, out_channels))
        self.weight.data = F.softmax(self.weight.data, dim=1)

    def forward(self, x, normalize=True):
        """
        x: point features of shape [B, N_samples, 3, N_feat, ...]
        """
        if normalize:
            weight = F.softmax(self.weight, dim=0)
        x_out = torch.matmul(x, weight)
        return x_out


class VNTLinearLeakyReLU(nn.Module):
    def __init__(self, in_channels, out_channels, negative_slope=0.2):
        super(VNTLinearLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.map_to_feat = VNTLinear(in_channels, out_channels)
        self.map_to_dir = VNTLinear(in_channels, 1)
        self.map_to_src = VNTLinear(in_channels, 1)

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, 3, N_feat, ...]
        '''
        q = self.map_to_feat(x)
        k = self.map_to_dir(x)
        o = self.map_to_src(x)

        ko = k - o
        qo = q - o

        dotprod = (ko * qo).sum(2, keepdims=True)
        d_norm_sq = (ko * ko).sum(2, keepdims=True)
        mask = (dotprod >= 0).float()
        x_out = self.negative_slope * q + (1 - self.negative_slope) * (
                mask * q + (1 - mask) * (q - (dotprod / (d_norm_sq + EPS)) * ko))
        return x_out


class VNTSVD(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VNTSVD, self).__init__()

        self.layer_a = VNTLinearLeakyReLU(in_channels, out_channels)
        self.layer_b = VNTLinearLeakyReLU(out_channels, 1)
        self.layer_c = VNLinearLeakyReLU(out_channels, out_channels)
        self.layer_d = VNLinearLeakyReLU(out_channels, 3)

    def forward(self, x):
        '''
        x: point features of shape [B, N_samples, 3, N_feat, ...]
        '''
        a = self.layer_a(x)
        b = self.layer_b(a)

        ab = a - b

        c = self.layer_c(ab)
        d = self.layer_d(c)

        cd = (c.transpose(2, -1) @ d).transpose(2, -1)

        A = a.mean(dim=1, keepdim=False)
        B = cd.mean(dim=1, keepdim=False)

        A_cent = A.mean(dim=2, keepdim=True)
        B_cent = B.mean(dim=2, keepdim=True)

        H = (A - A_cent) @ (B - B_cent).transpose(1, -1)
        U, S, Vh = torch.linalg.svd(H)
        n_batch = H.shape[0]
        S = torch.eye(3).reshape((1, 3, 3)).repeat((n_batch, 1, 1))
        S[:, 2, 2] = torch.linalg.det(U @ Vh)
        R = U @ S @ Vh
        t = A_cent - R @ B_cent

        return R, t, a, cd, A, B


if __name__ == "__main__":
    import random
    import math

    theta = random.random()
    c = math.cos(theta)
    s = math.sin(theta)

    R = torch.tensor([[c, -s, 0],
                      [s, c, 0],
                      [0, 0, 1.]])
    t = torch.tensor([1., 2., 3.])

    test = 2

    if test == 0:
        xo = torch.rand((2, 2, 3, 2))
        xt = (R[None, None, :, :] @ xo) + t[None, None, :, None]
        layer = VNTLinearLeakyReLU(2, 2)
        yo = layer.forward(xo)
        yt = layer.forward(xt)
        ytt = (R[None, None, :, :] @ yo) + t[None, None, :, None]
        print(yt - ytt)
    elif test == 1:
        xo = torch.rand((1, 2, 3, 2))
        xt = R[None, None, :, :] @ xo
        layer = VNLinearLeakyReLU(2, 2)
        yo = layer.forward(xo)
        yt = layer.forward(xt)
        ytt = R[None, None, :, :] @ yo
        print(yt - ytt)
    elif test == 2:
        xo = torch.rand((1, 8, 3, 2))
        xt = (R[None, None, :, :] @ xo) + t[None, None, :, None]
        layer = VNTSVD(2, 16)
        Ro, to, ao, cdo, Ao, Bo = layer.forward(xo)
        Rt, tt, at, cdt, At, Bt = layer.forward(xt)

        # print((R[None, None, :, :] @ Ao) + t[None, None, :, None])
        # print(At)
        # print((R[None, None, :, :] @ ao) + t[None, None, :, None])
        # print(at)
        # print(cdo)
        # print(cdt)

        # print(Ro)
        # print(to)
        # print(Rt)
        # print(tt)

        print(Ro @ Bo + to)
        print(Ao)
        print(Rt @ Bt + tt)
        print(At)

        # print(Bo)
        # print(Ro.transpose(1, -1) @ (Ao - to))
        # print(Bt)
        # print(Rt.transpose(1, -1) @ (At - tt))

        # print(Ro.shape, xo.shape, to.shape)
        # # print(R)
        # o = Ro.transpose(1, -1)[:, None, :, :] @ (xo - to[:, None, :, :])
        # t = Rt.transpose(1, -1)[:, None, :, :] @ (xt - tt[:, None, :, :])
        # # print(xo - xt)
        # print(torch.max(o - t))