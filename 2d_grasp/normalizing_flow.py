import torch
from torch import nn
import math
import so3
from vnn import VNTSVD, VNTORIG


class EuclideanFlow(nn.Module):
    def __init__(self, C):
        super(EuclideanFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(C + 9 * 1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2 * 2),
            nn.Sigmoid()
        )

    def forward(self, R, t, C):
        input = torch.concatenate([C, R.reshape((-1, 9))], dim=1)
        output = (self.net(input) - 0.5) * 2
        sig, mu = output[:, :2], output[:, 2:]
        t_new = torch.zeros(t.shape)
        t_new[:, :2] = t[:, :2] * torch.exp(sig) + mu
        log_det = sig.sum(-1)
        return R, t_new, log_det

    def inverse(self, R, t, C):
        input = torch.concatenate([C, R.reshape((-1, 9))], dim=1)
        output = (self.net(input) - 0.5) * 2
        sig, mu = output[:, :2], output[:, 2:]
        t_new = torch.zeros(t.shape)
        t_new[:, :2] = (t[:, :2] - mu) * torch.exp(-sig)
        return R, t_new


class MobiusFlow(nn.Module):
    def __init__(self, C):
        super(MobiusFlow, self).__init__()

        self.net = nn.Sequential(
            nn.Linear(C + 3 * 1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, R, t, C):
        input = torch.concatenate([C, t], dim=1)
        w_ = self.net(input) - 0.5
        c1 = R[:, :, 2]
        c2 = R[:, :, 0]
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)

        c2_w = c2 - w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l ** 2

        c2_new = constant * c2_w - w
        c3_new = torch.cross(c1, c2_new)
        R_new = torch.stack([c2_new, c3_new, c1], dim=2)

        B = R.shape[0]
        J = constant[:, :, None] * (
                torch.eye(3).reshape((1, 3, 3)).repeat(B, 1, 1) - 2 * c2_w[:, :, None] @ c2_w[:, None, :] / c2_w_l[
                                                                                                            :, :,
                                                                                                            None] ** 2)
        dc_dtheta = R[:, :, 1:2]
        log_det = torch.norm(J @ dc_dtheta, dim=1)[:, 0]

        return R_new, t, log_det

    def inverse(self, R, t, C):
        input = torch.concatenate([C, t], dim=1)
        w_ = self.net(input) - 0.5
        c1 = R[:, :, 2]
        c2 = R[:, :, 0]
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)

        c2_w = c2 + w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l ** 2

        c2_new = constant * c2_w + w
        c3_new = torch.cross(c1, c2_new)
        R_new = torch.stack([c2_new, c3_new, c1], dim=2)
        return R_new, t


class SE3Flow(nn.Module):
    def __init__(self, C):
        super(SE3Flow, self).__init__()
        self.net = nn.ModuleList(
            [MobiusFlow(C),
             EuclideanFlow(C)
             ]
        )

    def forward(self, R, t, C):
        log_jacobs = 0
        for layer in self.net:
            R, t, log_j = layer.forward(R, t, C)
            log_jacobs += log_j

        return R, t, log_jacobs

    def inverse(self, R, t, C):
        for layer in self.net[::-1]:
            R, t = layer.inverse(R, t, C)
        return R, t


class NormalizingFlow(nn.Module):
    def __init__(self, C):
        super(NormalizingFlow, self).__init__()
        self.transformation = VNTORIG(2, C)
        self.net = nn.ModuleList(
            [SE3Flow(C * 3),
             SE3Flow(C * 3),
             SE3Flow(C * 3),
             SE3Flow(C * 3),
             SE3Flow(C * 3),
             SE3Flow(C * 3),
             SE3Flow(C * 3),
             SE3Flow(C * 3),
             ]
        )
        self.C = C

    def se3_log_probability_normal(self, t, R, t_tar, R_tar, std):
        dR = torch.transpose(R_tar, 1, 2) @ R
        dtheta = so3.log_map(dR)
        dx = (t - t_tar)
        dist = torch.cat((dx, dtheta), dim=-1)
        return -.5 * dist.pow(2).sum(-1) / (std.pow(2))

    def forward(self, R, t, C):
        B = R.shape[0]
        Rc = torch.eye(3).repeat(B, 1, 1)
        tc = torch.zeros((3, 1)).repeat(B, 1, 1)
        Cc = torch.zeros((3, self.C)).repeat(B, 1, 1)

        Rc_temp, tc_temp, _, _, _, C_temp = self.transformation.forward(C[:, :, :2, :])
        Rc[:, :2, :2] = Rc_temp
        tc[:, :2] = tc_temp
        Cc[:, :2, :] = C_temp

        C = Cc.reshape((B, -1))
        R = Rc.transpose(1, -1) @ R
        t = Rc.transpose(1, -1) @ (t[:, :, None] - tc)
        t = t.reshape([B, -1])

        log_jacobs = 0

        R_mu = torch.eye(3).repeat(B, 1, 1)
        t_mu = torch.zeros(3).repeat(B, 1)
        std = 1 * torch.ones(B)

        Rs = []
        ts = []

        for layer in self.net:
            R, t, log_j = layer.forward(R, t, C)
            Rs.append(R)
            ts.append(t)
            log_jacobs += log_j

        log_pz = self.se3_log_probability_normal(t, R, t_mu, R_mu, std)

        return Rs, ts, log_jacobs, log_pz, Rc, tc

    def inverse(self, R, t, C):
        B = R.shape[0]
        Rc = torch.eye(3).repeat(B, 1, 1)
        tc = torch.zeros((3, 1)).repeat(B, 1, 1)
        Cc = torch.zeros((3, self.C)).repeat(B, 1, 1)

        Rc_temp, tc_temp, _, _, _, C_temp = self.transformation.forward(C[:, :, :2, :])
        Rc[:, :2, :2] = Rc_temp
        tc[:, :2] = tc_temp
        Cc[:, :2, :] = C_temp
        print(Rc[0], tc[0])
        C = Cc.reshape((B, -1))

        Rs = []
        ts = []

        for layer in self.net[::-1]:
            R, t = layer.inverse(R, t, C)
            Rs.append(Rc @ R)
            ts.append((Rc @ t[:, :, None] + tc).reshape([B, -1]))
        return Rs, ts
