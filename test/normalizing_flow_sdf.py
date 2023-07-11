import torch
from torch import nn
import math
import so3
from vnn import VNTSVD
from normalizing_flow import SE3Flow


class NormalizingFlow(nn.Module):
    def __init__(self):
        super(NormalizingFlow, self).__init__()
        self.transformation = VNTSVD(2, 8)
        self.net = nn.ModuleList(
            [SE3Flow(8 * 3),
             SE3Flow(8 * 3),
             SE3Flow(8 * 3),
             SE3Flow(8 * 3),
             ]
        )

    def se3_log_probability_normal(self, t, R, t_tar, R_tar, std):
        dR = torch.transpose(R_tar, 1, 2) @ R
        dtheta = so3.log_map(dR)
        dx = (t - t_tar)
        dist = torch.cat((dx, dtheta), dim=-1)
        return -.5 * dist.pow(2).sum(-1) / (std.pow(2))

    def forward(self, R, t, C):
        B = R.shape[0]
        Rc, tc, _, _, _, C = self.transformation.forward(C)
        C = C.reshape((B, -1))
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

        return Rs, ts, log_jacobs, log_pz

    def inverse(self, R, t, C):
        B = R.shape[0]
        Rc, tc, _, _, _, C = self.transformation.forward(C)
        # print(Rc, tc)
        C = C.reshape((B, -1))
        Rs = []
        ts = []

        for layer in self.net[::-1]:
            R, t = layer.inverse(R, t, C)
            Rs.append(Rc @ R)
            ts.append((Rc @ t[:, :, None] + tc).reshape([B, -1]))
        return Rs, ts
