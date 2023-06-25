import torch
from torch import nn
import math
import so3


class EuclideanFlow(nn.Module):
    def __init__(self, C):
        super(EuclideanFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(C + 9, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3 * 2)
        )

    def forward(self, R, t, C):
        input = torch.concatenate([C, R.reshape((-1, 9))], dim=1)
        output = self.net(input)
        sig, mu = output[:, :3], output[:, 3:]
        t_new = t * torch.exp(sig) + mu
        log_det = sig.sum(-1)
        return R, t_new, log_det

    def inverse(self, R, t, C):
        input = torch.concatenate([C, R.reshape((-1, 9))], dim=1)
        output = self.net(input)
        sig, mu = output[:, :3], output[:, 3:]
        t_new = (t - mu) * torch.exp(-sig)
        return R, t_new


class MobiusFlow(nn.Module):
    def __init__(self, C):
        super(MobiusFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(C + 3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 3),
            nn.Tanh()
        )

    def forward(self, R, t, C):
        input = torch.concatenate([C, t], dim=1)
        w_ = self.net(input) / math.sqrt(2)
        c1 = R[:, :, 2]
        c2 = R[:, :, 0]
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)

        c2_w = c2 - w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l

        c2_new = constant ** 2 * c2_w - w
        c3_new = torch.cross(c1, c2_new)
        R_new = torch.stack([c2_new, c3_new, c1], dim=2)

        B = R.shape[0]
        J = constant[:, :, None] * (torch.eye(3).reshape((1, 3, 3)).repeat(B, 1, 1) - 2 * c2_w[:, :, None] @ c2_w[:, None, :]) / c2_w_l[:, :, None] ** 2
        dc_dtheta = R[:, :, 1:2]
        log_det = torch.norm(J @ dc_dtheta, dim=1)[:, 0]
        return R_new, t, log_det

    def inverse(self, R, t, C):
        input = torch.concatenate([C, t], dim=1)
        w_ = self.net(input) / math.sqrt(2)
        w_ = w_ / torch.norm(w_, dim=1, keepdim=True)
        c1 = R[:, :, 2]
        c2 = R[:, :, 0]
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)

        c2_w = c2 + w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l

        c2_new = constant ** 2 * c2_w + w
        c3_new = torch.cross(c1, c2_new)
        R_new = torch.stack([c2_new, c3_new, c1], dim=2)
        return R_new, t


class NormalizingFlow(nn.Module):
    def __init__(self, C):
        super(NormalizingFlow, self).__init__()
        self.R1 = MobiusFlow(C)
        self.t1 = EuclideanFlow(C)
        self.R2 = MobiusFlow(C)
        self.t2 = EuclideanFlow(C)
        self.R3 = MobiusFlow(C)
        self.t3 = EuclideanFlow(C)

    def se3_log_probability_normal(self, t, R, t_tar, R_tar, std):
        dR = torch.transpose(R_tar, 1, 2) @ R
        dtheta = so3.log_map(dR)
        dx = (t - t_tar)
        dist = torch.cat((dx, dtheta), dim=-1)
        return -.5 * dist.pow(2).sum(-1) / (std.pow(2))

    def forward(self, R, t, C):
        B = R.shape[0]
        log_jacobs = 0

        R_mu = torch.eye(3).repeat(B, 1, 1)
        t_mu = torch.zeros(3).repeat(B, 1)
        std = 0.3 * torch.ones(B)

        R, t, log_j = self.R1.forward(R, t, C)
        log_jacobs += log_j
        R, t, log_j = self.t1.forward(R, t, C)
        log_jacobs += log_j
        R, t, log_j = self.R2.forward(R, t, C)
        log_jacobs += log_j
        R, t, log_j = self.t2.forward(R, t, C)
        log_jacobs += log_j
        R, t, log_j = self.R3.forward(R, t, C)
        log_jacobs += log_j
        R, t, log_j = self.t3.forward(R, t, C)
        log_jacobs += log_j

        log_pz = self.se3_log_probability_normal(t, R, t_mu, R_mu, std)
        log_jacobs += log_pz

        return R, t, log_jacobs

    def inverse(self, R, t, C):
        R, t = self.R1.inverse(R, t, C)
        R, t = self.t1.inverse(R, t, C)
        R, t = self.R2.inverse(R, t, C)
        R, t = self.t2.inverse(R, t, C)
        R, t = self.R3.inverse(R, t, C)
        R, t = self.t3.inverse(R, t, C)
        return R, t


if __name__ == "__main__":
    theta = math.pi
    c = math.cos(theta)
    s = math.sin(theta)
    R = torch.tensor([[[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]]])
    t = torch.tensor([[0.5, 0.5, 0]])
    C = torch.zeros(1, 5)

    enf = EuclideanFlow(5)
    _, t_new, log_det = enf.forward(R, t, C)
    _, t_orig = enf.inverse(R, t_new, C)
    print(t)
    print(t_new, log_det)
    print(t_orig)

    mnf = MobiusFlow(5)
    R_new, _, log_det = mnf.forward(R, t, C)
    R_orig, _ = mnf.inverse(R_new, t, C)
    print(R)
    print(R_new, log_det)
    print(R_orig)
