import torch
from torch import nn
import math


class EuclideanFlow(nn.Module):
    def __init__(self, C):
        super(EuclideanFlow, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(C + 9, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3 * 2)
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
            nn.Linear(C + 3, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 3)
        )

    def forward(self, R, t, C):
        input = torch.concatenate([C, t], dim=1)
        w_ = self.net(input)
        c1 = R[:, :, 2]
        c2 = R[:, :, 0]
        w = w_ - c1 * (torch.inner(c1, w_))

        c2_w = c2 - w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l

        c2_new = constant ** 2 * c2_w - w
        c3_new = torch.cross(c1, c2_new)
        R_new = torch.stack([c2_new, c3_new, c1], dim=2)

        B = R.shape[0]
        J = constant * (torch.eye(3).reshape((1, 3, 3)).repeat(B, 1, 1)
                        - 2 * c2_w[:, :, None] @ c2_w[:, None, :]) / c2_w_l ** 2
        dc_dtheta = R[:, :, 1:2]
        log_det = torch.norm(J @ dc_dtheta, dim=1)
        return R_new, t, log_det

    def inverse(self, R, t, C):
        input = torch.concatenate([C, t], dim=1)
        w_ = self.net(input)
        c1 = R[:, :, 2]
        c2 = R[:, :, 0]
        w = w_ - c1 * (torch.inner(c1, w_))

        c2_w = c2 + w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l

        c2_new = constant ** 2 * c2_w + w
        c3_new = torch.cross(c1, c2_new)
        R_new = torch.stack([c2_new, c3_new, c1], dim=2)
        return R_new, t


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
