import torch
from torch import nn
import math
import so3
from vnn import VNNSVD, VNNPointNet


class EuclideanFlow(nn.Module):
    def __init__(self, n_shape, xyz_index):
        super(EuclideanFlow, self).__init__()
        self.xyz_index = xyz_index
        self.inverse_xyz_index = torch.ones(3).bool()
        self.inverse_xyz_index[self.xyz_index] = False

        self.net = nn.Sequential(
            nn.Linear(n_shape + 9 * 1 + 1, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 2 * 2),
            nn.Sigmoid()
        )

    def forward(self, R, t, C):
        t_fixed = t[:, self.xyz_index][:, None]
        input = torch.concatenate([C, R.reshape((-1, 9)).to(torch.device("cuda:0")), t_fixed], dim=1)
        output = (self.net(input) - 0.5) * 2
        sig, mu = output[:, :2], output[:, 2:]
        t_new = torch.zeros(t.shape).to(torch.device("cuda:0"))
        t_new[:, self.xyz_index] = t[:, self.xyz_index]
        t_new[:, self.inverse_xyz_index] = t[:, self.inverse_xyz_index] * torch.exp(sig) + mu
        log_det = sig.sum(-1)
        return R, t_new, log_det

    def inverse(self, R, t, C):
        t_fixed = t[:, self.xyz_index][:, None]
        input = torch.concatenate([C, R.reshape((-1, 9)), t_fixed], dim=1)
        output = (self.net(input) - 0.5) * 2
        sig, mu = output[:, :2], output[:, 2:]
        t_new = torch.zeros(t.shape)
        t_new[:, self.xyz_index] = t[:, self.xyz_index]
        t_new[:, self.inverse_xyz_index] = (t[:, self.inverse_xyz_index] - mu) * torch.exp(-sig)
        return R, t_new


class MobiusFlow(nn.Module):
    def __init__(self, n_shape, xyz_index):
        super(MobiusFlow, self).__init__()
        self.xyz_index = xyz_index

        self.net = nn.Sequential(
            nn.Linear(n_shape + 3 * 1 + 3, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 16),
            nn.LeakyReLU(),
            nn.Linear(16, 3),
            nn.Sigmoid()
        )

    def forward(self, R, t, C):
        c1 = R[:, :, self.xyz_index].to(torch.device("cuda:0"))
        # print(C.shape, t.shape, c1.shape)
        input = torch.concatenate([C, t, c1], dim=1)
        # print(input.shape)
        w_ = self.net(input) - 0.5
        c2 = R[:, :, (self.xyz_index + 1) % 3].to(torch.device("cuda:0"))
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)

        c2_w = c2 - w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l ** 2

        c2_new = constant * c2_w - w
        c3_new = torch.cross(c1, c2_new)

        B = R.shape[0]
        R_new = torch.zeros(R.shape)
        R_new[:, :, self.xyz_index] = c1
        R_new[:, :, (self.xyz_index + 1) % 3] = c2_new
        R_new[:, :, (self.xyz_index + 2) % 3] = c3_new

        J = constant[:, :, None] * \
            (torch.eye(3).reshape((1, 3, 3)).repeat(B, 1, 1).to(torch.device("cuda:0"))
             - 2 * c2_w[:, :, None] @ c2_w[:, None, :] / c2_w_l[:, :, None] ** 2)
        dc_dtheta = R[:, :, self.xyz_index - 1][:, :, None].to(torch.device("cuda:0"))
        log_det = torch.norm(J @ dc_dtheta, dim=1)[:, 0]
        return R_new, t, log_det

    def inverse(self, R, t, C):
        c1 = R[:, :, self.xyz_index]
        input = torch.concatenate([C, t, c1], dim=1)
        w_ = self.net(input) - 0.5
        c2 = R[:, :, (self.xyz_index + 1) % 3]
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)

        c2_w = c2 + w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l ** 2

        c2_new = constant * c2_w + w
        c3_new = torch.cross(c1, c2_new)

        R_new = torch.zeros(R.shape)
        R_new[:, :, self.xyz_index] = c1
        R_new[:, :, (self.xyz_index + 1) % 3] = c2_new
        R_new[:, :, (self.xyz_index + 2) % 3] = c3_new
        return R_new, t


class SE3Flow(nn.Module):
    def __init__(self, n_shape):
        super(SE3Flow, self).__init__()
        self.net = nn.ModuleList(
            [MobiusFlow(n_shape, 0),
             EuclideanFlow(n_shape, 0),
             MobiusFlow(n_shape, 1),
             EuclideanFlow(n_shape, 1),
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
    def __init__(self, n_shape):
        super(NormalizingFlow, self).__init__()
        # self.transformation = VNNSVD(2, n_shape)
        self.transformation = VNNPointNet(2, n_shape)
        self.net = nn.ModuleList(
            [SE3Flow(n_shape * 3),
             SE3Flow(n_shape * 3),
             SE3Flow(n_shape * 3),
             SE3Flow(n_shape * 3),
             SE3Flow(n_shape * 3),
             SE3Flow(n_shape * 3),
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
        # Rc, _, _, C = self.transformation.forward(C)
        C = self.transformation.forward(C)

        C = C.reshape((B, -1))
        # R = Rc.transpose(1, -1) @ R
        # t = (Rc.transpose(1, -1) @ t[:, :, None]).reshape([B, -1])
        log_jacobs = 0

        R_mu = torch.eye(3).repeat(B, 1, 1).to(torch.device("cuda:0"))
        t_mu = torch.zeros(3).repeat(B, 1).to(torch.device("cuda:0"))
        std = 1 * torch.ones(B).to(torch.device("cuda:0"))

        Rs = [R]
        ts = [t]

        for layer in self.net:
            R, t, log_j = layer.forward(R, t, C)
            Rs.append(R)
            ts.append(t)
            log_jacobs += log_j

        log_pz = self.se3_log_probability_normal(t.to(torch.device("cuda:0")), R.to(torch.device("cuda:0")), t_mu, R_mu, std)

        return Rs, ts, log_jacobs, log_pz

    def inverse(self, R, t, C):
        B = R.shape[0]
        # Rc, _, _, C = self.transformation.forward(C)
        C = self.transformation.forward(C)

        C = C.reshape((B, -1))

        Rs = [R]
        ts = [t]

        for layer in self.net[::-1]:
            R, t = layer.inverse(R, t, C)
            # Rs.append(Rc @ R)
            # ts.append((Rc @ t[:, :, None]).reshape([B, -1]))
            Rs.append(R)
            ts.append(t)
        return Rs, ts


if __name__ == "__main__":
    theta = math.pi
    c = math.cos(theta)
    s = math.sin(theta)
    R = torch.tensor([[[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]]])
    t = torch.tensor([[0.5, 0.5, 0]])
    C = torch.zeros(1, 5)

    enf = EuclideanFlow(5, 2)
    _, t_new, log_det = enf.forward(R, t, C)
    _, t_orig = enf.inverse(R, t_new, C)
    print(t)
    print(t_new, log_det)
    print(t_orig)

    mnf = MobiusFlow(5, 2)
    R_new, _, log_det = mnf.forward(R, t, C)
    R_orig, _ = mnf.inverse(R_new, t, C)
    print(R)
    print(R_new, log_det)
    print(R_orig)
