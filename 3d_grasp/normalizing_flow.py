import torch
from torch import nn
import math
import so3
import torch.nn.functional as F
from vnn import VNNSVD
from pointnet2_cls_ssg import pointnet2_encoder


# from pointnet_utils import PointNetEncoder


class PointNetEncoder(nn.Module):
    def __init__(self, channel=6):
        super(PointNetEncoder, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv1d(channel, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1)
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 1024, 1)
        )

    def forward(self, x, idx=None):
        B, N, _ = x.shape
        feature = self.first_conv(x.transpose(2, 1))  # (B, 256, N)
        feature_global = torch.max(feature, dim=2, keepdim=True)[0]  # (B, 256, 1)
        feature = torch.cat([feature_global.expand(-1, -1, N), feature], dim=1)  # (B, 512, N)
        feature = self.second_conv(feature)  # (B, 1024, N)
        if idx is None:
            feature_global = torch.max(feature, dim=2, keepdim=False)[0]  # (B, 1024)
        else:
            feature_global = feature[torch.arange(B), :, idx[:, 0]]
        return feature_global


class LinearModel(nn.Module):
    '''
    A 4 layer NN with ReLU as activation function
    '''

    def __init__(self, Ni, No, Nh=256):
        super(LinearModel, self).__init__()

        self.lin1 = nn.Linear(Ni, Nh)
        self.lin2 = nn.Linear(Nh, Nh)
        self.lin3 = nn.Linear(Nh, Nh)
        self.lin4 = nn.Linear(Nh, Nh)
        self.lin5 = nn.Linear(Nh, No)

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = x + F.relu(self.lin2(x))
        x = x + F.relu(self.lin3(x))
        x = x + F.relu(self.lin4(x))
        x = self.lin5(x)
        return x


class EuclideanFlow(nn.Module):
    def __init__(self, n_shape, xyz_index):
        super(EuclideanFlow, self).__init__()
        self.xyz_index = xyz_index
        self.inverse_xyz_index = torch.ones(3).bool()
        self.inverse_xyz_index[self.xyz_index] = False
        self.net = LinearModel(n_shape + 9 + 1, 2 * 2)

    def forward(self, R, t, C, inverse=False, device=torch.device("cpu")):
        t_fixed = t[:, self.xyz_index][:, None]
        input = torch.concatenate([C, R.reshape((-1, 9)).to(device), t_fixed], dim=1)
        output = self.net.forward(input)
        sig, mu = F.tanh(output[:, :2]), output[:, 2:]
        t_new = torch.zeros(t.shape).to(device)
        t_new[:, self.xyz_index] = t[:, self.xyz_index]
        if not inverse:
            t_new[:, self.inverse_xyz_index] = t[:, self.inverse_xyz_index] * torch.exp(sig) + mu
            # t_new = t * torch.exp(sig) + mu
            log_det = sig.sum(-1)
            return R, t_new, log_det
        else:
            t_new[:, self.inverse_xyz_index] = (t[:, self.inverse_xyz_index] - mu) * torch.exp(-sig)
            # t_new = (t - mu) * torch.exp(-sig)
            return R, t_new

    def inverse(self, R, t, C, device=torch.device("cpu")):
        return self.forward(R, t, C, inverse=True, device=device)


class MobiusFlow(nn.Module):
    def __init__(self, n_shape, xyz_index):
        super(MobiusFlow, self).__init__()
        self.xyz_index = xyz_index
        self.net = LinearModel(n_shape + 3 * 1 + 3, 3 + 1)

    def forward(self, R, t, C, inverse=False, device=torch.device("cpu")):
        c1 = R[:, :, self.xyz_index].to(device)
        c2 = R[:, :, (self.xyz_index + 1) % 3].to(device)
        input = torch.concatenate([C, t, c1], dim=1)
        output = self.net.forward(input)
        weight, w_ = nn.functional.sigmoid(output[:, :1]), output[:, 1:]
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)
        w = 0.7 / (1e-8 + torch.norm(w, dim=-1, keepdim=True)) * w * weight

        if not inverse:
            c2_w = c2 - w
            c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
            constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l ** 2
            c2_new = constant * c2_w - w
            c3_new = torch.cross(c1, c2_new)
        else:
            c2_w = c2 + w
            c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
            constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l ** 2
            c2_new = constant * c2_w + w
            c3_new = torch.cross(c1, c2_new)

        B = R.shape[0]
        # c2_new = c2_new / (1e-8 + c2_new.norm(dim=-1, keepdim=True))
        # c3_new = c3_new / (1e-8 + c3_new.norm(dim=-1, keepdim=True))
        R_new = torch.zeros(R.shape).to(device)
        R_new[:, :, self.xyz_index] = c1
        R_new[:, :, (self.xyz_index + 1) % 3] = c2_new
        R_new[:, :, (self.xyz_index + 2) % 3] = c3_new

        if not inverse:
            c2_w = c2_w / c2_w_l
            J = constant[:, :, None] * \
                (torch.eye(3).reshape((1, 3, 3)).repeat(B, 1, 1).to(device)
                 - 2 * c2_w[:, :, None] @ c2_w[:, None, :])
            dc_dtheta = R[:, :, (self.xyz_index + 2) % 3][:, :, None].to(device)
            log_det = torch.norm(J @ dc_dtheta, dim=1)[:, 0]
            return R_new, t, log_det
        else:
            return R_new, t

    def inverse(self, R, t, C, device=torch.device("cpu")):
        return self.forward(R, t, C, inverse=True, device=device)


import mobiusflow


class wrapper(nn.Module):
    def __init__(self, n_shape, xyz_index):
        super(wrapper, self).__init__()
        self.net = mobiusflow.MobiusFlow(3, 1, condition=1, feature_dim=n_shape + 3)
        self.permute = [xyz_index, (xyz_index + 1) % 3, (xyz_index + 2) % 3]

    def forward(self, R, t, C, inverse=False, device=torch.device("cpu")):
        condition = torch.concatenate([C, t], dim=1)
        if not inverse:
            R_new, ldj = self.net.forward(R, permute=self.permute, feature=condition)
            return R_new, t, ldj
        else:
            R_new, _ = self.net.inverse(R, permute=self.permute, feature=condition)
            return R_new, t

    def inverse(self, R, t, C, device=torch.device("cpu")):
        return self.forward(R, t, C, inverse=True, device=device)


# class NormalizingFlow(nn.Module):
#     def __init__(self, n_shape):
#         super(NormalizingFlow, self).__init__()
#         # self.transformation = VNNSVD(2, n_shape)
#         # self.transformation = pointnet2_encoder()
#         self.transformation = PointNetEncoder()
#         self.net = nn.ModuleList(
#             [EuclideanFlow(n_shape, 0),
#              wrapper(n_shape, 0),
#              wrapper(n_shape, 1),
#              ] * 8
#         )
#
#     def se3_log_probability_normal(self, t, R):
#         dtheta = so3.log_map(R)
#         dx = t
#         dist = torch.cat((dx, dtheta), dim=-1)
#         return -.5 * dist.pow(2).sum(-1) / (1 ** 2)
#
#     def forward(self, R, t, C, inverse=False, device=torch.device("cpu")):
#         C = C.transpose(1, -1)
#         C = self.transformation.forward(C)
#
#         if not inverse:
#             log_jacobs = 0
#             for layer in self.net:
#                 R, t, log_j = layer.forward(R, t, C, inverse, device)
#                 log_jacobs += log_j
#             log_pz = self.se3_log_probability_normal(t.to(device), R.to(device))
#             return R, t, log_jacobs, log_pz
#         else:
#             for layer in self.net[::-1]:
#                 R, t = layer.inverse(R, t, C, device)
#             return R, t


class NormalizingFlow2(nn.Module):
    def __init__(self, n_shape):
        super(NormalizingFlow2, self).__init__()
        self.transformation = pointnet2_encoder(normal_channel=True)
        # self.transformation = PointNetEncoder()
        self.net = nn.ModuleList(
            [
                MobiusFlow(n_shape, 0),
                MobiusFlow(n_shape, 1),
                MobiusFlow(n_shape, 2),
                EuclideanFlow(n_shape, 0),
                EuclideanFlow(n_shape, 1),
                EuclideanFlow(n_shape, 2),
                MobiusFlow(n_shape, 0),
                MobiusFlow(n_shape, 1),
                MobiusFlow(n_shape, 2),
            ] * 1
        )

    def forward(self, R, t, C, idx=None, inverse=False, device=torch.device("cpu")):
        C = self.transformation.forward(C)
        if not inverse:
            log_jacobs = 0
            for layer in self.net:
                R, t, log_j = layer.forward(R, t, C, inverse, device)
                log_jacobs += log_j
            return R, t, log_jacobs
        else:
            for layer in self.net[::-1]:
                R, t = layer.inverse(R, t, C, device)
            return R, t


if __name__ == "__main__":
    theta = math.pi
    c = math.cos(theta)
    s = math.sin(theta)
    R = torch.tensor([[[c, -s, 0],
                       [s, c, 0],
                       [0, 0, 1]],
                      [[1, 0, 0],
                       [0, c, -s],
                       [0, s, c]]])
    t = torch.tensor([[0.5, 0.5, 0],
                      [0.5, 0, 0.5]])
    C = torch.zeros(2, 5)

    # enf = EuclideanFlow(5, 2)
    # _, t_new, log_det = enf.forward(R, t, C)
    # _, t_orig = enf.forward(R, t_new, C, inverse=True)
    # print(t)
    # print(t_new, log_det)
    # print(t_orig)

    mnf = MobiusFlow(5, 1)
    R_new, _, log_det = mnf.forward(R, t, C)
    R_orig, _ = mnf.forward(R_new, t, C, inverse=True)
    print(R)
    print(R_new, log_det)
    print(R_orig)
