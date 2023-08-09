import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearModel(nn.Module):
    '''
    A 4 layer NN with ReLU as activation function
    '''

    def __init__(self, Ni, No, Nh=64):
        super(LinearModel, self).__init__()

        layers = []
        self.layer1 = nn.Linear(Ni, Nh)
        self.layer2 = nn.Linear(Nh, Nh)
        self.layer3 = nn.Linear(Nh, Nh)
        self.layer4 = nn.Linear(Nh, No)

    def forward(self, x):
        x = self.layer1(x)
        x = x + F.relu(self.layer2(x))
        x = x + F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        return x


class EuclideanFlow(nn.Module):
    def __init__(self, n_shape, xyz_index):
        super(EuclideanFlow, self).__init__()
        self.xyz_index = xyz_index
        self.inverse_xyz_index = torch.ones(3).bool()
        self.inverse_xyz_index[self.xyz_index] = False
        self.net = LinearModel(n_shape + 9, 3 * 2, 256)

    def forward(self, R, t, C, device=torch.device("cpu")):
        input = torch.concatenate([C, R.reshape((-1, 9)).to(device)], dim=1)
        output = self.net.forward(input)
        sig, mu = output[:, :3], output[:, 3:]
        t_new = t * torch.exp(sig) + mu
        return R, t_new


class MobiusFlow(nn.Module):
    def __init__(self, n_shape, xyz_index):
        super(MobiusFlow, self).__init__()
        self.xyz_index = xyz_index
        self.net = LinearModel(n_shape + 3, 3 + 1, 256)

    def forward(self, R, t, C, device=torch.device("cpu")):
        c1 = R[:, :, self.xyz_index].to(device)
        c2 = R[:, :, (self.xyz_index + 1) % 3].to(device)
        input = torch.concatenate([C, t], dim=1)
        output = self.net.forward(input)
        weight, w_ = nn.functional.sigmoid(output[:, :1]), output[:, 1:]
        w = w_ - c1 * (c1 * w_).sum(1, keepdim=True)
        w = 0.7 / (1e-8 + torch.norm(w, dim=-1, keepdim=True)) * w * weight

        c2_w = c2 - w
        c2_w_l = torch.norm(c2_w, dim=1, keepdim=True)
        constant = (1 - torch.norm(w, dim=1, keepdim=True) ** 2) / c2_w_l ** 2
        c2_new = constant * c2_w - w
        c3_new = torch.cross(c1, c2_new)

        R_new = torch.zeros(R.shape)
        R_new[:, :, self.xyz_index] = c1
        R_new[:, :, (self.xyz_index + 1) % 3] = c2_new
        R_new[:, :, (self.xyz_index + 2) % 3] = c3_new

        return R_new, t


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


class GraspSampler2(nn.Module):
    def __init__(self):
        super(GraspSampler2, self).__init__()
        self.point_encoder = PointNetEncoder()
        self.net = nn.ModuleList(
            [
                MobiusFlow(1024, 0),
                MobiusFlow(1024, 1),
                MobiusFlow(1024, 2),
                EuclideanFlow(1024, 0),
            ] * 4
        )

    def forward(self, P, T, device):
        P = self.point_encoder.forward(P)
        R = T[:, :3, :3]
        t = T[:, :3, 3]

        for layer in self.net:
            R, t = layer.forward(R, t, P, device)

        B = T.size(0)
        T_new = torch.eye(4).reshape((1, 4, 4)).repeat(B, 1, 1).to(device)
        T_new[:, :3, :3] = R
        T_new[:, :3, 3] = t

        return T_new
