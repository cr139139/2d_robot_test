import torch
import torch.nn as nn
import torch.nn.functional as F


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


class GraspDistField(nn.Module):
    def __init__(self):
        super(GraspDistField, self).__init__()
        self.point_encoder = PointNetEncoder()
        self.hidden = 256

        self.lin1 = nn.Linear(2048+6, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.lin3 = nn.Linear(self.hidden, self.hidden)
        self.lin4 = nn.Linear(self.hidden, self.hidden)
        self.lin5 = nn.Linear(self.hidden, self.hidden)
        self.lin6 = nn.Linear(self.hidden, self.hidden)
        self.lin7 = nn.Linear(self.hidden, self.hidden)
        self.lin8 = nn.Linear(self.hidden, 1)

    def forward(self, P, T):
        P = self.point_encoder.forward(P)
        x = torch.cat([P, T], dim=1)

        x = self.lin1(x)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = F.relu(self.lin5(x))
        x = F.relu(self.lin6(x))
        x = F.relu(self.lin7(x))
        x = F.softplus(self.lin8(x))
        return x


class GraspSampler(nn.Module):
    def __init__(self):
        super(GraspSampler, self).__init__()
        self.point_encoder = PointNetEncoder()

        self.hidden = 1024

        self.lin1 = nn.Linear(1024 + 6, self.hidden)
        self.lin2 = nn.Linear(self.hidden, self.hidden)
        self.lin3 = nn.Linear(self.hidden, self.hidden)
        self.lin4 = nn.Linear(self.hidden, self.hidden)
        self.lin5 = nn.Linear(self.hidden, 6)

    def forward(self, P, T):
        P = self.point_encoder.forward(P)
        x = torch.cat([P, T], dim=1)

        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.relu(self.lin4(x))
        x = self.lin5(x)
        return x


class GraspDistFieldDec(nn.Module):
    def __init__(self):
        super(GraspDistFieldDec, self).__init__()
        self.lin1 = nn.Linear(1030, 512)
        self.lin2 = nn.Linear(512, 512)
        self.lin3 = nn.Linear(512, 512)
        self.lin4 = nn.Linear(512, 1)

    def forward(self, P, T):
        x = torch.cat([P, T], dim=1)
        x = self.lin1(x)
        x = F.relu(self.lin2(x))
        x = F.relu(self.lin3(x))
        x = F.softplus(self.lin4(x))
        return x
