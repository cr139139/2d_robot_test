import torch
from torch import nn
import torch.nn.functional as F


class GraspSampler(nn.Module):
    def __init__(self, channel=6):
        super(GraspSampler, self).__init__()
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

        self.third_conv = nn.Sequential(
            nn.Conv1d(2048 + 512, 512, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, 128, 1),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 9, 1),
        )

    def forward(self, x):
        B, N, _ = x.shape
        feature1 = self.first_conv(x.transpose(2, 1))  # (B, 256, N)
        feature_global = torch.max(feature1, dim=2, keepdim=True)[0]  # (B, 256, 1)

        feature2 = torch.cat([feature_global.expand(-1, -1, N), feature1], dim=1)  # (B, 512, N)
        feature2 = self.second_conv(feature2)  # (B, 1024, N)
        feature_global = torch.max(feature2, dim=2, keepdim=True)[0]

        feature3 = torch.cat([feature_global.expand(-1, -1, N), feature2, feature1], dim=1)
        feature3 = self.third_conv(feature3)

        c1 = feature3[:, :3]
        c2 = feature3[:, 3:6]
        a = feature3[:, 6:]
        b = c2 - c1
        b = b / torch.linalg.norm(b, dim=1)
        a = (a - torch.inner(b, a) * b) / torch.linalg.norm(a, dim=1)
        R = torch.hstack([b, torch.cross(a, b), a])
        t = (c1 + c2) / 2 - 1.12169998e-01 * a

        return c1, c2, R, t
