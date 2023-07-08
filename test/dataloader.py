import torch
from torch.utils.data import Dataset
import math
import numpy as np


class GraspDataset(Dataset):
    def __init__(self, shape='box'):
        self.shape = shape
        self.grasp_R, self.grasp_t, self.grasp_success = self.sample_grasps(shape=shape)
        self.n = self.grasp_R.shape[0]

    def sample_grasps(self, shape="box"):
        grasp_R = torch.tensor([[[1., 0.],
                                 [0., 1.]]])
        grasp_t = torch.tensor([[-0.5, 0.]])

        n = 8

        theta = math.pi * 2 / n
        c = math.cos(theta)
        s = math.sin(theta)
        generate_R = torch.tensor([[c, -s],
                                   [s, c]])

        for i in range(n-1):
            grasp_R = torch.concatenate([grasp_R, (generate_R @ grasp_R[-1])[None, :, :]])
            grasp_t = torch.concatenate([grasp_t, (generate_R @ grasp_t[-1])[None, :]])

        if shape == 'circle':
            grasp_success = torch.ones(n)
        elif shape == 'box':
            grasp_success = torch.tensor([1, 0]).repeat(n//2)

        return grasp_R, grasp_t, grasp_success

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # theta = np.random.uniform(0, math.pi * 2)
        theta = np.random.uniform(-math.pi / 180, math.pi / 180)
        c = math.cos(theta)
        s = math.sin(theta)
        R = torch.eye(3, dtype=torch.float)
        R[:2, :2] = torch.tensor([[c, -s],
                                  [s, c]])
        t = torch.from_numpy(np.random.uniform(-5, 5, size=3)).to(torch.float)
        t[2] = 0
        t = torch.zeros(3)

        grasp_R = torch.eye(3, dtype=torch.float)
        grasp_R[:2, :2] = self.grasp_R[idx]
        grasp_t = torch.zeros(3, dtype=torch.float)
        grasp_t[:2] = self.grasp_t[idx]
        grasp_success = self.grasp_success[idx]

        grasp_R = R @ grasp_R
        grasp_t = (R @ grasp_t[:, None])[:, 0] + t
        R = torch.eye(3, dtype=torch.float)

        return grasp_R, grasp_t, grasp_success, torch.concatenate([R[:2, :2].reshape(-1), t[:2]])
