import torch
from torch.utils.data import Dataset
import math
import numpy as np
import sdf
from functools import partial


class GraspDataset(Dataset):
    def __init__(self, shape='circle'):
        self.shape = shape
        self.grasp_R, self.grasp_t, self.grasp_success = self.sample_grasps(shape=shape)
        self.n = self.grasp_R.shape[0]

    def sample_grasps(self, shape="circle"):
        grasp_R = torch.tensor([[[1., 0.],
                                 [0., 1.]]])
        grasp_t = torch.tensor([[-0.5, 0.]])

        n = 4

        theta = math.pi * 2 / n
        c = math.cos(theta)
        s = math.sin(theta)
        generate_R = torch.tensor([[c, -s],
                                   [s, c]])

        for i in range(n - 1):
            grasp_R = torch.concatenate([grasp_R, (generate_R @ grasp_R[-1])[None, :, :]])
            grasp_t = torch.concatenate([grasp_t, (generate_R @ grasp_t[-1])[None, :]])

        if shape == 'circle':
            grasp_success = torch.ones(n)
        elif shape == 'box':
            # grasp_success = torch.ones(n)
            grasp_success = torch.tensor([1, 0]).repeat(n // 2)

        return grasp_R, grasp_t, grasp_success

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # theta = np.random.uniform(0, math.pi * 2)
        theta = np.random.uniform(-math.pi, math.pi)
        c = math.cos(theta)
        s = math.sin(theta)
        R = torch.eye(3, dtype=torch.float)
        R[:2, :2] = torch.tensor([[c, -s],
                                  [s, c]])
        t = torch.from_numpy(np.random.uniform(-5, 5, size=3)).to(torch.float)
        t[2] = 0
        t = torch.zeros(3)
        query_rand = np.random.uniform(-1, 1, size=(100, 3))
        query = torch.from_numpy(query_rand).to(torch.float)
        query[:, 2] = 0
        query2 = torch.from_numpy(query_rand + np.random.normal(0, 1, size=(100, 3))).to(torch.float)
        query2[:, 2] = 0

        SDF = partial(sdf.sdf, shape=self.shape)

        query.requires_grad_()
        d = SDF(sdf.transform(query, R.T, -R.T @ t)[:, :2])
        grad = torch.autograd.grad(d, query, torch.ones_like(d))[0]
        surface = query - d * grad
        query.requires_grad_(False)
        surface = surface.detach().requires_grad_(False)

        query2.requires_grad_()
        d2 = SDF(sdf.transform(query2, R.T, -R.T @ t)[:, :2])
        grad2 = torch.autograd.grad(d2, query2, torch.ones_like(d2))[0]
        surface2 = query2 - d2 * grad2
        query2.requires_grad_(False)
        surface2 = surface2.detach().requires_grad_(False)

        # shape = torch.concatenate([query[:, :, None], surface[:, :, None]], dim=-1)
        shape = torch.concatenate([surface[:, :, None], surface2[:, :, None]], dim=-1)

        grasp_R = torch.eye(3, dtype=torch.float)
        grasp_R[:2, :2] = self.grasp_R[idx]
        grasp_t = torch.zeros(3, dtype=torch.float)
        grasp_t[:2] = self.grasp_t[idx]
        grasp_success = self.grasp_success[idx]

        grasp_R = R @ grasp_R
        grasp_t = (R @ grasp_t[:, None])[:, 0] + t
        # R = torch.eye(3, dtype=torch.float)

        return grasp_R, grasp_t, grasp_success, shape, R, t
