import torch
from torch.utils.data import Dataset
import numpy as np
import csv
import open3d as o3d
import so3


class GraspDataset(Dataset):
    def __init__(self, n_points=1024):
        self.files = self.openfile()
        self.n = len(self.files)
        self.n_points = n_points

    def openfile(self):
        filename = "datasets.csv"
        with open(filename, 'r') as csvfile:
            csvreader = csv.reader(csvfile)
            files = []
            for lines in csvreader:
                files.append(lines)
        return files

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        idx = 0
        pcd = o3d.io.read_point_cloud(self.files[idx][1])
        pcd = np.asarray(pcd.points)
        pcd_idx = np.random.randint(8000, size=self.n_points)
        pcd = torch.from_numpy(pcd[pcd_idx, :]).to(torch.float)

        data = np.load(self.files[idx][0])
        grasps = data['grasp_transform'][data['successful'].astype(bool), :, :]

        if torch.randint(0, 2, (1,)) == 0:
            limits = [-2, 2]
        else:
            limits = [-0.3, 0.3]
        ee_R = so3.rand()
        ee_t = torch.rand(3) * (limits[1] - limits[0]) + limits[0]

        ee_T = torch.eye(4)
        ee_T[:3, :3] = ee_R
        ee_T[:3, 3] = ee_t
        ee_T_inv = torch.linalg.inv(ee_T)
        grasp = torch.from_numpy(grasps).to(torch.float)

        grasp_dist = ee_T_inv @ grasp
        w = so3.log_map(grasp_dist[:, :3, :3])
        v = grasp_dist[:, :3, 3]
        dist = torch.linalg.norm(v, dim=1) + torch.linalg.norm(w, dim=1)

        min_arg = torch.argmin(dist)
        grasp_T = grasp[min_arg, :, :]

        return pcd, grasp_T, ee_T
