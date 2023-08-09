import torch
from torch.utils.data import Dataset
import math
import numpy as np
import csv
import open3d as o3d
import relie


class GraspDataset(Dataset):
    def __init__(self, n_points=256):
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
        data = np.load(self.files[idx][0])
        pcd = o3d.io.read_point_cloud(self.files[idx][1])

        normals = np.asarray(pcd.normals)
        pcd = np.asarray(pcd.points)
        pcd_idx = np.random.randint(8000, size=self.n_points)
        normals = normals[pcd_idx, :]
        pcd = pcd[pcd_idx, :]

        normals = torch.from_numpy(normals).to(torch.float)
        pcd = torch.from_numpy(pcd).to(torch.float)

        grasps = data['grasp_transform'][data['successful'].astype(bool), :, :]
        grasp = torch.from_numpy(grasps[np.random.randint(0, grasps.shape[0]), :, :]).to(torch.double)
        grasp_vee = relie.utils.se3_tools.se3_vee(relie.utils.se3_tools.se3_log(grasp)).to(torch.float)

        return torch.cat([pcd, normals], dim=1), grasp_vee
