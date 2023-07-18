import torch
from torch.utils.data import Dataset
import math
import numpy as np
import csv
import open3d as o3d
from sklearn.neighbors import KDTree

class GraspDataset(Dataset):
    def __init__(self):
        self.files = self.openfile()
        self.n = len(self.files)

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
        pcd = np.asarray(pcd.points)
        pcd_idx = np.random.randint(8000, size=256)
        tree = KDTree(pcd)
        nearest_dist, nearest_ind = tree.query(pcd[pcd_idx, :], k=2)
        pcd1 = torch.from_numpy(pcd[nearest_ind[:, 0], :]).to(torch.float)[:, :, None]
        pcd2 = torch.from_numpy(pcd[nearest_ind[:, 1], :]).to(torch.float)[:, :, None]
        pcd = torch.concatenate([pcd1, pcd2], dim=-1)

        grasps = data['grasp_transform'][data['successful'].astype(bool), :, :]
        grasp = torch.from_numpy(grasps[np.random.randint(0, grasps.shape[0]), :, :]).to(torch.float)

        return pcd, grasp
