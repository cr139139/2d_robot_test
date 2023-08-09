import torch
from torch.utils.data import Dataset

from sklearn.neighbors import KDTree
import open3d as o3d
import numpy as np
import csv


class GraspDataset(Dataset):
    def __init__(self, n_points=256):
        self.files = self.openfile()
        self.n = len(self.files)
        self.n_points = n_points
        self.grasp_points = np.array([[4.10000000e-02, 0, 6.59999996e-02],
                                      [4.10000000e-02, 0, 1.12169998e-01],
                                      [-4.100000e-02, 0, 6.59999996e-02],
                                      [-4.100000e-02, 0, 1.12169998e-01],
                                      [0, 0, 0],
                                      [0, 0, 6.59999996e-02]])

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

        grasp_success = data['successful'].astype(bool)
        contact_points = data['contact_points'][grasp_success]
        grasps = data['grasp_transform'][grasp_success]

        n_grasps = grasps.shape[0]
        tree = KDTree(contact_points.reshape(n_grasps * 2, 3))
        dd, ii = tree.query(pcd, k=1)

        grasp_index = ii // 2
        point_index = ii % 2

        closest_contact_point = contact_points[grasp_index, point_index]
        corresponding_point = contact_points[grasp_index, (point_index + 1) % 2]
        a_gt = grasps[grasp_index, :3, 2]
        grasp_points1 = grasps[grasp_index, :3, :3] @ self.grasp_points.T + grasps[grasp_index, :3, 3][:, :, np.newaxis]
        grasps_temp = grasps[grasp_index, :3, :3]
        grasps_temp[:, :3, :2] = -grasps_temp[:, :3, :2]
        grasp_points2 = grasps_temp @ self.grasp_points.T + grasps[grasp_index, :3, 3][:, :, np.newaxis]

        closest_contact_point = torch.from_numpy(closest_contact_point).to(torch.float)
        corresponding_point = torch.from_numpy(corresponding_point).to(torch.float)
        a_gt = torch.from_numpy(a_gt).to(torch.float)
        grasp_points1 = torch.from_numpy(grasp_points1).to(torch.float)
        grasp_points2 = torch.from_numpy(grasp_points2).to(torch.float)
        normals = torch.from_numpy(normals).to(torch.float)
        pcd = torch.from_numpy(pcd).to(torch.float)

        return torch.cat([pcd, normals],
                         dim=1), closest_contact_point, corresponding_point, a_gt, grasp_points1, grasp_points2
