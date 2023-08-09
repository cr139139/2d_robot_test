import torch
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps
import numpy as np

# torch.manual_seed(0)

dataset = GraspDataset(256)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(test_loader):
    pcd, closest_contact_point, corresponding_point, a_gt, grasp_points1, grasp_points2 = data

    a = a_gt
    b = grasp_points2 - grasp_points1
    b = b / torch.linalg.norm(b, dim=1)
    a = (a - torch.inner(b, a) * b) / torch.linalg.norm(a, dim=1)
    R = torch.hstack([b, torch.cross(a, b), a])
    t = (grasp_points1 + grasp_points2) / 2 - 1.12169998e-01 * a

    xyz = pcd[0, :, :3]
    batch_size = xyz.size(0)
    H = torch.eye(4)[None, ...].repeat(batch_size, 1, 1)
    H[:, :3, :3] = R
    H[:, :3, 3] = t

    colors = torch.zeros_like(H[:, :3, -1])
    colors[:, 0] = 1
    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)

    if i == 0:
        break
