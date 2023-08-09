import torch
from model import GraspSampler
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps
import numpy as np

grasp_model = GraspSampler()

PATH = './weights/epoch991.pth'
checkpoint = torch.load(PATH)
grasp_model.load_state_dict(checkpoint['model_state_dict'])
# torch.manual_seed(0)
print(torch.cuda.is_available())

dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(training_loader):
    pcd, closest_contact_point, corresponding_point, a_gt, grasp_points1, grasp_points2 = data

    import time

    start = time.time()
    c1, c2, R, t = grasp_model.forward(pcd)
    print(time.time() - start)

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
