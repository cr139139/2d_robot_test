import torch
import so3
from models import GraspSampler2
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps

model = GraspSampler2()
PATH = './weights/epoch31.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
# torch.manual_seed(0)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device("cpu")
model.to(device)


def sample_from_se3_uniform(B):
    from torch.distributions.uniform import Uniform
    _x = Uniform(-5, 5).sample((B, 3))
    theta = Uniform(-5, 5).sample((B, 3))
    _R = so3.exp_map(theta)
    return _x, _R

B = 100
dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)

for i, data in enumerate(training_loader):
    pcd, grasp_T, ee_T = data
    pcd = pcd.to(device)
    grasp_T = grasp_T.to(device)
    ee_T = ee_T.to(device)

    import time

    start = time.time()

    grasp_T_pred = model.forward(pcd, ee_T, device)

    print(time.time() - start)

    # print(grasp_T)
    # print(grasp_T_pred)

    xyz = pcd[0]
    # H = torch.eye(4)[None, ...].repeat(3, 1, 1)
    # H[0] = grasp_T_pred
    # H[1] = ee_T
    # H[2] = grasp_T
    #
    # colors = torch.zeros_like(H[:, :3, -1])
    # colors[0, 0] = 1
    # colors[1, 1] = 1
    # colors[2, 2] = 1
    # scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)
    colors = torch.zeros_like(grasp_T_pred[:, :3, -1])
    colors[:, 0] = 1
    scene = visualize_grasps(Hs=grasp_T_pred.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)
    if i == 0:
        break
