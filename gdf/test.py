import torch
import so3
from models import GraspDistField, GraspSampler
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps

# model = GraspDistField()
model = GraspSampler()
PATH = './weights/epoch991.pth'
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


B = 20
ee_R = so3.rand()
ee_w = so3.log_map(ee_R)


dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=B, shuffle=True)

for i, data in enumerate(training_loader):
    pcd, grasp_wv, dist, ee_wv = data
    pcd = pcd.to(device)
    grasp_wv = grasp_wv.to(device)
    dist = dist.to(device)
    ee_wv = ee_wv.to(device).requires_grad_()
    import time

    start = time.time()

    # dist_pred = model.forward(pcd, ee_wv)
    # grad_loss = \
    # torch.autograd.grad(dist_pred, ee_wv, grad_outputs=torch.ones_like(dist_pred), create_graph=True)[0]
    # grad_loss_norm = torch.linalg.norm(grad_loss, dim=1, keepdim=True)
    # ee_wv_update = ee_wv - dist_pred * grad_loss / grad_loss_norm
    ee_wv_update = model.forward(pcd, ee_wv)

    grasp_pred_R = so3.exp_map(ee_wv_update[:, :3], device=device)
    grasp_pred_t = ee_wv_update[:, 3:]
    print(time.time() - start)

    # print(ee_wv)
    # print(ee_wv_update)
    # print(grasp_wv)
    # print(grad_loss_norm)

    xyz = pcd[0]
    # H = torch.eye(4)[None, ...].repeat(3, 1, 1)
    # H[0, :3, :3] = grasp_pred_R
    # H[0, :3, 3] = grasp_pred_t
    # H[1, :3, :3] = so3.exp_map(ee_wv[:, :3])
    # H[1, :3, 3] = ee_wv[:, 3:]
    # H[2, :3, :3] = so3.exp_map(grasp_wv[:, :3])
    # H[2, :3, 3] = grasp_wv[:, 3:]
    # # H[3] = grasp
    #
    # colors = torch.zeros_like(H[:, :3, -1])
    # colors[0, 0] = 1
    # colors[1, 1] = 1
    # colors[2, 2] = 1
    # # colors[3, 1:3] = 1

    H = torch.eye(4)[None, ...].repeat(B, 1, 1)
    H[:, :3, :3] = grasp_pred_R
    H[:, :3, 3] = grasp_pred_t
    colors = torch.zeros_like(H[:, :3, -1])
    colors[:, 0] = 1

    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)
    if i == 0:
        break
