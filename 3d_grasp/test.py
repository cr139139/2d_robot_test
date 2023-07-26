import torch
import so3
from normalizing_flow import NormalizingFlow
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps

model = NormalizingFlow(1024)
PATH = './weights/epoch1.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
torch.manual_seed(0)
print(torch.cuda.is_available())


def sample_from_se3_gaussian(x_tar, R_tar, std):
    x_eps = std[:, None] * torch.randn_like(x_tar)
    theta_eps = std[:, None] * torch.randn_like(x_tar)
    rot_eps = so3.exp_map(theta_eps)
    _x = x_tar + x_eps
    _R = torch.einsum('bmn,bnk->bmk', R_tar, rot_eps)
    return _x, _R


def sample_from_se3_uniform(B):
    from torch.distributions.uniform import Uniform
    _x = Uniform(-5, 5).sample((B, 3))
    _x[:, 2] = 0
    theta = Uniform(-5, 5).sample((B, 3))
    theta[:, 1] = 0
    theta[:, 0] = 0
    _R = so3.exp_map(theta)
    return _x, _R


B = 20
R_mu = torch.eye(3).repeat(B, 1, 1)
x_mu = torch.zeros(3).repeat(B, 1)
std = 1 * torch.ones(B)
x_samples, R_samples = sample_from_se3_gaussian(x_mu, R_mu, std)
# x_samples, R_samples = sample_from_se3_uniform(B)

dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(training_loader):
    object_info, H_real = data
    import time

    start = time.time()
    # R, t, log_jacobs, log_pz = model.forward(H_real[:, :3, :3], H_real[:, :3, 3], object_info.repeat(1, 1, 1))
    grasp_R, grasp_t = model.forward(R_samples, x_samples, object_info.repeat(B, 1, 1), inverse=True)
    print(time.time() - start)
    xyz = object_info[0]
    # xyz = torch.concatenate([object_info[0, :, :, 0], object_info[0, :, :, 1]], dim=0)

    H = torch.eye(4)[None, ...].repeat(B, 1, 1)
    H[:, :3, :3] = grasp_R
    H[:, :3, 3] = grasp_t

    # xyz /= 100
    # H[:, :3, 3] /= 100

    colors = torch.zeros_like(H[:, :3, -1])
    colors[:, 0] = 1
    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)
    # scene = visualize_grasps(Hs=H_real.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1.0)
    if i == 4:
        break
