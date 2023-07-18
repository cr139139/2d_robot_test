import torch
import so3
from normalizing_flow import NormalizingFlow
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps

model = NormalizingFlow(64)
PATH = './weights/epoch41.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
torch.manual_seed(0)


def sample_from_se3_gaussian(x_tar, R_tar, std):
    x_eps = std[:, None] * torch.randn_like(x_tar)
    theta_eps = std[:, None] * torch.randn_like(x_tar)
    rot_eps = so3.exp_map(theta_eps)
    _x = x_tar + x_eps
    _R = torch.einsum('bmn,bnk->bmk', R_tar, rot_eps)
    return _x, _R


B = 20
R_mu = torch.eye(3).repeat(B, 1, 1)
x_mu = torch.zeros(3).repeat(B, 1)
std = 1 * torch.ones(B)
x_samples, R_samples = sample_from_se3_gaussian(x_mu, R_mu, std)

dataset = GraspDataset()
training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(training_loader):
    object_info, _ = data
    grasp_R, grasp_t = model.inverse(R_samples, x_samples, object_info.repeat(B, 1, 1, 1))

    xyz = torch.concatenate([object_info[0, :, :, 0], object_info[0, :, :, 1]], dim=0)

    H = torch.eye(4)[None, ...].repeat(B, 1, 1)
    H[:, :3, :3] = grasp_R[-1]
    H[:, :3, -1] = grasp_t[-1]

    colors = torch.zeros_like(H[:, :3, -1])
    colors[:, 0] = 1

    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1.0)
    break
