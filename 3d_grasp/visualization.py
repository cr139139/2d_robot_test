import torch
import numpy as np

import matplotlib.pyplot as plt
from grasp_visualization import visualize_grasps

import io
from PIL import Image
import so3


def sample_from_se3_gaussian(x_tar, R_tar, std):
    x_eps = std[:, None] * torch.randn_like(x_tar)
    theta_eps = std[:, None] * torch.randn_like(x_tar)
    rot_eps = so3.exp_map(theta_eps)

    _x = x_tar + x_eps
    _R = torch.einsum('bmn,bnk->bmk', R_tar, rot_eps)
    return _x, _R


B = 100
R_mu = so3.rand().repeat(B, 1, 1)
x_mu = torch.randn(1, 3).repeat(B, 1)
Hmu = torch.eye(4)[None, ...]
Hmu[:, :3, :3] = R_mu[:1, ...]
Hmu[:, :3, -1] = x_mu[:1, :]
std = 0.3 * torch.ones(B)

x_samples, R_samples = sample_from_se3_gaussian(x_mu, R_mu, std)

H = torch.eye(4)[None, ...].repeat(B, 1, 1)
H[:, :3, :3] = R_samples
H[:, :3, -1] = x_samples

H = torch.cat((H, Hmu), dim=0)
colors = torch.zeros_like(H[:, :3, -1])
colors[-1, 0] = 1

scene = visualize_grasps(Hs=H, colors=colors.numpy(), show=False)
data = scene.save_image(resolution=(1080, 1080))
image = np.array(Image.open(io.BytesIO(data)))
plt.imshow(image)
plt.show()


def se3_log_probability_normal(x, R, x_tar, R_tar, std):
    dR = torch.transpose(R_tar, 1, 2) @ R
    dtheta = so3.log_map(dR)

    dx = (x - x_tar)

    dist = torch.cat((dx, dtheta), dim=-1)
    return -.5 * dist.pow(2).sum(-1) / (std.pow(2))


log_prob = se3_log_probability_normal(x_samples, R_samples, x_mu, R_mu, std)
prob = torch.exp(log_prob)
colors[:-1, 1] = (prob - prob.min()) / (prob.max() + prob.min())

scene = visualize_grasps(Hs=H, colors=colors.numpy(), show=False)
data = scene.save_image(resolution=(1080, 1080))
image = np.array(Image.open(io.BytesIO(data)))
plt.imshow(image)
plt.show()
