import torch
import so3
from normalizing_flow_sdf import NormalizingFlow
import numpy as np
import math
import matplotlib.pyplot as plt

model = NormalizingFlow()
PATH = './weights/epoch99001.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])

torch.manual_seed(0)


def sample_from_se3_gaussian(x_tar, R_tar, std):
    x_eps = std[:, None] * torch.randn_like(x_tar)
    theta_eps = std[:, None] * torch.randn_like(x_tar)
    x_eps[:, 2] = 0
    theta_eps[:, 1] = 0
    theta_eps[:, 0] = 0
    rot_eps = so3.exp_map(theta_eps)
    _x = x_tar + x_eps
    _R = torch.einsum('bmn,bnk->bmk', R_tar, rot_eps)
    return _x, _R


B = 100
R_mu = torch.eye(3).repeat(B, 1, 1)
x_mu = torch.zeros(3).repeat(B, 1)
std = 1 * torch.ones(B)
x_samples, R_samples = sample_from_se3_gaussian(x_mu, R_mu, std)

theta = np.random.uniform(-math.pi / 180, math.pi / 180)
c = math.cos(theta)
s = math.sin(theta)
R = torch.eye(3, dtype=torch.float)
R[:2, :2] = torch.tensor([[c, -s],
                          [s, c]])
t = torch.from_numpy(np.random.uniform(-5, 5, size=3)).to(torch.float)
t[2] = 0

import sdf
query = torch.from_numpy(np.random.uniform(-5, 5, size=(100, 3))).to(torch.float)
d = sdf.sdf(sdf.transform(query, R.T, -R.T @ t), 'circle')
surface = query - d * sdf.sdf_grad(sdf.transform(query, R.T, -R.T @ t), 'circle')
shape = torch.concatenate([query[:, :, None], surface[:, :, None]], dim=-1)

C = shape[None, :, :, :].repeat(B, 1, 1, 1)
grasp_Rs, grasp_ts = model.inverse(R_samples, x_samples, C)
n_way = -1
grasp_R, grasp_t = grasp_Rs[n_way], grasp_ts[n_way]
grasp_R = grasp_R.detach().numpy()
grasp_t = grasp_t.detach().numpy()
grasp_R = grasp_R[:, :2, :2]
grasp_t = grasp_t[:, :2]

w = 100
r = 5
x = torch.linspace(-r, r, steps=w)
y = torch.linspace(-r, r, steps=w)
x_index, y_index = torch.meshgrid(x, y, indexing="xy")
z_index = torch.zeros((w, w))
xy = torch.concatenate([x_index, y_index, z_index]).reshape((3, -1)).T

from sdf import sdf, transform
query = transform(xy, R.T, -R.T @ t)
query.requires_grad_(True)
d = sdf(query, shape='circle').detach().numpy()

# grad = torch.autograd.grad(d, query, grad_outputs=torch.ones_like(d), create_graph=True)[0]
# query.requires_grad_(False)
# d = d.detach().numpy()
# grad = transform(grad.detach().numpy(), R)

hand_points = torch.tensor([[0, -0.1],
                            [0, 0.1],
                            [0.1, -0.1],
                            [0.1, 0.1]])
hand_edges = [[0, 1], [0, 2], [1, 3]]

kw = dict(extent=(-r, r, -r, r), vmin=-r, vmax=r)
contour = plt.contourf(d.reshape((w, w)), 10, cmap='coolwarm', **kw)
# plt.quiver(xy[:, 0], xy[:, 1], grad[:, 0], grad[:, 1], alpha=0.3)
plt.contour(d.reshape((w, w)), levels=[0.0], colors='black', **kw)

for i in range(grasp_R.shape[0]):
    for j in range(len(hand_edges)):
        new_hand_points = hand_points @ grasp_R[i].T + grasp_t[i]
        plt.plot(new_hand_points[hand_edges[j], 0], new_hand_points[hand_edges[j], 1], 'k')

plt.colorbar(contour)
plt.xlim([-r, r])
plt.ylim([-r, r])
plt.axis('equal')
plt.xlabel('x')
plt.ylabel('y')
plt.show()