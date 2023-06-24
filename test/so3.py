import torch
import math


def rand():
    def generate_random_z_axis_rotation():
        """Generate random rotation matrix about the z axis."""
        R = torch.eye(3)
        x1 = torch.rand(1)
        R[0, 0] = R[1, 1] = torch.cos(2 * math.pi * x1)
        R[0, 1] = -torch.sin(2 * math.pi * x1)
        R[1, 0] = torch.sin(2 * math.pi * x1)
        return R

    # There are two random variables in [0, 1) here (naming is same as paper)
    x2 = 2 * math.pi * torch.rand(1)
    x3 = torch.rand(1)
    # Rotation of all points around x axis using matrix
    R = generate_random_z_axis_rotation()
    v = torch.tensor([
        torch.cos(x2) * torch.sqrt(x3),
        torch.sin(x2) * torch.sqrt(x3),
        torch.sqrt(1 - x3)
    ])
    H = torch.eye(3) - (2 * torch.outer(v, v))
    M = -(H @ R)
    return M


def exp_map(phi):
    angle = torch.norm(phi, dim=1, keepdim=True)

    def ordinary(phi, angle):
        n1, n2, n3 = (phi / angle).T
        c = torch.cos(angle)[:, 0]
        s = torch.sin(angle)[:, 0]
        R = torch.stack([c + n1 ** 2 * (1 - c),
                         n1 * n2 * (1 - c) - n3 * s,
                         n1 * n3 * (1 - c) + n2 * s,
                         n1 * n2 * (1 - c) + n3 * s,
                         c + n2 ** 2 * (1 - c),
                         n2 * n3 * (1 - c) - n1 * s,
                         n1 * n3 * (1 - c) - n2 * s,
                         n2 * n3 * (1 - c) + n1 * s,
                         c + n3 ** 2 * (1 - c)]).T
        return R

    def first_order_taylor(phi):
        w1, w2, w3 = phi.T
        ones = torch.ones(w1.shape)
        R = torch.stack([ones, -w3, w2,
                         w3, ones, -w1,
                         -w2, w1, ones]).T
        return R

    indexing = torch.isclose(angle, torch.zeros(angle.shape))

    R = torch.empty((phi.shape[0], 9))
    R[indexing[:, 0], :] = first_order_taylor(phi[indexing[:, 0], :])
    R[torch.logical_not(indexing)[:, 0], :] = ordinary(phi[torch.logical_not(indexing)[:, 0], :],
                                                       angle[torch.logical_not(indexing)[:, 0], :])
    R = R.reshape((-1, 3, 3))
    return R


def log_map(R):
    R = R.reshape((-1, 9))
    c = 0.5 * (R[:, 0] + R[:, 4] + R[:, 8]) - 0.5
    c = torch.clip(c, -1., 1.)
    angle = torch.acos(c)[:, None]

    def ordinary(R, angle):
        constant = 0.5 * angle / torch.sin(angle)
        phi = constant * torch.stack([R[:, 7] - R[:, 5], R[:, 2] - R[:, 6], R[:, 3] - R[:, 1]]).T
        return phi

    def first_order_taylor(R):
        phi = torch.stack([R[:, 7], R[:, 2], R[:, 3]]).T
        return phi

    indexing = torch.isclose(angle, torch.zeros(angle.shape))

    phi = torch.empty((R.shape[0], 3))
    phi[indexing[:, 0], :] = first_order_taylor(R[indexing[:, 0], :])
    phi[torch.logical_not(indexing)[:, 0], :] = ordinary(R[torch.logical_not(indexing)[:, 0], :],
                                                         angle[torch.logical_not(indexing)[:, 0], :])
    return phi


# R = rand()
# print(R)
# exp = exp_map(torch.tensor([[0., 1., 0.],
#                             [1., 0., 0.],
#                             [0., 0., 0.]]))
# log = log_map(exp)
# print(exp)
# print(log)

import numpy as np


