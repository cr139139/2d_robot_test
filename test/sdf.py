import matplotlib.pyplot as plt
import torch
import math


def sdf(p, shape="circle"):
    if shape == 'circle':
        radius = 0.25
        length = torch.norm(p, dim=1, keepdim=True)
        return length - radius
    elif shape == 'box':
        b = torch.tensor([0.25, 0.25])
        d = torch.abs(p) - b
        return torch.norm(torch.clamp(d, min=0.0), dim=1, keepdim=True) \
            + torch.clamp(torch.max(d, dim=1, keepdim=True)[0], max=0.0)


def grasp_sample(R=torch.eye(2), t=torch.zeros(2), shape="circle"):
    grasp_R = torch.tensor([[[1., 0.],
                             [0., 1.]]])
    grasp_t = torch.tensor([[-0.5, 0.]])

    theta = math.pi / 4
    c = math.cos(theta)
    s = math.sin(theta)
    generate_R = torch.tensor([[c, -s],
                               [s, c]])

    for i in range(7):
        grasp_R = torch.concatenate([grasp_R, (generate_R @ grasp_R[-1])[None, :, :]])
        grasp_t = torch.concatenate([grasp_t, (generate_R @ grasp_t[-1])[None, :]])

    grasp_R = R @ grasp_R
    grasp_t = (R @ grasp_t[:, :, None])[:, :, 0] + t

    if shape == 'circle':
        grasp_success = torch.ones(8)
    elif shape == 'box':
        grasp_success = torch.tensor([1, 0]).repeat(4)

    return grasp_R, grasp_t, grasp_success


def transform(xy, R=torch.eye(2), t=torch.zeros(2)):
    return (R @ xy.T).T + t


def show_slice(sdf, grasp, R, t, shape, w=100, r=1):
    x = torch.linspace(-r, r, steps=w)
    y = torch.linspace(-r, r, steps=w)
    x_index, y_index = torch.meshgrid(x, y, indexing="xy")
    xy = torch.concatenate([x_index, y_index]).reshape((2, -1)).T

    query = transform(xy, R.T, -R.T @ t)
    query.requires_grad_(True)
    d = sdf(query, shape)

    grad = torch.autograd.grad(d, query, grad_outputs=torch.ones_like(d), create_graph=True)[0]
    query.requires_grad_(False)
    d = d.detach().numpy()
    grad = transform(grad.detach().numpy(), R)

    hand_points = torch.tensor([[0, -0.1],
                                [0, 0.1],
                                [0.1, -0.1],
                                [0.1, 0.1]])
    hand_edges = [[0, 1], [0, 2], [1, 3]]

    grasp_R, grasp_t, grasp_success = grasp(R, t, shape)

    kw = dict(extent=(-r, r, -r, r), vmin=-r, vmax=r)
    contour = plt.contourf(d.reshape((w, w)), 10, cmap='coolwarm', **kw)
    plt.quiver(xy[:, 0], xy[:, 1], grad[:, 0], grad[:, 1], alpha=0.3)
    plt.contour(d.reshape((w, w)), levels=[0.0], colors='black', **kw)

    for i in range(grasp_R.shape[0]):
        if grasp_success[i] == 0:
            continue
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


# theta = math.pi / 3
# c = math.cos(theta)
# s = math.sin(theta)
# R = torch.tensor([[c, -s],
#                   [s, c]])
# t = torch.tensor([0.5, .5])
# show_slice(sdf, grasp_sample, R, t, shape="circle")
