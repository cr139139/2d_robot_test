import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# parameter settings for RQ GPIS
alpha = 100
sigma = 1
l = 0.01

# parameter settings for log-GPIS
Lambda = 100
v = 3 / 2
scale = np.sqrt(2 * v)

# parameter setting for GP
noise = 0.05


def pair_dist(x1, x2):
    return np.linalg.norm(x2 - x1, axis=-1)


def cdist(x, y):
    return np.sqrt(np.sum((x[:, None] - y[None]) ** 2, axis=-1))


def pair_diff(x1, x2):
    return x1 - x2


def cdiff(x, y):
    return x[:, None] - y[None]


def cov(x1, x2, gpis=1):
    d = cdist(x1, x2)
    # RQ GPIS
    if gpis == 1:
        return sigma ** 2 * (1 + (d / l) ** 2 / (2 * alpha)) ** (-alpha)
    # log-GPIS
    elif gpis == 2:
        return (1 + d * (np.sqrt(2 * v)) * (Lambda / scale)) * np.exp(-1 * (d * (np.sqrt(2 * v)) * (Lambda / scale)))


def cov_grad(x1, x2, gpis=1):
    pair_diff = cdiff(x1, x2)
    d = np.linalg.norm(pair_diff, axis=-1)
    # RQ GPIS
    if gpis == 1:
        return (sigma ** 2 * (-alpha) * (1 + (d / l) ** 2 / (2 * alpha)) ** (-alpha - 1) * (
                d / (alpha * l ** 2))).reshape((x1.shape[0], x2.shape[0], 1)) * pair_diff
    # log-GPIS
    elif gpis == 2:
        return (- ((np.sqrt(2 * v)) * (Lambda / scale)) ** 2 * np.exp(
            -1 * (d * (np.sqrt(2 * v)) * (Lambda / scale)))).reshape((x1.shape[0], x2.shape[0], 1)) * pair_diff


def fibonacci_sphere(num_samples):
    points = []
    phi = np.pi * (3. - np.sqrt(5.))  # golden angle in radians
    y = np.linspace(1, -1, num_samples)
    radius = np.sqrt(1 - y * y)
    theta = phi * np.arange(num_samples)
    x = np.cos(theta) * radius
    z = np.sin(theta) * radius
    return x, y, z


def reverting_function(x, gpis=1):
    # RQ GPIS
    if gpis == 1:
        return np.sqrt(np.abs(2 * alpha * l ** 2 * ((x / sigma ** 2) ** (1 / (-alpha)) - 1)))
    # log-GPIS
    elif gpis == 2:
        return -(1 / Lambda) * np.log(np.abs(x))


def reverting_function_derivative(x, gpis=1):
    # RQ GPIS
    if gpis == 1:
        return (l ** 2 * (-(x / sigma ** 2) ** (-1 / alpha - 1)) / sigma ** 2) / reverting_function(x, gpis)
    # log-GPIS
    elif gpis == 2:
        return -1 / (Lambda * x)


model = None


def queryOne(points, query):
    global model
    # 1: RQ GPIS, 2: log-GPIS
    gpis = 1

    N_obs = points.shape[0]
    k = cov(query, points, gpis)

    if model is None:
        K = cov(points, points, gpis)
        y = np.ones((N_obs, 1))
        model = np.linalg.solve(K + noise * np.identity(N_obs), y)

    # distance inference
    mu = k @ model
    mean = reverting_function(mu, gpis)

    # gradient inference
    covariance_grad = cov_grad(query, points, gpis)
    mu_derivative = np.sum(covariance_grad * model[None, :, :], axis=1)
    grad = reverting_function_derivative(mu, gpis) * mu_derivative
    grad = grad.T

    # gradient normalization
    # norms = np.linalg.norm(grad, axis=0, keepdims=True)
    # grad = np.where(norms != 0, grad, grad / np.min(np.abs(grad), axis=0))
    grad /= np.linalg.norm(grad, axis=0, keepdims=True)
    # grad = np.where(norms != 0, grad / norms, grad)

    return mean, grad


def GPIS(points):
    global model
    # 1: RQ GPIS, 2: log-GPIS
    gpis = 1

    # using a 2D plane to query the distances
    start = -np.pi
    end = np.pi
    samples = int((end - start) / 0.05) + 1
    xg, yg = np.meshgrid(np.linspace(start, end, samples), np.linspace(start, end, samples))
    querySlice = np.column_stack((xg.reshape(-1), yg.reshape(-1), np.zeros(xg.shape).reshape(-1)))

    N_obs = points.shape[0]

    K = cov(points, points, gpis)
    k = cov(querySlice, points, gpis)
    y = np.ones((N_obs, 1))
    model = np.linalg.solve(K + noise * np.identity(N_obs), y)

    # distance inference
    mu = k @ model
    mean = reverting_function(mu, gpis)

    # gradient inference
    covariance_grad = cov_grad(querySlice, points, gpis)
    mu_derivative = np.sum(covariance_grad * model[None, :, :], axis=1)
    grad = reverting_function_derivative(mu, gpis) * mu_derivative
    grad = grad.T

    # gradient normalization
    # norms = np.linalg.norm(grad, axis=0, keepdims=True)
    # grad = np.where(norms != 0, grad, grad / np.min(np.abs(grad), axis=0))
    grad /= np.linalg.norm(grad, axis=0, keepdims=True)

    # GPIS visualization
    xg = querySlice[:, 0].reshape(xg.shape)
    yg = querySlice[:, 1].reshape(yg.shape)
    zg = np.zeros(xg.shape)

    xd = grad[0].reshape(xg.shape)
    yd = grad[1].reshape(yg.shape)
    zd = grad[2].reshape(xg.shape)
    colors = np.arctan2(xd, yd)

    mean = mean.reshape(xg.shape)

    return xg, yg, zg, xd, yd, zd, mean, colors


if __name__ == "__main__":
    # creating a sphere point cloud
    N_obs = 100  # number of observations
    sphereRadius = 1
    xa, yb, zc = fibonacci_sphere(N_obs)
    sphere = sphereRadius * np.column_stack((xa, yb, zc))

    xg, yg, zg, xd, yd, zd, mean, colors = GPIS(sphere)

    fig = plt.figure(figsize=(15, 5))
    fig.suptitle('GPIS result')

    # First subplot
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(xg, yg, mean, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.colorbar(surf, shrink=0.5, aspect=5, ax=ax)
    ax.set_title('Distance field')

    # Second subplot
    ax = fig.add_subplot(1, 2, 2)
    colormap = cm.inferno
    ax.scatter(sphere[:, 0], sphere[:, 1], alpha=0.05)
    ax.quiver(xg, yg, xd, yd, colors, angles='xy', scale=100)
    ax.set_aspect('equal')
    ax.set_title('Gradient field')

    plt.show()
