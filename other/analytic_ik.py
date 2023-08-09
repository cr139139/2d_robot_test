import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

link_lengths = np.array([3, 2])
n_dof = len(link_lengths)


def fk(joint_positions):
    thetas = np.cumsum(joint_positions)
    xs = np.cumsum(np.cos(thetas) * link_lengths)
    ys = np.cumsum(np.sin(thetas) * link_lengths)
    return xs, ys, thetas


def Jacobian(joint_positions):
    thetas = np.cumsum(joint_positions)
    J_x = np.cumsum(-(np.sin(thetas) * link_lengths)[::-1])[::-1]
    J_y = np.cumsum((np.cos(thetas) * link_lengths)[::-1])[::-1]
    J_theta = np.ones(len(joint_positions))
    return np.stack([J_x, J_y, J_theta], axis=0)


def ik_2d(x, y):
    temp = (x ** 2 + y ** 2 - link_lengths[0] ** 2 - link_lengths[1] ** 2) / (2 * link_lengths[0] * link_lengths[1])
    q2_d = np.arccos(temp)
    q1_d = np.arctan2(y, x) - np.arctan2(
        link_lengths[1] * np.sin(q2_d), (link_lengths[0] + link_lengths[1] * np.cos(q2_d)))
    q2_u = -np.arccos(temp)
    q1_u = np.arctan2(y, x) - np.arctan2(
        link_lengths[1] * np.sin(q2_u), (link_lengths[0] + link_lengths[1] * np.cos(q2_u)))
    return np.array([[q1_d, q2_d],
                     [q1_u, q2_u]])


fig, axs = plt.subplots(1, 3, figsize=(12, 4))
current_joint = np.random.uniform(-np.pi, np.pi, size=(2,))
current_joint = np.array([0, 0])
target_joints = np.empty((0, 2))
thetas = np.linspace(-np.pi, np.pi, 30)

drawings = []
for theta in thetas:
    # target_pose = np.array([2 * np.cos(theta), 1 * np.sin(theta)])
    target_pose = np.array([16 * np.sin(theta) ** 3,
                            13 * np.cos(theta) - 5 * np.cos(2 * theta) - 2 * np.cos(3 * theta) - np.cos(
                                4 * theta)]) * 0.2
    target_joint = ik_2d(target_pose[0], target_pose[1])
    target_joints = np.concatenate([target_joints, target_joint], axis=0)

    for target in target_joint:
        xs, ys, _ = fk(target)

        joint_origin = np.zeros(2)
        for i in range(n_dof):
            drawings.append(axs[0].plot([joint_origin[0], xs[i]], [joint_origin[1], ys[i]], 'o-k'))
            joint_origin = np.array([xs[i], ys[i]])

    drawings.append(axs[0].scatter(target_pose[0], target_pose[1], c='r', zorder=3))
    drawings.append(axs[1].scatter(target_joint[:, 0], target_joint[:, 1], c='k', zorder=0))
    drawings.append(axs[2].scatter(target_pose[0], target_pose[1], c='r'))

from gpis import GPIS, queryOne

target_joints = np.concatenate([target_joints, np.zeros((target_joints.shape[0], 1))], axis=1)
xg, yg, zg, xd, yd, zd, mean, colors = GPIS(target_joints)

xs, ys, _ = fk(current_joint)
joint_origin = np.zeros(2)
joint_draw = []
for i in range(n_dof):
    joint_draw.append(axs[2].plot([joint_origin[0], xs[i]], [joint_origin[1], ys[i]], 'o-k'))
    joint_origin = np.array([xs[i], ys[i]])
pose_draw = axs[1].scatter(current_joint[0], current_joint[1], c='r', zorder=2)
target_draw = axs[1].scatter([], [], c='c', zorder=1)

xs, ys, _ = fk(current_joint)
joint_origin = np.zeros(2)
target_joint_draw = []
for i in range(n_dof):
    target_joint_draw.append(axs[2].plot([joint_origin[0], xs[i]], [joint_origin[1], ys[i]], 'o-c', zorder=1))
    joint_origin = np.array([xs[i], ys[i]])

axs[0].set_xlim([-6, 6])
axs[0].set_ylim([-6, 6])
axs[0].set_aspect('equal')
axs[0].grid()
axs[0].set_title('Target poses and IK')
axs[0].set_xlabel('x')
axs[0].set_ylabel('y')

axs[1].set_xlim([-np.pi, np.pi])
axs[1].set_ylim([-np.pi, np.pi])
axs[1].set_aspect('equal')
axs[1].grid()
axs[1].quiver(xg, yg, xd, yd, colors, angles='xy', scale=100, cmap='hsv', zorder=-1)
axs[1].set_title('Joint space')
axs[1].set_xlabel('q1')
axs[1].set_ylabel('q2')

axs[2].set_xlim([-6, 6])
axs[2].set_ylim([-6, 6])
axs[2].set_aspect('equal')
axs[2].grid()
axs[2].set_title('Cartesian space')
axs[2].set_xlabel('x')
axs[2].set_ylabel('y')


def update(frame):
    global current_joint
    mean, grad = queryOne(target_joints, np.concatenate([current_joint, np.zeros(1)])[None, :])
    target_joint = current_joint - grad[:2, 0] * mean[0, 0]
    current_joint = current_joint - 0.1 * grad[:2, 0] * mean[0, 0]
    xs, ys, thetas = fk(current_joint)

    joint_origin = np.zeros(2)
    for i in range(len(joint_draw)):
        joint_draw[i][0].set_xdata([joint_origin[0], xs[i]])
        joint_draw[i][0].set_ydata([joint_origin[1], ys[i]])
        joint_origin = np.array([xs[i], ys[i]])

    pose_draw.set_offsets(current_joint)
    target_draw.set_offsets(target_joint)
    xs, ys, thetas = fk(target_joint)

    joint_origin = np.zeros(2)
    for i in range(len(target_joint_draw)):
        target_joint_draw[i][0].set_xdata([joint_origin[0], xs[i]])
        target_joint_draw[i][0].set_ydata([joint_origin[1], ys[i]])
        joint_origin = np.array([xs[i], ys[i]])

    return joint_draw + [pose_draw]


ani = animation.FuncAnimation(fig=fig, func=update, frames=100, interval=1)
f = r"animation2.gif"
writergif = animation.PillowWriter(fps=30)
ani.save(f, writer=writergif)
plt.show()
