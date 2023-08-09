import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

link_lengths = np.array([2, 1, 1])
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


fig, ax = plt.subplots(1, 1)
current_joint = np.zeros(n_dof)
target_pose = np.array([-2, 2, np.pi / 2])
ax.scatter(target_pose[0], target_pose[1], c='r')
ax.quiver(target_pose[0], target_pose[1], np.cos(target_pose[2]), np.sin(target_pose[2]), angles='uv', scale=2,
          scale_units='xy', color='r')
xs, ys, thetas = fk(current_joint)

joint_origin = np.zeros(2)
joint_draw = []
for i in range(n_dof):
    joint_draw.append(ax.plot([joint_origin[0], xs[i]], [joint_origin[1], ys[i]], 'o-k'))
    joint_origin = np.array([xs[i], ys[i]])

hand_points = np.array([[0, -0.5],
                        [0, 0.5],
                        [0.5, -0.5],
                        [0.5, 0.5]])
hand_edges = [[0, 1], [0, 2], [1, 3]]

hand_cos = np.cos(thetas[-1])
hand_sin = np.sin(thetas[-1])
hand_R = np.array([[hand_cos, -hand_sin],
                   [hand_sin, hand_cos]])
hand_points = hand_points @ hand_R.T
hand_points[:, 0] += xs[-1]
hand_points[:, 1] += ys[-1]
hand_draw = []
for i in range(len(hand_edges)):
    hand_draw.append(ax.plot(hand_points[hand_edges[i], 0], hand_points[hand_edges[i], 1], 'k'))

ax.set_xlim([-6, 6])
ax.set_ylim([-6, 6])
ax.set_aspect('equal')
ax.grid()


def update(frame):
    global current_joint
    xs, ys, thetas = fk(current_joint)
    current_pose = np.array([xs[-1], ys[-1], thetas[-1]])
    J = Jacobian(current_joint)
    lmbda = 0.01
    q_delta = np.linalg.inv(J.T @ J + lmbda * np.eye(n_dof)) @ J.T @ (target_pose - current_pose)
    current_joint += q_delta * 0.01

    joint_origin = np.zeros(2)
    for i in range(n_dof):
        joint_draw[i][0].set_xdata([joint_origin[0], xs[i]])
        joint_draw[i][0].set_ydata([joint_origin[1], ys[i]])
        joint_origin = np.array([xs[i], ys[i]])

    hand_points = np.array([[0, -0.5],
                            [0, 0.5],
                            [0.5, -0.5],
                            [0.5, 0.5]])
    hand_edges = [[0, 1], [0, 2], [1, 3]]

    hand_cos = np.cos(thetas[-1])
    hand_sin = np.sin(thetas[-1])
    hand_R = np.array([[hand_cos, -hand_sin],
                       [hand_sin, hand_cos]])
    hand_points = hand_points @ hand_R.T
    hand_points[:, 0] += xs[-1]
    hand_points[:, 1] += ys[-1]
    for i in range(len(hand_edges)):
        hand_draw[i][0].set_xdata([hand_points[hand_edges[i], 0]])
        hand_draw[i][0].set_ydata([hand_points[hand_edges[i], 1]])
    return joint_draw + hand_draw


ani = animation.FuncAnimation(fig=fig, func=update, frames=1000, interval=30)
# f = r"animation.gif"
# writergif = animation.PillowWriter(fps=30)
# ani.save(f, writer=writergif)
plt.show()
