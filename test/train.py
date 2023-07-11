import torch
from dataloader import GraspDataset
from normalizing_flow_sdf import NormalizingFlow
import so3

dataset = GraspDataset()
training_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
model = NormalizingFlow()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 100000
flag = 0
torch.autograd.set_detect_anomaly(True)


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


for epoch_index in range(epochs):
    running_loss = 0.
    if flag == 1:
        break
    for i, data in enumerate(training_loader):
        grasp_R, grasp_t, grasp_success, object_info = data
        optimizer.zero_grad()
        Rs, ts, log_jacobs, log_pz = model.forward(grasp_R, grasp_t, object_info)
        loss = 0
        loss += (-grasp_success * (log_pz + log_jacobs)).mean()
        # loss -= (grasp_success * log_jacobs ** 2).mean() * 0.1

        # negative = torch.exp(((1 - grasp_success) * log_jacobs).mean())
        # loss += negative

        B = 8
        std = 0.01 * torch.ones(B)
        x_samples, R_samples = sample_from_se3_gaussian(ts[-1], Rs[-1], std)
        grasp_Rs, grasp_ts = model.inverse(R_samples, x_samples, object_info)
        loss += (grasp_success * (torch.norm(grasp_Rs[-1] @ grasp_R.transpose(1, -1)) + torch.norm(grasp_ts[-1] - grasp_t))).mean()

        # B = 8
        # R_mu = torch.eye(3).repeat(B, 1, 1)
        # x_mu = torch.zeros(3).repeat(B, 1)
        # std = 1 * torch.ones(B)
        # x_samples, R_samples = sample_from_se3_gaussian(x_mu, R_mu, std)
        # grasp_Rs, grasp_ts = model.inverse(R_samples, x_samples, object_info)
        # T1 = len(grasp_Rs)
        #
        # for i in range(1, T1):
        #     Rt = so3.exp_map(so3.log_map(grasp_R @ R_samples.transpose(1, -1)) * i / T1) @ grasp_R
        #     tt = (grasp_t - x_samples) * i / T1 + grasp_t
        #     loss += (torch.norm(grasp_Rs[i] @ Rt.transpose(1, -1)) + torch.norm(grasp_ts[i] - tt)) / (T1 - 1)

        if torch.isnan(loss).any():
            flag = 1

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 1000 == 0:
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
