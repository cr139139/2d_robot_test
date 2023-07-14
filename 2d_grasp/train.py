import torch
from dataloader import GraspDataset
from normalizing_flow import NormalizingFlow
import so3

combined_dataset = torch.utils.data.ConcatDataset([GraspDataset(shape='circle'), GraspDataset('box')])
training_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=16, shuffle=True)
model = NormalizingFlow(64)
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
        grasp_R, grasp_t, grasp_success, object_info, R, t = data
        optimizer.zero_grad()
        Rs, ts, log_jacobs, log_pz, Rc, tc = model.forward(grasp_R, grasp_t, object_info)
        loss = 0
        total = torch.sum(grasp_success)
        loss += (-grasp_success * (log_pz + log_jacobs)).mean() / total
        loss -= (grasp_success * log_jacobs ** 2).mean() * 0.1 / total
        loss += (grasp_success * (torch.norm(Rc @ R.transpose(1, -1)) + torch.norm(tc[:, :, 0] - t))).mean()
        # loss += torch.norm(torch.eye(3)[None, :, :].repeat(16, 1, 1) - Rc @ Rc.transpose(1, -1)).mean()

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
