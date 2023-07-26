import torch
from dataloader import GraspDataset
from normalizing_flow import NormalizingFlow

dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True)
model = NormalizingFlow(1024)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

epochs = 1000
flag = 0
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
model.to(device)

import so3


def sample_from_se3_gaussian(x_tar, R_tar, std, device):
    x_eps = std[:, None] * torch.randn_like(x_tar).to(device)
    theta_eps = std[:, None] * torch.randn_like(x_tar).to(device)
    rot_eps = so3.exp_map(theta_eps, device)
    _x = x_tar + x_eps
    _R = torch.einsum('bmn,bnk->bmk', R_tar.to(device), rot_eps.to(device))
    return _x, _R


for epoch_index in range(epochs):
    running_loss = 0.
    if flag == 1:
        break
    for i, data in enumerate(training_loader):
        object_info, grasp_T = data
        object_info = object_info.to(device)
        grasp_T = grasp_T.to(device)
        optimizer.zero_grad()
        R, t, log_jacobs, log_pz = model.forward(grasp_T[:, :3, :3], grasp_T[:, :3, 3], object_info, device=device)
        loss = -(log_pz + log_jacobs).mean()

        if torch.isnan(loss).any():
            flag = 1

        loss.backward()
        optimizer.step()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 10 == 0:
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
