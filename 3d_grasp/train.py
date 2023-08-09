import torch
from dataloader import GraspDataset
from normalizing_flow import NormalizingFlow2

from relie import (
    SO3ExpTransform,
    SO3MultiplyTransform,
    LocalDiffeoTransformedDistribution as LDTD,
)
from relie.utils.so3_tools import (
    so3_exp,
    so3_log,
    so3_vee,
    so3_xset,
    so3_log_abs_det_jacobian,
)

dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
model = NormalizingFlow2(1024)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 1000
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
model.to(device)

for epoch_index in range(epochs):
    running_loss = 0.
    for i, data in enumerate(training_loader):
        object_info, grasp_T, idx = data
        object_info = object_info.to(device)
        grasp_T = grasp_T.to(device)

        B = grasp_T.size(0)
        alg_loc = torch.zeros((B, 3), dtype=torch.double).to(device)
        scale = torch.ones((B, 3), dtype=torch.double).to(device)
        loc = so3_exp(alg_loc)
        alg_distr = torch.distributions.Normal(torch.zeros_like(scale), scale)
        transforms = [SO3ExpTransform(k_max=3), SO3MultiplyTransform(loc)]
        group_distr = LDTD(alg_distr, transforms)

        optimizer.zero_grad()

        R, t, log_jacobs = model.forward(grasp_T[:, :3, :3], grasp_T[:, :3, 3], object_info, device=device)
        I = torch.eye(3).reshape(1, 3, 3).repeat(B, 1, 1).to(device)
        dtheta = (R - I).flatten(1)
        dx = t
        dist = torch.cat((dx, dtheta), dim=-1)
        log_pz = -.5 * dist.pow(2).sum(-1) / (1 ** 2)
        loss = -(log_pz + log_jacobs).mean()

        if not torch.isnan(loss):
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
