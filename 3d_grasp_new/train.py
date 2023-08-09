import torch
from dataloader import GraspDataset
from normalizing_flow import NormalizingFlow, PointNetEncoder

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
import relie

# torch.set_default_dtype(torch.double)
# torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)

batch_size = 256
dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

pcdencoder = PointNetEncoder(channel=6)
normalizingflow = NormalizingFlow(12, 6, 1024)
optimizer = torch.optim.Adam(list(pcdencoder.parameters()) + list(normalizingflow.parameters()), lr=1e-4)

pcdencoder.to(device)
normalizingflow.to(device)

epochs = 1000

for epoch_index in range(epochs):
    running_loss = 0.
    for i, data in enumerate(training_loader):
        object_info, grasp_vee = data
        object_info = object_info.to(device)
        grasp_vee = grasp_vee.to(device)

        B = grasp_vee.size(0)
        alg_loc = torch.zeros((B, 3), dtype=torch.double).to(device)
        scale = torch.ones((B, 3), dtype=torch.double).to(device)
        loc = so3_exp(alg_loc)
        alg_distr = torch.distributions.Normal(torch.zeros_like(scale), scale)
        transforms = [SO3ExpTransform(k_max=3), SO3MultiplyTransform(loc)]
        group_distr = LDTD(alg_distr, transforms)

        optimizer.zero_grad()

        object_latent = pcdencoder.forward(object_info)
        z, ldj = normalizingflow.forward(grasp_vee, object_latent)
        if torch.isnan(z).any() or torch.isnan(ldj).any():
            continue
        z = z.to(torch.double)
        ldj = ldj.to(torch.double)
        Z = so3_exp(z[:, :3])
        ldj_rot = so3_log_abs_det_jacobian(z[:, :3]).to(torch.float)
        ldz_rot = group_distr.log_prob(Z).to(torch.float)
        ldz_trn = alg_distr.log_prob(z[:, 3:]).sum(1).to(torch.float)
        loss = -(ldj + ldj_rot + ldz_rot + ldz_trn).mean()

        if not torch.isnan(loss).any():
            loss.backward()
            optimizer.step()
        # print(ldj.mean().item(), ldj_rot.mean().item(), ldz_rot.mean().item(), ldz_trn.mean().item())
        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 5 == 0:
        torch.save({
            'epoch': epoch_index + 1,
            'pcdencoder_state_dict': pcdencoder.state_dict(),
            'normalizingflow_state_dict': normalizingflow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
