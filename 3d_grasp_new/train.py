import torch
from dataloader import GraspDataset
from normalizing_flow import NormalizingFlow, PointNetEncoder

from relie import (
    SO3ExpTransform,
    SO3MultiplyTransform,
    LocalDiffeoTransformedDistribution as LDTD,
)
import relie

batch_size = 128
dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)

pcdencoder = PointNetEncoder(channel=6)
normalizingflow = NormalizingFlow(6, 6, 1024)
optimizer = torch.optim.Adam(list(pcdencoder.parameters()) + list(normalizingflow.parameters()), lr=1e-4)

alg_loc = torch.zeros((batch_size, 3), dtype=torch.double)
scale = torch.ones((batch_size, 3), dtype=torch.double)
loc = relie.utils.so3_tools.so3_exp(alg_loc)

alg_distr = torch.distributions.Normal(torch.zeros_like(scale), scale)
transforms = [SO3ExpTransform(k_max=3), SO3MultiplyTransform(loc)]
group_distr = LDTD(alg_distr, transforms)

epochs = 1000
flag = 0
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
pcdencoder.to(device)
normalizingflow.to(device)

for epoch_index in range(epochs):
    running_loss = 0.
    if flag == 1:
        break
    for i, data in enumerate(training_loader):
        object_info, grasp_vee = data
        object_info = object_info.to(device)
        grasp_T = grasp_vee.to(device)

        optimizer.zero_grad()

        object_latent = pcdencoder.forward(object_info)
        z, ldj = normalizingflow.forward(grasp_vee, object_latent)
        ldz = group_distr.log_prob(z)

        loss = -(ldj + ldz).mean()

        if torch.isnan(loss).any():
            flag = 1

        loss.backward()
        optimizer.step()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 10 == 0:
        torch.save({
            'epoch': epoch_index,
            'pcdencoder_state_dict': pcdencoder.state_dict(),
            'normalizingflow_state_dict': normalizingflow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))

