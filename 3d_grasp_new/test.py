import torch
from normalizing_flow import NormalizingFlow, PointNetEncoder
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps

from relie import (
    SO3ExpTransform,
    SO3MultiplyTransform,
    LocalDiffeoTransformedDistribution as LDTD,
)
import relie

pcdencoder = PointNetEncoder(channel=6)
normalizingflow = NormalizingFlow(6, 6, 1024)
PATH = './weights/epoch171.pth'
checkpoint = torch.load(PATH)
pcdencoder.load_state_dict(checkpoint['pcdencoder_state_dict'])
normalizingflow.load_state_dict(checkpoint['normalizingflow_state_dict'])
torch.manual_seed(0)
print(torch.cuda.is_available())

batch_size = 128
alg_loc = torch.zeros((batch_size, 3), dtype=torch.double)
scale = torch.ones((batch_size, 3), dtype=torch.double)
loc = relie.utils.so3_tools.so3_exp(alg_loc)

alg_distr = torch.distributions.Normal(torch.zeros_like(scale), scale)
transforms = [SO3ExpTransform(k_max=3), SO3MultiplyTransform(loc)]
group_distr = LDTD(alg_distr, transforms)
z = group_distr.rsample()

dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(training_loader):
    object_info, grasp_vee = data
    import time

    start = time.time()
    object_latent = pcdencoder.forward(object_info)
    x, ldj = normalizingflow.forward(z, object_latent, reverse=True)

    print(time.time() - start)
    xyz = object_info[0]
    H = relie.utils.se3_tools.se3_exp(x)
    colors = torch.zeros_like(H[:, :3, -1])
    colors[:, 0] = 1
    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)
    # scene = visualize_grasps(Hs=H_real.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1.0)
    if i == 1:
        break
