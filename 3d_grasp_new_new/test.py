import torch
from normalizing_flow import TranslationFlow, OrientationFlow, PointNetEncoder
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps
import numpy as np
from pointnet2_cls_ssg import pointnet2_encoder
from relie import (
    SO3ExpTransform,
    SO3MultiplyTransform,
    LocalDiffeoTransformedDistribution as LDTD,
)
import relie

encoder = PointNetEncoder()
# encoder = pointnet2_encoder()
tflow = TranslationFlow(1024)
oflow = OrientationFlow(1024 + 3)

PATH = './weights/epoch991.pth'
checkpoint = torch.load(PATH)
encoder.load_state_dict(checkpoint['encoder_state_dict'])
tflow.load_state_dict(checkpoint['tflow_state_dict'])
oflow.load_state_dict(checkpoint['oflow_state_dict'])
# torch.manual_seed(0)
print(torch.cuda.is_available())

batch_size = 100
alg_loc = torch.zeros((batch_size, 3), dtype=torch.double)
scale = torch.ones((batch_size, 3), dtype=torch.double) * 1
loc = relie.utils.so3_tools.so3_exp(alg_loc)

alg_distr = torch.distributions.Normal(torch.zeros_like(scale), scale)
transforms = [SO3ExpTransform(k_max=3), SO3MultiplyTransform(loc)]
group_distr = LDTD(alg_distr, transforms)
R_samples = group_distr.rsample().to(torch.float)
t_samples = alg_distr.rsample().to(torch.float)


dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(training_loader):
    object_info, H_real, idx = data
    import time

    start = time.time()
    shape_latent = encoder.forward(object_info)
    shape_latent = shape_latent.repeat(batch_size, 1)
    t = tflow.forward(t_samples, shape_latent, inverse=True)
    R = oflow.forward(R_samples, torch.cat([shape_latent, t], dim=1), inverse=True)
    print(time.time() - start)
    xyz = object_info[0, :, :3]
    H = torch.eye(4)[None, ...].repeat(batch_size, 1, 1)
    H[:, :3, :3] = R
    H[:, :3, 3] = t

    colors = torch.zeros_like(H[:, :3, -1])
    colors[:, 0] = 1
    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)

    # xyz_all = torch.cat([xyz, H[:, :3, 3]], dim=0)
    # scene = visualize_grasps(Hs=np.empty((0, 4, 4)), colors=colors, p_cloud=xyz_all.detach().numpy(), show=True, scale=1)
    # scene = visualize_grasps(Hs=H_real.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1.0)
    if i == 0:
        break
