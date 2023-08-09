import torch
import so3
from normalizing_flow import NormalizingFlow2
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps

from relie import (
    SO3ExpTransform,
    SO3MultiplyTransform,
    LocalDiffeoTransformedDistribution as LDTD,
)
import relie

model = NormalizingFlow2(1024)
PATH = './weights/epoch201.pth'
checkpoint = torch.load(PATH)
model.load_state_dict(checkpoint['model_state_dict'])
# torch.manual_seed(0)
print(torch.cuda.is_available())

batch_size = 20
alg_loc = torch.zeros((batch_size, 3), dtype=torch.double)
scale = torch.ones((batch_size, 3), dtype=torch.double) * .1
loc = relie.utils.so3_tools.so3_exp(alg_loc)

alg_distr = torch.distributions.Normal(torch.zeros_like(scale), scale)
transforms = [SO3ExpTransform(k_max=3), SO3MultiplyTransform(loc)]
group_distr = LDTD(alg_distr, transforms)
R_samples = group_distr.rsample().to(torch.float)
x_samples = alg_distr.rsample().to(torch.float)

dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(training_loader):
    object_info, H_real, idx = data
    import time

    start = time.time()
    # R, t, log_jacobs, log_pz = model.forward(H_real[:, :3, :3], H_real[:, :3, 3], object_info.repeat(1, 1, 1))
    # grasp_R, grasp_t = model.forward(R_samples, x_samples, object_info, idx, inverse=True)
    grasp_R, grasp_t = model.forward(R_samples, x_samples, object_info.repeat(batch_size, 1, 1), inverse=True)
    print(time.time() - start)
    xyz = object_info[0, :, :3]
    # xyz = torch.concatenate([object_info[0, :, :, 0], object_info[0, :, :, 1]], dim=0)

    H = torch.eye(4)[None, ...].repeat(batch_size, 1, 1)
    H[:, :3, :3] = grasp_R
    H[:, :3, 3] = grasp_t

    colors = torch.zeros_like(H[:, :3, -1])
    colors[:, 0] = 1
    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)
    # scene = visualize_grasps(Hs=H_real.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1.0)
    if i == 5:
        break
