import torch
import so3
from models import GraspDistFieldDec
from pcn import PCNENC
from dataloader import GraspDataset
from grasp_visualization import visualize_grasps

encoder = PCNENC()
decoder = GraspDistFieldDec()
PATH = './weights/epoch1.pth'
checkpoint = torch.load(PATH)
decoder.load_state_dict(checkpoint['model_state_dict'])
# torch.manual_seed(0)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device("cpu")
encoder.to(device)
decoder.to(device)

B = 20
dataset = GraspDataset(512)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False)

for i, data in enumerate(training_loader):
    pcd, grasp_wv, dist, ee_wv = data
    pcd = pcd.to(device)
    grasp_wv = grasp_wv.to(device)
    dist = dist.to(device)
    ee_wv = ee_wv.to(device).requires_grad_()
    import time
    start = time.time()

    P = encoder(pcd)

    ee_wv_update = ee_wv.clone()
    for _ in range(1):
        dist_pred = decoder.forward(P, ee_wv_update)
        grad_loss = torch.autograd.grad(dist_pred, ee_wv_update, grad_outputs=torch.ones_like(dist_pred), create_graph=True)[0]
        ee_wv_update = ee_wv - grad_loss
    grasp_pred_R = so3.exp_map(ee_wv_update[:, :3], device=device)
    grasp_pred_t = ee_wv_update[:, 3:]

    print(ee_wv)
    print(ee_wv_update)
    print(grasp_wv)
    print(dist_pred)
    print(dist)

    xyz = pcd[0]
    H = torch.eye(4)[None, ...].repeat(3, 1, 1)
    H[0, :3, :3] = grasp_pred_R
    H[0, :3, 3] = grasp_pred_t
    H[1, :3, :3] = so3.exp_map(ee_wv[:, :3])
    H[1, :3, 3] = ee_wv[:, 3:]
    H[2, :3, :3] = so3.exp_map(grasp_wv[:, :3])
    H[2, :3, 3] = grasp_wv[:, 3:]

    colors = torch.zeros_like(H[:, :3, -1])
    colors[0, 0] = 1
    colors[1, 1] = 1
    colors[2, 2] = 1
    scene = visualize_grasps(Hs=H.detach().numpy(), colors=colors, p_cloud=xyz, show=True, scale=1)
    if i == 5:
        break
