import torch
from dataloader import GraspDataset
from normalizing_flow import TranslationFlow, OrientationFlow, PointNetEncoder
from pointnet2_cls_ssg import pointnet2_encoder

dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
encoder = PointNetEncoder()
# encoder = pointnet2_encoder()
tflow = TranslationFlow(1024)
oflow = OrientationFlow(1024 + 3)
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(tflow.parameters()) + list(oflow.parameters()), lr=1e-3)

epochs = 1000
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
encoder.to(device)
tflow.to(device)
oflow.to(device)

for epoch_index in range(epochs):
    running_loss = 0.
    for i, data in enumerate(training_loader):
        object_info, grasp_T, idx = data
        object_info = object_info.to(device)
        grasp_T = grasp_T.to(device)
        optimizer.zero_grad()

        B = grasp_T.size(0)

        shape_latent = encoder.forward(object_info)
        t, log_j_t = tflow.forward(grasp_T[:, :3, 3], shape_latent, device=device)
        R, log_j_o = oflow.forward(grasp_T[:, :3, :3], torch.cat([shape_latent, grasp_T[:, :3, 3]], dim=1), device=device)

        I = torch.eye(3).reshape(1, 3, 3).repeat(B, 1, 1).to(device)
        dtheta = (R - I).flatten(1)
        dx = t
        dist = torch.cat((dx, dtheta), dim=-1)
        log_pz = -.5 * dist.pow(2).sum(-1) / (1 ** 2)
        loss = -(log_pz + log_j_t + log_j_o).mean()

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 10 == 0:
        torch.save({
            'epoch': epoch_index,
            'encoder_state_dict': encoder.state_dict(),
            'tflow_state_dict': tflow.state_dict(),
            'oflow_state_dict': oflow.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
