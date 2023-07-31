import torch
from dataloader import GraspDataset
from models import GraspDistField, GraspSampler
import so3
from functools import partial
import torch.optim.lr_scheduler as lr_scheduler

dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
# model = GraspDistField()
model = GraspSampler()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)

epochs = 1001
flag = 0
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
model.to(device)

for epoch_index in range(epochs):
    running_loss = 0.
    if flag == 1:
        break
    model.train()
    for i, data in enumerate(training_loader):
        optimizer.zero_grad()
        pcd, grasp_wv, dist, ee_wv = data
        pcd = pcd.to(device)
        grasp_wv = grasp_wv.to(device).requires_grad_()
        dist = dist.to(device)
        ee_wv = ee_wv.to(device).requires_grad_()

        # dist_pred = model.forward(pcd, ee_wv)
        # grad_loss = torch.autograd.grad(dist_pred, ee_wv, grad_outputs=torch.ones_like(dist_pred), create_graph=True)[0]
        # grad_loss_norm = torch.linalg.norm(grad_loss, dim=1, keepdim=True)
        # ee_wv_update = ee_wv - dist_pred * grad_loss / grad_loss_norm

        ee_wv_update = model.forward(pcd, ee_wv)

        loss = 0
        # loss += (torch.abs(dist - dist_pred)).mean() * 100
        loss += torch.sum(torch.abs(grasp_wv[:, :] - ee_wv_update[:, :]), dim=1).mean()
        loss += torch.sum((grasp_wv[:, :] - ee_wv_update[:, :]) ** 2, dim=1).mean()
        # loss += (torch.abs(grad_loss_norm - 1)).mean()

        if torch.isnan(loss).any():
            flag = 1

        loss.backward()
        optimizer.step()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))
    # scheduler.step()

    if epoch_index % 10 == 0:
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
