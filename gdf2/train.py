import torch
from dataloader import GraspDataset
from models import GraspSampler2
import so3
from functools import partial
import torch.optim.lr_scheduler as lr_scheduler

dataset = GraspDataset(1024)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=True)
model = GraspSampler2()
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
        pcd, grasp_T, ee_T = data
        pcd = pcd.to(device)
        grasp_T = grasp_T.to(device)
        ee_T = ee_T.to(device)

        grasp_T_pred = model.forward(pcd, ee_T, device)

        B = ee_T.size(0)
        # print(grasp_T_pred[0], grasp_T[0])
        loss = torch.sum(
            (grasp_T.inverse() @ grasp_T_pred - torch.eye(4).reshape((1, 4, 4)).repeat(B, 1, 1).to(device)) ** 2,
            dim=[1, 2]).mean()
        loss + torch.sum(
            ((grasp_T.inverse() @ grasp_T_pred).inverse() - torch.eye(4).reshape((1, 4, 4)).repeat(B, 1, 1).to(device)) ** 2,
            dim=[1, 2]).mean()

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
