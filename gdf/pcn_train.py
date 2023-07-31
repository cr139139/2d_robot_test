import torch
from dataloader import GraspDataset
from models import GraspDistFieldDec
from pcn import PCNENC
import torch.optim.lr_scheduler as lr_scheduler

dataset = GraspDataset(512)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)
encoder = PCNENC()
decoder = GraspDistFieldDec()
optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-4)
scheduler = lr_scheduler.LinearLR(optimizer, start_factor=1.0, end_factor=0.1, total_iters=100)

epochs = 1000
flag = 0
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
encoder.to(device)
decoder.to(device)

for epoch_index in range(epochs):
    running_loss = 0.
    if flag == 1:
        break

    encoder.eval()
    decoder.train()
    for i, data in enumerate(training_loader):

        pcd, grasp_wv, dist, ee_wv = data
        pcd = pcd.to(device)
        grasp_wv = grasp_wv.to(device).requires_grad_()
        dist = dist.to(device)
        ee_wv = ee_wv.to(device).requires_grad_()

        optimizer.zero_grad()
        P = encoder(pcd)
        dist_pred = decoder.forward(P, ee_wv)
        grad_loss = torch.autograd.grad(dist_pred, ee_wv, grad_outputs=torch.ones_like(dist_pred), create_graph=True)[0]
        ee_wv_update = ee_wv - dist_pred * grad_loss

        loss = (torch.abs(dist - dist_pred)).mean()
        # loss = (torch.sum(torch.abs(grasp_wv[:, :] - ee_wv_update[:, :]), dim=1)).mean()

        if torch.isnan(loss).any():
            flag = 1

        loss.backward()
        optimizer.step()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))
    scheduler.step()

    if epoch_index % 10 == 0:
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': decoder.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
