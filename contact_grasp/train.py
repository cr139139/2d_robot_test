import torch
import torch.nn.functional as F
from dataloader import GraspDataset
from model import GraspSampler

dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=256, shuffle=False)

grasp_model = GraspSampler()
optimizer = torch.optim.Adam(grasp_model.paramters(), lr=1e-3)

epochs = 1000
torch.autograd.set_detect_anomaly(True)

if torch.cuda.is_available():
    dev = "cuda:0"
else:
    dev = "cpu"
device = torch.device(dev)
grasp_model.to(device)

for epoch_index in range(epochs):
    running_loss = 0.
    grasp_points = torch.from_numpy(dataset.grasp_points).to(torch.float).to(device)
    for i, data in enumerate(training_loader):
        pcd, closest_contact_point, corresponding_point, a_gt, grasp_points1, grasp_points2 = data
        pcd = pcd.to(device)
        closest_contact_point = closest_contact_point.to(device)
        corresponding_point = corresponding_point.to(device)
        a_gt = a_gt.to(device)
        grasp_points1 = grasp_points1.to(device)
        grasp_points2 = grasp_points2.to(device)

        optimizer.zero_grad()

        B = a_gt.size(0)

        c1, c2, R, t = grasp_model.forward(pcd)

        c1_loss = F.mse_loss(c1, closest_contact_point)
        c2_loss = F.mse_loss(c2, closest_contact_point)

        grasp_points_pred = R @ grasp_points + t[:, :, :, None]
        grasp_points_loss1 = ((grasp_points1 - grasp_points_pred) ** 2).sum(2, 3)
        grasp_points_loss2 = ((grasp_points2 - grasp_points_pred) ** 2).sum(2, 3)
        grasp_points_loss = torch.minimum(grasp_points_loss1, grasp_points_loss2).mean() * 10

        loss = c1_loss + c2_loss + grasp_points_loss

        if not torch.isnan(loss):
            loss.backward()
            optimizer.step()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 10 == 0:
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': grasp_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
