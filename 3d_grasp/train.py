import torch
from dataloader import GraspDataset
from normalizing_flow import NormalizingFlow

dataset = GraspDataset()
training_loader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
model = NormalizingFlow(64)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 1000
flag = 0
torch.autograd.set_detect_anomaly(True)


for epoch_index in range(epochs):
    running_loss = 0.
    if flag == 1:
        break
    for i, data in enumerate(training_loader):
        object_info, grasp_T = data
        optimizer.zero_grad()
        Rs, ts, log_jacobs, log_pz = model.forward(grasp_T[:, :3, :3], grasp_T[:, :3, 3], object_info)
        loss = 0
        loss += (-log_pz + log_jacobs).mean()
        loss -= (log_jacobs ** 2).mean() * 0.1

        if torch.isnan(loss).any():
            flag = 1

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 10 == 0:
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
