import torch
from dataloader import GraspDataset
from normalizing_flow import NormalizingFlow

dataset = GraspDataset()
training_loader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=True)
model = NormalizingFlow(6)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 100000
flag = 0
torch.autograd.set_detect_anomaly(True)
for epoch_index in range(epochs):
    running_loss = 0.
    if flag == 1:
        break
    for i, data in enumerate(training_loader):
        grasp_R, grasp_t, grasp_success, object_info = data
        optimizer.zero_grad()
        R, t, log_jacobs = model.forward(grasp_R, grasp_t, object_info)
        R_ps, t_ps = model.inverse(R, t, object_info)

        loss = (-grasp_success * log_jacobs).mean()
        negative = torch.exp(((1-grasp_success) * log_jacobs).mean())
        loss += negative

        if torch.isnan(loss).any():
            flag = 1

        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        print('epoch {} batch {} loss: {}'.format(epoch_index + 1, i + 1, loss.item()))

    if epoch_index % 1000 == 0:
        torch.save({
            'epoch': epoch_index,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss.item(),
        }, './weights/epoch{}.pth'.format(epoch_index + 1))
