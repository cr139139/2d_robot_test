import torch
from dataloader import GraspDataset

dataset = GraspDataset(256)
training_loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)
for i, data in enumerate(training_loader):
    # object_info, grasp_T, = data
    # print(grasp_T.size(), dist.size())
    break
