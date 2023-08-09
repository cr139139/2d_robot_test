import torch
import relie
vee = torch.tensor([[0, 0, 0]], dtype=torch.double)

print(relie.utils.so3_tools.so3_exp(vee))