from torch import nn
from torch import softmax
import torch

aft_nn = torch.tensor([3.2, 1.3, 0.2, 0.8], dtype=torch.float)
true_label = torch.tensor([1.0, 0.0, 0.0, 0.0])  
criterion = nn.CrossEntropyLoss()
output = criterion(aft_nn.unsqueeze(0), true_label.unsqueeze(0))
print("Log Loss:", output.item())