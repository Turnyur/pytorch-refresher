import torch
import torch.nn as nn
import numpy as np


loss = nn.CrossEntropyLoss()

#Y = torch.tensor([0]) # index of the ground truth label
# nsamples x nclass = 1x3
#Y_pred_good = torch.tensor([[2.0, 1.0, 0.1]]) #1x3
#Y_pred_bad = torch.tensor([[0.5, 2.0, 0.3]]) #1x3


Y = torch.tensor([2, 0, 1]) # three possible classes i.e their indexes

Y_pred_good = torch.tensor([[0.1, 1.0, 2.1], [2.0, 1.0, 0.1], [0.1, 3.0, 0.1]]) #3x3
Y_pred_bad = torch.tensor([[2.1, 1.0, 0.1], [0.1, 1.0, 2.1], [0.1, 3.0, 0.1]]) #3x3

l1 = loss(Y_pred_good, Y)
l2 = loss(Y_pred_bad, Y)

print(l1.item())
print(l2.item())

_, predictions1 = torch.max(Y_pred_good, 1) # return index
_, predictions2 = torch.max(Y_pred_bad, 1) # return index

print(predictions1)
print(predictions2)