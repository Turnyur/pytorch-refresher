import torch
import torch.nn as nn
import numpy as np


def softmax(x):
    return np.exp(x)/np.sum(np.exp(x), axis=0)

a = [2.0, 1.0, 0.1]
x = np.array(a)
outputs = softmax(x)

print('softmax numpy: ', outputs)
x= torch.tensor(a)
outputs = torch.softmax(x, dim=0)

print('softmax numpy: ', outputs)