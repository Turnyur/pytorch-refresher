
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from tqdm import tqdm

class WineDataset(Dataset):
    def __init__(self, transform=None):
    # data loading
        xy = np.loadtxt('./assets/wine.csv', dtype=np.float64 ,delimiter=",", skiprows=1)
        self.x = xy[:, 1:]
        self.y = xy[:, 0] # n_samples, 1
        self.n_samples = xy.shape[0]
        self.transform = transform
    def __getitem__(self, index) -> torch.Tensor:
        # dataset[0]
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample
    
    def __len__(self):
        return self.n_samples



class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(np.array(targets))

class MulTransform:
    def __init__(self, factor) -> None:
        self.factor = factor

    def __call__(self, sample):
        inputs, target  = sample
        inputs *= self.factor
        return inputs, target



dataset = WineDataset(transform=None)
#dataset = WineDataset(transform=ToTensor())
first_data = dataset[0]
features, labels = first_data
print(features, labels)
print(type(features), type(labels))

composed = torchvision.transforms.Compose([ToTensor(), MulTransform(4)])
dataset =  WineDataset(transform=composed)
first_data = dataset[0]
features, labels = first_data
print(features, labels)
print(type(features), type(labels))


# dataset = torchvision.datasets.MNIST(
#     root='./data', transform=torchvision.transforms.ToTensor())

