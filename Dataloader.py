import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import numpy as np
import math
from tqdm import tqdm

class WineDataset(Dataset):
    def __init__(self):
    # data loading
        xy = np.loadtxt('./assets/wine.csv', delimiter=",", skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y = torch.from_numpy(xy[:, 0]) # n_samples, 1
        self.n_samples = xy.shape[0]
    def __getitem__(self, index) -> torch.Tensor:
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        return self.n_samples


dataset = WineDataset()
# first_data = dataset[0]
# # print label and features
# features, label = first_data
# print(features.tolist())
BATCH_SIZE = 8
dataloader = DataLoader(dataset=dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

# dataiterator = iter(dataloader)
# data = dataiterator._next_data()
# features, labels = data
# print(features, labels)

# data = dataiterator._next_data()
# features, labels = data
# print(features, labels)


# dummy training loop
num_epochs = 200
progress_bar = tqdm(range(num_epochs), desc="Training Progress", total=num_epochs)
total_samples = len(dataset)

n_iterations = math.ceil(total_samples/BATCH_SIZE)
#print(total_samples, n_iterations)

for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(dataloader):
        # forward backward, update
#        if (i+1) % 10 == 0:
#            print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
        res =1 # Just a Dummy scafold
    progress_bar.set_description(f'Training model...')
    progress_bar.update(1)