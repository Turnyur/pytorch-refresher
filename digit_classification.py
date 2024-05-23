import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm


# MNITS
# Dataloader, Transformation
# Multilayer Neural Net, activation function
# Loss and Optimizer
# Training Loop (batch training)
# Model evaluation
# GPU support


class NeuralNet(nn.Module):
    def __init__(self, fan_in, fan_out, num_classes):
        super(NeuralNet, self).__init__()
        self.l1 =nn.Linear(fan_in, fan_out)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(fan_out, num_classes)

    def forward(self, x):
        out = self.l1(x)
        out = self.relu(out)
        out = self.l2(out)

        return out


# device config
device =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
input_size = 784 # 28x28
hiden_size = 100
num_classes =  10
num_epochs = 20
batch_size = 100
learning_rate = 0.01

progress_bar = tqdm(range(num_epochs), desc="Training Progress", total=num_epochs)


# MNIST
train_dataset = torchvision.datasets.MNIST(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)


test_dataset = torchvision.datasets.MNIST(
    root='./data', train=False, transform=transforms.ToTensor())


train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)


examples = iter(train_loader)
samples, labels = examples._next_data()


# print(samples.shape, labels.shape)
# u, _ = torch.unique(labels, return_counts=True)
# print(u)

# for i in range(6):
#     plt.subplot(2,3, i+1)
#     plt.imshow(samples[i][0], cmap='gray')
# plt.show()


model = NeuralNet(input_size, hiden_size, num_classes)

# loss and optimizer
criterion =nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# training loop
n_total_steps = len(train_loader)


for epoch in range(num_epochs):
    for i, (images, labels) in enumerate(train_loader):
       
        # reshpae image
        # 100, 1, 28, 18
        # 100, 784
        images = images.reshape(-1, 28*28).to(device)
        labels = labels.to(device)
        # forward
        pred = model.forward(images)
        # loss
        loss = criterion(pred, labels)

        # empty gradient
        optimizer.zero_grad()
        # gradient
        loss.backward()
        # update step
        optimizer.step()
        #if (i+1) % 100 ==0:
        #    print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss={loss.item():.4f}')
        progress_bar.set_description(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss={loss.item():.4f}')
    progress_bar.set_description(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_total_steps}, loss={loss.item():.4f}')
    progress_bar.update(1)

# test
with torch.no_grad():
   n_correct = 0
   n_samples = 0

#   first_sample = iter(test_loader)
#   images, labels = first_sample._next_data()
#    images = images.reshape(-1, 28*28).to(device)
#    labels = labels.to(device)
#    # predict digit class
#    digit_class = model(images)
#    _, pred_out = torch.max(digit_class, dim=1)
#    print(f'Ground Truth: {label}, Model Prediction: {pred_out}')




# PREDICTIONS tensor([7, 2, 1, 0, 4, 1, 4, 9, 6, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4])
#
# LABELS tensor([7, 2, 1, 0, 4, 1, 4, 9, 5, 9, 0, 6, 9, 0, 1, 5, 9, 7, 3, 4])
#
#
# tensor([ True,  True,  True,  True,  True,  True,  True,  True, False,  True,
#      True,  True,  True,  True,  True,  True,  True,  True,  True,  True])

# print(f'PREDICTIONS {predictions}')
# print()
# print(f'LABELS {labels}')
# print()
# print()
# print(predictions==labels)
# n_correct +=(pred==labels).sum().item()



   for images, labels in test_loader:
       images = images.reshape(-1, 28*28).to(device)
       labels = labels.to(device)
       outputs = model(images)
       # value, index
       _, predictions = torch.max(outputs, dim=1)
       n_samples += labels.shape[0]
       n_correct +=(predictions==labels).sum().item()


accuracy = (n_correct / n_samples) * 100   
print(f'ACCURACY = {accuracy}%')    
