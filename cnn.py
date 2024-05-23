import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm




class ConvNet(nn.Module):
    def __init__(self, input_channels, output_channels, num_classes, filter_size=5):
        super(ConvNet, self).__init__()
        # [3, 32, 32] -> [6, 28, 28]
        self.conv1 = nn.Conv2d(
            in_channels=input_channels, out_channels=output_channels, kernel_size=filter_size) 
        # [6, 28, 28] -> [6, 14, 14]
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        # [6, 14, 14] -> [16, 10, 10]
        self.conv2 = nn.Conv2d(
            in_channels=output_channels, out_channels=16, kernel_size=filter_size) 
        # after second maxpool
        # [6, 10, 10] -> [16, 5, 5]    the last feature image width and size is 5x5    
        self.fc1 =nn.Linear(16*5*5, 120)
        self.fc2 =nn.Linear(120, 84)
        self.fc3 =nn.Linear(84, num_classes)




    def forward(self, x):
        # first conv Layer
        out = self.conv1(x)
        out = F.relu(out)
        out = self.pool(out)
        # second conv layer
        out = self.conv2(out)
        out = F.relu(out)
        out = self.pool(out)

        # flatten image
        out = out.view(-1, 16*5*5)
        score = out.size()
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)


        return out




if __name__=="__main__":
    # device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # hyper parameters
    input_channel = 3 # RGB
    output_channel = 5 # 5 filters
    num_epochs = 20
    batch_size = 4
    learning_rate = 0.001

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'truck', 'ship')
    num_classes = len(classes)

    progress_bar = tqdm(range(num_epochs), desc="Training Progress", total=num_epochs)


    # dataset has PILImage images of range [0,1].
    # We transform them to Tensors of normalized range [-1, 1]

    transform = transforms.Compose(
        [transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )


    # dataset
    # CIFAR10
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, transform=transforms.ToTensor(), download=True)


    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, transform=transforms.ToTensor())


    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)




    model = ConvNet(input_channel, output_channel, num_classes).to(device)

    # loss and optimizer
    criterion =nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    # training loop
    n_total_steps = len(train_loader)

    #print(f'N Training Samples: {n_total_steps}')


    first_sample = iter(test_loader)
    images, labels = first_sample._next_data()
    images = images.to(device)
    labels = labels.to(device)
    # predict digit class
    #digit_class = model(images)
    #_, pred_out = torch.max(digit_class, dim=1)
    #print(f'IMAGES Shape: {images.shape}, LABELS Shape: {labels[0]}')


    # for i in range(4):
    #     plt.subplot(2,3, i+1)
    #     img_np = images.numpy()[i].transpose((1, 2, 0))
    #     plt.imshow(img_np)
    #     plt.axis('off')
    # plt.show()


    for epoch in range(num_epochs):
        for i, (images, labels) in enumerate(train_loader):
        
            # reshape image
            # original shape: [4, 3, 32, 32] = 4,3,1024
            # 
            # input_layer: 3 input channels, 6 output channels, 5 kernel_size
            images = images.to(device)
            labels = labels.to(device)
            # forward pass
            outputs = model.forward(images)
            # loss
            loss = criterion(outputs, labels)

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


    print('Finishing Training...')
    PATH = './froz_state/cnn_1.pth'
    torch.save(model.state_dict(), PATH)


