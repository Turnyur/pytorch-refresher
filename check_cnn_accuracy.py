import torch
import torchvision
import torchvision.transforms as transforms
from cnn import ConvNet as ConvNet 


# device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'truck', 'ship')
num_classes = len(classes)

input_channel = 3 # RGB
output_channel = 5 # 5 filters


model = ConvNet(input_channel, output_channel, num_classes)  


model_path = './froz_state/cnn_2.pth'
checkpoint = torch.load(model_path)

#print(checkpoint.keys())
model.load_state_dict({
    'conv1.weight': checkpoint['conv1.weight'],
    'conv1.bias': checkpoint['conv1.bias'],
    'conv2.weight': checkpoint['conv2.weight'],
    'conv2.bias': checkpoint['conv2.bias'],
    'fc1.weight': checkpoint['fc1.weight'],
    'fc1.bias': checkpoint['fc1.bias'],
    'fc2.weight': checkpoint['fc2.weight'],
    'fc2.bias': checkpoint['fc2.bias'],
    'fc3.weight': checkpoint['fc3.weight'],
    'fc3.bias': checkpoint['fc3.bias']
})
#model.load_state_dict({key: checkpoint[key] for key in keys})

model.eval()


# dataset
# CIFAR10
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, transform=transforms.ToTensor(), download=True)


test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, transform=transforms.ToTensor())

batch_size = 4
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)



with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(10)]
    n_class_samples = [0 for i in range(10)]
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        score =images.size()
        outputs = model.forward(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()
        
        for i in range(batch_size):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = (n_correct / n_samples) * 100.0
    print(f'Accuracy of the network: {acc} %')
    print("\n")
    for i in range(10):
        acc =  (n_class_correct[i] / n_class_samples[i]) * 100.0
        print(f'Accuracy of {classes[i]}: {acc} %')

