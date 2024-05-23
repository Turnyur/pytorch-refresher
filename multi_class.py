import torch
import torch.nn as nn
from NeuralNet import NeuralNetwork2





model = NeuralNetwork2(input_size=28*28, hidden_size=5, num_classes=3)
criterion = nn.CrossEntropyLoss() # auto applies softmax

