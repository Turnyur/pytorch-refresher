import torch
import torch.nn as nn
from NeuralNet import NeuralNetwork1






model = NeuralNetwork1(input_size=28*28, hidden_size=5)
criterion = nn.BCELoss() 

