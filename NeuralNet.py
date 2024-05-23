import torch
import torch.nn as nn
import torch.nn.functional as F


# Binary class problem
class NeuralNetwork1(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(NeuralNetwork1, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        #self.softmax = nn.Softmax()
        #self.tanh = nn.Tanh()
        #self.l_relu = nn.LeakyReLU()
        self.linear2 = nn.Linear(hidden_size, 1)


    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        out = self.linear2(out)
        # sigmoid at the end
        y_pred = torch.sigmoid(out)
        return y_pred



# Multiclass problem

class NeuralNetwork2(nn.Module):
    def __init__(self, input_size, hidden_size, num_classes):
        super(NeuralNetwork2, self).__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, num_classes)


    def forward(self, x):
        out = self.linear1(x)
        out = self.relu(out)
        #out = torch.relu(out) # Alternative 2
        #out = F.relu(out)     # Alternative 3
        out = self.linear2(out)
        # no softmax at the end -> nn.CrossEntropy
        return out
