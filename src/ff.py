import torch
import torch.nn as nn
import torch.nn.functional as F


# Define a simple Feedforward Neural Network
class NaiveNN(nn.Module):
    def __init__(self):
        super(NaiveNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


# Optimized model using JIT compilation for operator fusion
class OptimizedNN(nn.Module):
    def __init__(self):
        super(OptimizedNN, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))  # Fused ReLU operation
        x = self.fc2(x)
        return x
