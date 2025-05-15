import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(10, 10, kernel_size=3, padding=1)
        # Let's calculate the correct dimensions:
        # Input: 28x28
        # After conv1 (3x3): 26x26
        # After maxpool1: 13x13
        # After conv2 (3x3): 11x11
        # After maxpool2: 5x5
        # Final size: 10 channels * 5 * 5 = 250
        self.fc1 = nn.Linear(10 * 7 * 7, 128)  # Changed from 10 * 10 * 7 to 10 * 5 * 5
        self.fc2 = nn.Linear(128, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2,2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = x.view(-1, 10 * 7 * 7)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())