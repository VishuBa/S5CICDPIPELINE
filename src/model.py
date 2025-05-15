import torch
import torch.nn as nn

class MNISTModel(nn.Module):
    def __init__(self):
        super(MNISTModel, self).__init__()
        # Minimal architecture
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, padding=1)  # Reduced to 4 filters
        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, padding=1)  # Reduced to 8 filters
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)
        # After two maxpool layers: 28x28 -> 14x14 -> 7x7
        self.fc1 = nn.Linear(8 * 7 * 7, 16)  # Reduced to 16 neurons
        self.fc2 = nn.Linear(16, 10)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout1(x)
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())