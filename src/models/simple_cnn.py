"""
CNN model for celebrity classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    """
    A CNN with 4 convolutional layers and 2 fully connected layers.
    """
    
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)  # 3 -> 32 channels
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # 32 -> 64 channels
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # 64 -> 128 channels
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding=1)  # 128 -> 256 channels
        
        # Max pooling layer
        self.pool = nn.MaxPool2d(2, 2)  # Reduces size by half
        
        # Dropout for regularization
        self.dropout = nn.Dropout(0.5)
        
        # Fully connected layers
        # After 4 max pools: 224 -> 112 -> 56 -> 28 -> 14
        # So final size is 256 * 14 * 14
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        # Convolutional layers with ReLU and MaxPool
        x = self.pool(F.relu(self.conv1(x)))  # 224x224 -> 112x112
        x = self.pool(F.relu(self.conv2(x)))  # 112x112 -> 56x56
        x = self.pool(F.relu(self.conv3(x)))  # 56x56 -> 28x28
        x = self.pool(F.relu(self.conv4(x)))  # 28x28 -> 14x14
        
        # Flatten for fully connected layers
        x = x.view(-1, 256 * 14 * 14)
        
        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x


def create_model(num_classes):
    """Create a simple CNN model."""
    return SimpleCNN(num_classes)


def count_parameters(model):
    """Count the number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)