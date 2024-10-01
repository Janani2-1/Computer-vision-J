# src/model.py

import torch
import torch.nn as nn

class NestedUNet(nn.Module):
    def __init__(self):
        super(NestedUNet, self).__init__()
        # Example: simple layers to demonstrate a model with parameters
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)  # Input channels = 3 (e.g., RGB)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        x = nn.ReLU()(x)
        x = self.conv2(x)
        return x

class AttentionUNet(nn.Module):
    def __init__(self):
        super(AttentionUNet, self).__init__()
        # Define layers here as needed

    def forward(self, x):
        # Define the forward pass
        return x
