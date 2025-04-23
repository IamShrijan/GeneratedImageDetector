import torch.nn as nn
import torch

# Inspired from https://futurescienceleaders.com/blog/2024/06/differentiating-ai-vs-human-art-a-convolutional-neural-network-approach/#:~:text=It%20uses%20Convolutional%20Neural%20Networks,and%20decreasing%20loss%20over%20iterations.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DeepCNN(nn.Module):
    def __init__(self, in_channels_list, out_channels_list, kernel_size_list, dropout_rate=0.5):
        
        super().__init__()
        
        # Make sure the lengths of lists are the same
        assert len(in_channels_list) == len(out_channels_list) == len(kernel_size_list), \
            "The lengths of in_channels, out_channels, and kernel_size lists must be the same."
        
        # Initialize convolutional layers
        conv_layers = []
        for i in range(len(in_channels_list)):
            conv_layers.append(nn.Conv2d(
                in_channels=in_channels_list[i], 
                out_channels=out_channels_list[i], 
                kernel_size=kernel_size_list[i], 
                padding=1 if kernel_size_list[i] > 1 else 0
            ))

            if i <= len(in_channels_list) - 1:
                # Add ReLU activation function and Max Pooling
                conv_layers.append(nn.ReLU())
                conv_layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        
        self.conv_block = nn.Sequential(*conv_layers)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Calculate the size after all convolutions and pooling
        # Assuming input size is 128x128, calculate the feature map size after all conv layers and pooling
        self.fc1 = nn.Linear(out_channels_list[-1] * 8 * 8, 128) 
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 1) 
        
    def forward(self, x):
        # Pass through convolutional layers
        x: torch.Tensor = self.conv_block(x)
        
        x = x.view(x.size(0), -1)
        
        x = F.relu(self.fc1(x))  
        x = self.dropout(x)      
        
        x = F.relu(self.fc2(x)) 
        x = self.dropout(x)      
        
        x = torch.sigmoid(self.fc3(x))  # Output Layer (sigmoid for binary classification)

        return x
