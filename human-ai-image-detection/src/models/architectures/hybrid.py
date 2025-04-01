import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import numpy as np

def compute_handcrafted_features(image_tensor):
    """
    Compute handcrafted features from an RGB image:
      - Edge strength per channel (using Sobel)
      - Brightness per channel (mean intensity)
      - Contrast per channel (std deviation)
    Returns: Flattened feature vector of size 9 (3 channels × 3 features).
    """
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy() * 255
    image_np = image_np.astype(np.uint8)

    # Compute brightness & contrast per channel
    brightness = [image_np[:, :, c].mean() for c in range(3)]
    contrast = [image_np[:, :, c].std() for c in range(3)]

    # Compute edge strength per channel
    edge_strength = []
    for channel in range(3):
        sobelx = cv2.Sobel(image_np[:, :, channel], cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(image_np[:, :, channel], cv2.CV_64F, 0, 1, ksize=3)
        edge_strength.append(np.sqrt(sobelx**2 + sobely**2).mean())

    # Combine all features into a 9D vector
    features = np.array(edge_strength + brightness + contrast, dtype=np.float32)
    return torch.tensor(features)



class SpatialAttention(nn.Module):
    """
    Spatial attention module to focus on informative regions in the image.
    """
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=7, padding=3)
    
    def forward(self, x):
        attention_map = torch.sigmoid(self.conv(x))
        return x * attention_map



class CNNBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.attention = SpatialAttention(in_channels=64)
        self.fc = nn.Linear(64 * 32 * 32, 50)  # Adjust for input size
    
    def forward(self, x):
        x = self.conv_layers(x)
        x = self.attention(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
    

class HandcraftedBranch(nn.Module):
    def __init__(self, output_dim=50):
        super().__init__()
        self.fc1 = nn.Linear(9, 32)
        self.fc2 = nn.Linear(32, 64)
        self.fc3 = nn.Linear(64, output_dim)
    
    def forward(self, features):
        x = F.relu(self.fc1(features))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return x


class HybridClassifier(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.cnn_branch = CNNBranch()
        self.handcrafted_branch = HandcraftedBranch()
        self.fc_fusion = nn.Linear(50 + 50, 64)
        self.classifier = nn.Linear(64, num_classes)
    
    def forward(self, images):
        cnn_features = self.cnn_branch(images)

        # Compute handcrafted features for each image in the batch
        batch_features = [compute_handcrafted_features(img) for img in images]
        handcrafted_features = torch.stack(batch_features).to(images.device)
        handcrafted_rep = self.handcrafted_branch(handcrafted_features)
        
        # Fuse CNN and handcrafted features
        fused = torch.cat([cnn_features, handcrafted_rep], dim=1)

        fused = F.relu(self.fc_fusion(fused))
        logits = self.classifier(fused)
        return torch.sigmoid(logits)
