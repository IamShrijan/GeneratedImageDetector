import torch

import torch.nn as nn
import torchvision.models as models

class ModifiedResNet(nn.Module):
    def __init__(self, num_classes=2):
        super(ModifiedResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, num_classes)

    def forward(self, x):
        x = self.resnet(x)
        x = torch.sigmoid(x)
        return x