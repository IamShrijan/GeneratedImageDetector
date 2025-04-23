import torch 
import torch.nn as nn
import torchvision.models as models

class Vit(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()
        self.vit =  models.vit_b_16(pretrained= True)
        self.vit.heads.head = nn.Linear(self.vit.heads.head.in_features, num_classes)
    def forward(self, x):
        x = self.vit(x)
        x = torch.sigmoid(x)
        return x