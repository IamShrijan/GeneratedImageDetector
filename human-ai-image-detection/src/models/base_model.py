from abc import ABC, abstractmethod
import torch.nn as nn
import torch

class BaseModel(ABC, nn.Module):
    def __init__(self):
        super().__init__()
        
    @abstractmethod
    def forward(self, x):
        pass
    
    @abstractmethod
    def configure_optimizers(self, config):
        pass
    
    def save_model(self, path):
        torch.save(self.state_dict(), path)
    
    def load_model(self, path):
        self.load_state_dict(torch.load(path))