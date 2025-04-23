from .architectures.cnn_models import DeepCNN
from .architectures.hybrid import HybridClassifier
from .architectures.resNet import ModifiedResNet
from .architectures.vit import Vit

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs):
        models = {
            'deep_cnn': DeepCNN,
            'hybrid_classifier': HybridClassifier,
            'resNet': ModifiedResNet,
            'vit': Vit
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
            
        return models[model_name](**kwargs)