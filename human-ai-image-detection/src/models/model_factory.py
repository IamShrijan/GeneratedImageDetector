from .architectures.cnn_models import DeepCNN
from .architectures.hybrid import HybridClassifier

class ModelFactory:
    @staticmethod
    def get_model(model_name: str, **kwargs):
        models = {
            # 'deep_cnn': DeepCNN,
            # 'random_forest'
            'hybrid_classifier': HybridClassifier
        }
        
        if model_name not in models:
            raise ValueError(f"Model {model_name} not found")
            
        return models[model_name](**kwargs)