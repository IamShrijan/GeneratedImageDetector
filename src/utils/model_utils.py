import torch

def save_model(params, score, model, model_name: str = 'best_model'):
    save_path = f'trained_models/{model_name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
        'score': score
    }, save_path)

    print(f"Model saved to {save_path}")


def load_model(model_class, model_path: str, device= 'cuda' if torch.cuda.is_available() else 'cpu'):
    checkpoint = torch.load(model_path, map_location=device)

    # Initialize model
    model = model_class()
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    print(f"Model loaded from {model_path}, with score: {checkpoint['score']}")
    
    return model, checkpoint['params'], checkpoint['score']