import torch

def save_model(params, score, model, model_name):
    save_path = f'trained_models/{model_name}.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'params': params,
        'score': score
    }, save_path)

    print(f"Model saved to {save_path}")