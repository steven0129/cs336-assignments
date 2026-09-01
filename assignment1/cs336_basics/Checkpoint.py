import torch

def save_checkpoint(model, optimizer, iteration, path):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration
    }, path)

def load_checkpoint(path):
    if not torch.cuda.is_available():
        return torch.load(path, map_location='cpu')

    return torch.load(path)