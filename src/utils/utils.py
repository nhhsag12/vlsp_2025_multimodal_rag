import time
import torch
import os
# from src.multimodal_retriever.retriever import Retriever

# Remove hardcoded device - will be set dynamically in distributed training
# DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

def save_model(model, base_path: str, model_name: str) -> str:
    """Save model state dictionary to a file with timestamp.

    Args:
        model: The Retriever model to save
        base_path: Base directory to save the model file in
        model_name: name of the model file
    Returns:
        str: Path to the saved model file
    """
    # timestamp = time.strftime("%Y%m%d-%H%M%S")
    os.makedirs(base_path, exist_ok=True)
    path_file = os.path.join(base_path, f"{model_name}.pt")
    torch.save(model.state_dict(), path_file)
    return path_file


def load_model(model, path_file: str) :
    """Load saved model state from file.

    Args:
        model: The Retriever model instance to load state into
        path_file: Path to the saved model file

    Returns:
        Retriever: Model with loaded state
    """
    model.load_state_dict(torch.load(path_file))
    return model