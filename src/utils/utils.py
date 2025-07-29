import time
import torch
from src.multimodal_retriever.retriever import Retriever


def save_model(model: Retriever) -> str:
    """Save model state dictionary to a file with timestamp.

    Args:
        model: The Retriever model to save

    Returns:
        str: Path to the saved model file
    """
    path_file = f"./trained_models/model_{time.strftime('%Y%m%d_%H%M%S')}.pt"
    torch.save(model.state_dict(), path_file)
    return path_file


def load_model(model: Retriever, path_file: str) -> Retriever:
    """Load saved model state from file.

    Args:
        model: The Retriever model instance to load state into
        path_file: Path to the saved model file

    Returns:
        Retriever: Model with loaded state
    """
    model.load_state_dict(torch.load(path_file))
    return model
