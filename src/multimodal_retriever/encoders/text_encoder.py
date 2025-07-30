from typing import Union, List

import torch
from sentence_transformers import SentenceTransformer
from src.utils.utils import DEVICE
class TextEncoder:
    """
    Encodes text queries into a dense vector embedding using sentence transformers
    This is not a nn.Module because sentence-transformers handles the model and processing internally
    """
    def __init__(self, model_name="BAAI/bge-m3"):
        self.model = SentenceTransformer(model_name, device=DEVICE)

    def encode(self, texts: Union[str, List[str]])->torch.Tensor:
        """
        Encodes a single text string into embedding

        :param texts: the input text (str)
        :return: torch.Tensor: A tensor of shape (1, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        # The model directly return a tensor
        embedding = self.model.encode(texts, convert_to_tensor=True)

        # Ensure the embedding is on the correct device
        return embedding





