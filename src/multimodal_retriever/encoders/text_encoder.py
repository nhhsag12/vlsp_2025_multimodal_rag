from typing import Union, List

import torch
from sentence_transformers import SentenceTransformer


class TextEncoder:
    """
    Encodes text queries into a dense vector embedding using sentence transformers
    This is not a nn.Module because sentence-transformers handles the model and processing internally
    """

    def __init__(self, model_name="BAAI/bge-m3"):
        # Initialize without specifying device - will be handled in encode method
        self.model = SentenceTransformer(model_name)

    def encode(self, texts: Union[str, List[str]]) -> torch.Tensor:
        """
        Encodes a single text string into embedding

        :param texts: the input text (str)
        :return: torch.Tensor: A tensor of shape (1, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        # Get the device from CUDA context or use CPU
        device = torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'

        # Move model to current device if not already there
        if self.model.device != device:
            self.model = self.model.to(device)

        # The model directly return a tensor
        embedding = self.model.encode(texts, convert_to_tensor=True)

        return embedding