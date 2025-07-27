import torch
from sentence_transformers import SentenceTransformer
class TextEncoder:
    """
    Encodes text queries into a dense vector embedding using sentence transformers
    This is not a nn.Module because sentence-transformers handles the model and processing internally
    """
    def __init__(self, model_name="BAAI/bge-m3"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SentenceTransformer(model_name, device=self.device)

    def encode(self, text: str)->torch.Tensor:
        """
        Encodes a single text string into embedding

        :param text: the input text (str)
        :return: torch.Tensor: A tensor of shape (1, embedding_dim)
        """
        # The model directly return a tensor
        embedding = self.model.encode(text, convert_to_tensor=True)

        # Ensure the embedding is on the correct device
        return embedding.unsqueeze(0).to(self.device)





