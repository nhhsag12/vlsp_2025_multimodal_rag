import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image

VISION_MODEL_NAME = "google/vit-base-patch16-224-in21k"

class VisionEncoder(nn.Module):
    """
    Encodes images into a sequence of patch embeddings
    """
    def __init__(self, model_name=VISION_MODEL_NAME):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, device_map="auto")
        self.vision_model = AutoModel.from_pretrained(model_name, device_map="auto")

    def forward(self, image: Image.Image) -> torch.Tensor:
        """
        Processes a PIL image and returns its embeddings.

        :param image: the input image (Image.Image)
        :return: torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) representing the image patch embeddings
        """
        # The image processor converts the PIL image to the format the model expects
        inputs = self.image_processor(images=image, return_tensors="pt").to(self.device)

        # Get the model's output
        with torch.inference_mode():
            outputs = self.vision_model(**inputs)

        # We use the `last_hidden_state` which contains the embeddings for each patch of the image.
        # This will be our K and V for the cross-attention layer
        # Shape: (1, 197, 768) for VIT-Base (196 patches + 1 CLS token)
        return outputs.last_hidden_state


