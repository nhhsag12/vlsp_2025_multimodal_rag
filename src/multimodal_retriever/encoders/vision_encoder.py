import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import Union, List

VISION_MODEL_NAME = "google/vit-base-patch16-224-in21k"


class VisionEncoder(nn.Module):
    """
    Encodes images into a sequence of patch embeddings
    """

    def __init__(self, model_name=VISION_MODEL_NAME):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, device_map=self.device)
        self.vision_model = AutoModel.from_pretrained(model_name, device_map=self.device)

    def forward(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Processes PIL image(s) and returns their embeddings.

        :param images: Single image (Image.Image) or list of images (List[Image.Image])
        :return: torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) representing the image patch embeddings
        """
        # Handle single image case (backward compatibility)
        if isinstance(images, Image.Image):
            images = [images]

        # The image processor can handle batch of images
        inputs = self.image_processor(images=images, return_tensors="pt").to(self.device)

        # Get the model's output
        # with torch.inference_mode():
        outputs = self.vision_model(**inputs)

        # We use the `last_hidden_state` which contains the embeddings for each patch of the image.
        # This will be our K and V for the cross-attention layer
        # Shape: (batch_size, 197, 768) for VIT-Base (196 patches + 1 CLS token per image)
        return outputs.last_hidden_state