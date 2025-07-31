import torch
import torch.nn as nn
from transformers import AutoImageProcessor, AutoModel
from PIL import Image
from typing import Union, List
from src.utils.utils import DEVICE

VISION_MODEL_NAME = "google/vit-base-patch16-224-in21k"


class VisionEncoder(nn.Module):
    """
    Enhanced vision encoder with additional processing layers
    """

    def __init__(self, model_name=VISION_MODEL_NAME, hidden_dim=768, dropout=0.1):
        super().__init__()
        self.image_processor = AutoImageProcessor.from_pretrained(model_name, device_map=DEVICE)
        self.vision_model = AutoModel.from_pretrained(model_name, device_map=DEVICE)
        
        # Additional processing layers after the vision transformer
        self.post_processing = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable position embeddings for patch refinement
        self.patch_refinement = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=8,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, images: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
        """
        Processes PIL image(s) and returns their enhanced embeddings.

        :param images: Single image (Image.Image) or list of images (List[Image.Image])
        :return: torch.Tensor: A tensor of shape (batch_size, sequence_length, embedding_dim) representing the enhanced image patch embeddings
        """
        # Handle single image case (backward compatibility)
        if isinstance(images, Image.Image):
            images = [images]

        # The image processor can handle batch of images
        inputs = self.image_processor(images=images, return_tensors="pt").to(DEVICE)

        # Get the model's output
        outputs = self.vision_model(**inputs)

        # Get the last hidden state
        # Shape: (batch_size, 197, 768) for VIT-Base (196 patches + 1 CLS token per image)
        hidden_states = outputs.last_hidden_state
        
        # Apply post-processing layers
        processed_states = self.post_processing(hidden_states)
        
        # Apply patch refinement transformer layer
        refined_states = self.patch_refinement(processed_states)
        
        return refined_states