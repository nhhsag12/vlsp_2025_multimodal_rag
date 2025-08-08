from typing import Union, List
from PIL import Image
import torch
from torch import nn
from FlagEmbedding.research.visual_bge.visual_bge.modeling import Visualized_BGE

class CrossEncoder(nn.Module):
    def __init__(self, pretrained_model_path:str="pretrained_model/Visualized_m3.pth"):
        super().__init__()
        self.model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight=pretrained_model_path)
        self.tokenizer = self.model.tokenizer
        self.preprocess = self.model.preprocess_val
        self.device = next(self.parameters()).device

    def encode_text(self, texts: Union[str, List[str]])->torch.Tensor:
        texts = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        return self.model.encode_text(texts) # (batch_size, embed_dim)

    def encode_image(self, images: torch.Tensor)->torch.Tensor:
        device = next(self.parameters()).device
        if images.dim() < 3: # Only 1 image without batch dim
            return self.model.encode_image(images.to(device).unsqueeze(0)) # (1, embed_dim)

        return self.model.encode_image(images.to(device))

    def encode_mm(self, texts: Union[str, List[str]], processed_images: torch.Tensor)->torch.Tensor:
        texts = self.tokenizer(texts, return_tensors="pt", padding=True).to(self.device)
        return self.model.encode_mm(processed_images, texts) # (batch_size,...)

    def forward(self, texts: List[str], images: torch.Tensor=None)->torch.Tensor:
        if images is None:
            return self.encode_text(texts)
        return self.encode_mm(texts, images)

