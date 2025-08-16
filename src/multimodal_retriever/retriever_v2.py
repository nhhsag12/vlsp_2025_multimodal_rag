from typing import Union, List
from PIL import Image
import torch
from torch import nn
from transformers.masking_utils import padding_mask_function

from FlagEmbedding.research.visual_bge.visual_bge.modeling import Visualized_BGE

class RetrieverV2(nn.Module):
    def __init__(self, pretrained_model_path:str="src/multimodal_retriever/pretrained_model/Visualized_m3.pth", proj_dim:int = 1024, dropout:float = 0.1):
        super().__init__()
        self.model = Visualized_BGE(model_name_bge = "BAAI/bge-m3", model_weight=pretrained_model_path)
        # Add the following line to convert the BGE model to half precision.
        # self.model.half()
        self.tokenizer = self.model.tokenizer
        self.preprocess_train = self.model.preprocess_train
        self.preprocess_val = self.model.preprocess_val
        # self.device = next(self.parameters()).device
        # self.device = self.model.device

        self.proj_dim = proj_dim
        self.dropout = dropout

        # self.encoder_layer = nn.TransformerEncoderLayer(
        #     d_model=self.proj_dim,
        #     nhead=8,
        #     dim_feedforward=self.proj_dim*4,
        #     dropout=self.dropout,
        #     batch_first=True,
        # )
        # self.fusion_module = nn.TransformerEncoder(
        #     self.encoder_layer,
        #     num_layers=6,
        #     norm=nn.LayerNorm(self.proj_dim)
        # )
        self.fusion_module = nn.Sequential(
            nn.Linear(self.proj_dim, self.proj_dim * 2),
            nn.LayerNorm(self.proj_dim * 2),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_dim * 2, self.proj_dim),
            nn.LayerNorm(self.proj_dim),
            nn.Dropout(self.dropout),
            nn.Linear(self.proj_dim, self.proj_dim)
        )
        # # Add the following line to convert the fusion module to half precision.
        # self.fusion_module.half()


    def encode_text(self, texts: Union[str, List[str]])->torch.Tensor:
        device = next(self.parameters()).device
        texts = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
        return self.model.encode_text(texts) # (batch_size, embed_dim)

    def encode_image(self, images: torch.Tensor)->torch.Tensor:
        device = next(self.parameters()).device
        if images.dim() < 3:
            return self.model.encode_image(images.to(device).unsqueeze(0))
        return self.model.encode_image(images.to(device))

    def encode_mm(self, texts: Union[str, List[str]], processed_images: torch.Tensor)->torch.Tensor:
        device = next(self.parameters()).device
        texts = self.tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=8192).to(device)
        return self.model.encode_mm(processed_images, texts)

    def forward(self, texts: List[str], images: torch.Tensor=None)->torch.Tensor:
        if images is None:
            return self.encode_text(texts)
        return self.fusion_module(self.encode_mm(texts, images))