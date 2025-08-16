import torch
import torch.nn as nn

from src.multimodal_retriever.retriever_v2 import RetrieverV2

retriever = RetrieverV2()
base_path = "data/VLSP 2025 - MLQA-TSR Data Release/train_data/train_images/train"
preprocess = retriever.preprocess
with open()