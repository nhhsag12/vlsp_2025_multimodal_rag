import torch
import os
import cv2
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class MultimodalDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        # Load the image
        image_file = item["image_file"]
        base_path = "../../data/VLSP 2025 - MLQA-TSR Data Release/law_db/images"
        image_path = os.path.join(base_path, image_file)
        image = Image.open(image_path).convert("RGB")

        # Process the image
        image = image.resize((224, 224))  # Resize to a fixed size

        query_text = item["query_text"]
        positive_document = item["positive_document"]
        negative_document = item["negative_document"]

        return image, query_text, positive_document, negative_document
    
