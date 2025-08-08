import os

from torch.utils.data import Dataset
from PIL import Image

class MultimodalTripletDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]

        image_path = item["image_file"]
        query_text = item["query_text"]
        positive_record_id = item["positive_record_id"]
        negative_record_id = item["negative_record_id"]

        return image_path, query_text, positive_record_id, negative_record_id

class CrossEncoderTripleDataset(Dataset):
    def __init__(self, dataset, record_id_to_document, preprocessing):
        self.dataset = dataset
        self.preprocessing = preprocessing
        self.record_id_to_document = record_id_to_document

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        query_texts = item["query_text"]

        # Open and pre-process image
        image_file = item["image_file"]
        base_path = "data/VLSP 2025 - MLQA-TSR Data Release/train_data/train_images/train"
        image_path = os.path.join(base_path, image_file)
        image = Image.open(image_path)
        processed_image = self.preprocessing(image)

        positive_record_id = item["positive_record_id"]
        negative_record_id = item["negative_record_id"]
        positive_text = self.record_id_to_document[positive_record_id]["text"]
        negative_text = self.record_id_to_document[negative_record_id]["text"]

        return query_texts, processed_image, positive_text, negative_text

