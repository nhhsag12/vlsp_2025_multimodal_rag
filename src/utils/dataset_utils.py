from torch.utils.data import Dataset

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

