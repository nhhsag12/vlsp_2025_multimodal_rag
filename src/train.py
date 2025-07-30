import json
import os
import time
import torch
import torch.nn as nn
from tqdm import tqdm
from PIL import Image

from src.multimodal_retriever.retriever import Retriever
from src.utils.dataset_utils import MultimodalTripletDataset
from src.utils.utils import save_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def collate_fn(batch):
    # Load the necessary materials
    base_image_path = "data/VLSP 2025 - MLQA-TSR Data Release/train_data/train_images/train"
    record_id_to_document_embedding_path = "data/record_id_to_document_embedding.json"
    with open(record_id_to_document_embedding_path, "r") as f:
        record_id_to_document_embedding = json.load(f)

    # Process the data with the material
    images = []
    query_texts = []
    positive_record_embeddings = []
    negative_record_embeddings = []
    for item in batch:
        image = Image.open(os.path.join(base_image_path, item[0])).convert("RGB")
        image = image.resize((224, 224))
        images.append(image)

        query_text = item[1]
        query_texts.append(query_text)

        positive_record_id = item[2]
        try:
            # Remove requires_grad=True since these are target embeddings, not trainable parameters
            positive_record_embedding = torch.tensor(record_id_to_document_embedding[positive_record_id]["embedding"])
            positive_record_embeddings.append(positive_record_embedding)
        except:
            print(f"Record ID {positive_record_id} not found in the document embeddings.")
            print(item)
            exit()
        negative_record_id = item[3]
        # Remove requires_grad=True since these are target embeddings, not trainable parameters
        negative_record_embedding = torch.tensor(record_id_to_document_embedding[negative_record_id]["embedding"])
        negative_record_embeddings.append(negative_record_embedding)
        
    # Stack tensors
    positive_record_embeddings = torch.stack(positive_record_embeddings)
    negative_record_embeddings = torch.stack(negative_record_embeddings)
    # positive_record_embeddings.requires_grad = True
    # negative_record_embeddings.requires_grad = True
    return images, query_texts, positive_record_embeddings, negative_record_embeddings


def train():
    # Initialize the retriever
    print("Initializing the retriever...")
    retriever = Retriever(projection_dim=1024).to(DEVICE)
    retriever.train()
    print("Retriever initialized.")

    # Freeze the pre-trained encoders to save memory and prevent catastrophic forgetting (Optional)
    for param in retriever.vision_encoder.parameters():
        param.requires_grad = False
    for param in retriever.text_encoder.model.parameters():
        param.requires_grad = False

    # Only train the parameter of the fusion and output projection layers
    optimizer = torch.optim.Adam(
        list(retriever.fusion_module.parameters()) + list(retriever.output_projection.parameters()),
        lr=1e-5
    )

    # Define the loss function
    loss_fn = nn.TripletMarginLoss(margin=1.0)

    # -- Setup the dataset --
    print("Loading dataset...")
    dataset_path = "data/processed_vlsp_2025_multimodal_rag_dataset.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    dataset = MultimodalTripletDataset(dataset)

    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=32,
        shuffle=True,
        collate_fn=collate_fn,
    )
    print("Dataset loaded.")

    # -- Setup the saving place for models --
    # Create the folder save the trained model of the session
    folder_path = f"trained_model/trained_model_{time.strftime('%Y%m%d-%H%M%S')}"
    os.makedirs(folder_path, exist_ok=True)
    epoch_folder_path = os.path.join(folder_path, "epochs")
    os.makedirs(epoch_folder_path, exist_ok=True)
    print(f"Model will be saved to: {folder_path}")

    # -- Training Loop --
    num_epochs = 5
    retriever.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        print(f"Epoch {epoch + 1}/{num_epochs}")
        
        progress_bar = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Unpack data from the dataloader
            images, query_texts, positive_record_embeddings, negative_record_embeddings = batch

            # Move data to the correct device - no need to clone since they're already proper tensors
            positive_record_embeddings = positive_record_embeddings.to(DEVICE)
            negative_record_embeddings = negative_record_embeddings.to(DEVICE)

            # Forward pass through the retriever
            multimodal_embeddings = retriever(images, query_texts)

            # Calculate triplet loss
            # anchor: multimodal_embeddings, positive: positive_record_embeddings, negative: negative_record_embeddings
            loss = loss_fn(multimodal_embeddings, positive_record_embeddings, negative_record_embeddings)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}', 'Avg Loss': f'{total_loss/(batch_idx+1):.4f}'})

        # Print epoch statistics
        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

        # Save model checkpoint every epoch
        model_path = save_model(retriever, epoch_folder_path)
        print(f"Model saved to: {model_path}")

    print("Training completed!")
    
    # Save final model
    final_model_path = save_model(retriever, folder_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    train()