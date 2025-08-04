import json
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv
from comet_ml import Experiment

from src.reranker.cross_encoder import CrossEncoder
from src.multimodal_retriever.encoders.text_encoder import TextEncoder
from src.multimodal_retriever.encoders.vision_encoder import VisionEncoder
from src.utils.dataset_utils import MultimodalTripletDataset
from src.utils.utils import save_model


def setup(rank, world_size):
    """Initialize the process group for distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()


def collate_fn_contrastive(batch, record_id_to_text_map):
    """
    Collate function for contrastive learning with CrossEncoder.
    """
    base_image_path = "data/VLSP 2025 - MLQA-TSR Data Release/train_data/train_images/train"
    images, query_texts, positive_doc_texts, negative_doc_texts = [], [], [], []

    for item in batch:
        image_path, query_text, pos_id, neg_id = item
        try:
            image = Image.open(os.path.join(base_image_path, image_path)).convert("RGB")
            image = image.resize((224, 224))
            images.append(image)
            query_texts.append(query_text)
            positive_doc_texts.append(record_id_to_text_map[pos_id]["text"])
            negative_doc_texts.append(record_id_to_text_map[neg_id]["text"])
        except Exception as e:
            # Handle cases where an image or record_id might be missing
            print(f"Skipping item due to error: {e}")
            continue

    return images, query_texts, positive_doc_texts, negative_doc_texts

def train_worker(rank, world_size):
    """Training function for each GPU process"""
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # --- Hyperparameters and Configuration ---
    learning_rate = 1e-5
    batch_size = 3072  # Smaller batch size for CrossEncoder
    num_epochs = 5
    margin = 0.5  # Margin for contrastive loss
    num_gpus = world_size

    experiment = None
    if rank == 0:
        load_dotenv()
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="vlsp_multimodal_cross_encoder",
            workspace=os.getenv("COMET_WORKSPACE"),
        )
        experiment.log_parameters({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "margin": margin,
            "num_gpus": num_gpus,
            "loss_function": "MarginRankingLoss"
        })

    # --- Model Initialization ---
    if rank == 0:
        print("Initializing the CrossEncoder...")
    text_encoder = TextEncoder()
    vision_encoder = VisionEncoder()
    model = CrossEncoder(text_model=text_encoder, vision_model=vision_encoder).to(device)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    model.train()
    if rank == 0:
        print("CrossEncoder initialized with DDP.")

    # --- Freeze parts of the model for focused training ---
    for param in model.module.text_model.model.parameters():
        param.requires_grad = False
    for param in model.module.vision_model.vision_model.parameters():
        param.requires_grad = False

    # --- Optimizer, Scheduler, and Loss Function ---
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
    loss_fn = nn.MarginRankingLoss(margin=margin)

    # --- Dataset and DataLoader ---
    if rank == 0:
        print("Loading dataset...")
    # !! IMPORTANT !!: Adjust this path to your file mapping record IDs to text
    record_id_to_document_text_path = "data/record_id_to_document_embedding.json"
    with open(record_id_to_document_text_path, 'r', encoding='utf-8') as f:
        record_id_to_text_map = json.load(f)

    dataset_path = "data/processed_vlsp_2025_multimodal_rag_dataset_120000.json"
    with open(dataset_path, "r") as f:
        raw_dataset = json.load(f)

    dataset = MultimodalTripletDataset(raw_dataset)
    train_sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=True)
    batch_size_per_gpu = batch_size // world_size

    # Use a partial function to pass the text map to the collate function
    from functools import partial
    collate_fn = partial(collate_fn_contrastive, record_id_to_text_map=record_id_to_text_map)

    train_dataloader = DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
    )
    if rank == 0:
        print(f"Dataset loaded. Total triplets: {len(dataset)}. Batch size per GPU: {batch_size_per_gpu}")

    # --- Model Saving Setup ---
    folder_path = None
    if rank == 0:
        folder_path = f"trained_model/cross_encoder_contrastive_{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(folder_path, exist_ok=True)
        print(f"Model will be saved to: {folder_path}")

    # --- Training Loop ---
    model.train()
    for epoch in range(num_epochs):
        train_sampler.set_epoch(epoch)
        total_loss = 0.0

        dataloader_iter = train_dataloader
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")
            dataloader_iter = tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}")

        for batch_idx, batch in enumerate(dataloader_iter):
            images, query_texts, positive_doc_texts, negative_doc_texts = batch

            if not images:  # Skip batch if all items had errors
                continue

            # Forward pass for positive and negative pairs
            positive_scores = model(images, query_texts, positive_doc_texts).squeeze()
            negative_scores = model(images, query_texts, negative_doc_texts).squeeze()

            # Target tensor should be all 1s, indicating positive_scores should be > negative_scores
            target = torch.ones_like(positive_scores)

            # Calculate contrastive loss
            loss = loss_fn(positive_scores, negative_scores, target)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            total_loss += loss.item()
            if rank == 0 and experiment:
                experiment.log_metric("batch_loss", loss.item(), step=batch_idx + epoch * len(train_dataloader))

        # --- End of Epoch ---
        lr_scheduler.step()

        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / (len(train_dataloader) * world_size)

        if rank == 0:
            if experiment:
                experiment.log_metric("epoch_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

            # Save model checkpoint
            model_path = save_model(model.module, folder_path, f"cross_encoder_epoch_{epoch + 1}")
            if experiment:
                experiment.log_model(f"model_epoch_{epoch + 1}", model_path)
            print(f"Model saved to: {model_path}")

        dist.barrier()

    # --- End of Training ---
    if rank == 0:
        print("Training completed!")
        final_model_path = save_model(model.module, folder_path, "cross_encoder_final")
        if experiment:
            experiment.log_model("final_model", final_model_path)
            experiment.end()
        print(f"Final model saved to: {final_model_path}")

    cleanup()


def train():
    """Main training function that spawns multiple processes"""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU training requires CUDA.")

    world_size = max(1, torch.cuda.device_count())
    print(f"Starting multi-GPU training with {world_size} GPUs")

    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    train()