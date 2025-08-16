from comet_ml import Experiment
import json
import os
import time
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from PIL import Image
from dotenv import load_dotenv

from src.multimodal_retriever.retriever_v2 import RetrieverV2
from src.utils.dataset_utils import MultimodalTripletDataset
from src.utils.utils import save_model
from evaluation import evaluate_model, load_evaluation_data, print_evaluation_summary


def setup(rank, world_size):
    """Initialize the process group for distributed training"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()


class CollateClass:
    def __init__(self, preprocessor):
        self.preprocessor = preprocessor
        self.base_image_path = "data/VLSP 2025 - MLQA-TSR Data Release/train_data/train_images/train"
        record_id_to_document_embedding_path = "data/record_id_to_document_embedding.json"
        with open(record_id_to_document_embedding_path, "r") as f:
            self.record_id_to_document_embedding = json.load(f)

    def __call__(self, batch):
        # Process the data with the material
        preprocessed_images = []
        query_texts = []
        positive_record_embeddings = []
        negative_record_embeddings = []
        for item in batch:
            image = Image.open(os.path.join(self.base_image_path, item[0]))
            preprocessed_image = self.preprocessor(image)
            preprocessed_images.append(preprocessed_image)

            query_text = item[1]
            query_texts.append(query_text)

            positive_record_id = item[2]
            try:
                positive_record_embedding = torch.tensor(self.record_id_to_document_embedding[positive_record_id]["embedding"])
                positive_record_embeddings.append(positive_record_embedding)
            except:
                print(f"Record ID {positive_record_id} not found in the document embeddings.")
                print(item)
                exit()
            negative_record_id = item[3]
            negative_record_embedding = torch.tensor(self.record_id_to_document_embedding[negative_record_id]["embedding"])
            negative_record_embeddings.append(negative_record_embedding)

        # Stack tensors
        positive_record_embeddings = torch.stack(positive_record_embeddings)
        negative_record_embeddings = torch.stack(negative_record_embeddings)
        preprocessed_images = torch.stack(preprocessed_images)
        return preprocessed_images, query_texts, positive_record_embeddings, negative_record_embeddings



def train_worker(rank, world_size):
    """Training function for each GPU process"""
    # Setup distributed training
    setup(rank, world_size)
    device = torch.device(f'cuda:{rank}')

    # Initialize Comet ML experiment tracking only on rank 0
    experiment = None
    learning_rate = 1e-4
    batch_size = 256
    num_epochs = 3
    margin = 0.3
    projection_dim = 1024
    num_gpus = world_size
    dropout=0.1
    if rank == 0:
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="vlsp_multimodal_retriever_v2",
            workspace=os.getenv("COMET_WORKSPACE"),
        )
        # num_epochs = 3
        experiment.log_parameters({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "margin": margin,
            "num_gpus": num_gpus,
            "dropout": dropout
        })

    # Load evaluation data (only on rank 0)
    public_test = None
    document_embeddings = None
    image_base_path = None
    if rank == 0:
        print("Loading evaluation data...")
        public_test, document_embeddings, image_base_path = load_evaluation_data()
        print("Evaluation data loaded.")

    # Initialize the retriever
    if rank == 0:
        print("Initializing the retriever...")
    retriever = RetrieverV2(
        pretrained_model_path="src/multimodal_retriever/pretrained_model/Visualized_m3.pth",
        proj_dim=projection_dim,
        dropout=dropout
    ).to(device)

    # Wrap model with DDP
    retriever = DDP(retriever, device_ids=[rank], output_device=rank, find_unused_parameters=True)
    retriever.train()
    # Get the preprocessor
    preprocessor = retriever.module.preprocess_train

    # Freeze the pre-trained encoders
    for param in retriever.module.model.parameters():
        param.requires_grad = False

    # Only train the parameters of the fusion and output projection layers
    optimizer = torch.optim.Adam(
        list(retriever.module.fusion_module.parameters()),
        lr=learning_rate
    )
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

    # Define the loss function
    loss_fn = nn.TripletMarginLoss(margin=margin)

    # Setup the dataset
    dataset_path = "data/processed_vlsp_2025_multimodal_rag_dataset_120000.json"
    with open(dataset_path, "r") as f:
        dataset = json.load(f)
    dataset = MultimodalTripletDataset(dataset)

    # Create distributed sampler
    train_sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )

    # Create an instance of the collate class
    custom_collate_fn = CollateClass(preprocessor)

    # Adjust batch size for distributed training
    batch_size_per_gpu = batch_size // world_size
    train_dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size_per_gpu,
        sampler=train_sampler,
        collate_fn=custom_collate_fn,  # Pass the instance of the class
        num_workers=4,
        pin_memory=True,
    )


    if rank == 0:
        print(f"Dataset loaded. Batch size per GPU: {batch_size_per_gpu}")

    # Setup the saving place for models (only on rank 0)
    folder_path = None
    epoch_folder_path = None
    if rank == 0:
        folder_path = f"trained_model/trained_model_{time.strftime('%Y%m%d-%H%M%S')}"
        os.makedirs(folder_path, exist_ok=True)
        epoch_folder_path = os.path.join(folder_path, "epochs")
        os.makedirs(epoch_folder_path, exist_ok=True)
        # Create evaluation results folder
        eval_folder_path = os.path.join(folder_path, "evaluations")
        os.makedirs(eval_folder_path, exist_ok=True)
        print(f"Model will be saved to: {folder_path}")

    # Training Loop
    # num_epochs = 3
    retriever.train()
    for epoch in range(num_epochs):
        # Set epoch for distributed sampler
        train_sampler.set_epoch(epoch)

        total_loss = 0.0
        if rank == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}")

        # Use progress bar only on rank 0
        dataloader_iter = enumerate(train_dataloader)
        if rank == 0:
            dataloader_iter = enumerate(tqdm(train_dataloader, desc=f"Training Epoch {epoch + 1}"))

        for batch_idx, batch in dataloader_iter:
            # Unpack data from the dataloader
            images, query_texts, positive_record_embeddings, negative_record_embeddings = batch

            # Move data to the correct device
            positive_record_embeddings = positive_record_embeddings.to(device)
            negative_record_embeddings = negative_record_embeddings.to(device)

            # Forward pass through the retriever
            # print(query_texts)
            # print(type(query_texts))
            multimodal_embeddings = retriever(query_texts, images)

            # Calculate triplet loss
            loss = loss_fn(multimodal_embeddings, positive_record_embeddings, negative_record_embeddings)

            # Log batch metrics only on rank 0
            if rank == 0 and experiment:
                experiment.log_metric("batch_loss", loss.item(), step=batch_idx + epoch * len(train_dataloader))

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Accumulate loss
            total_loss += loss.item()

        # Update the learning rate
        lr_scheduler.step()

        # Calculate average loss across all GPUs
        total_loss_tensor = torch.tensor(total_loss, device=device)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        avg_loss = total_loss_tensor.item() / (len(train_dataloader) * world_size)

        # Print epoch statistics and save model (only on rank 0)
        if rank == 0:
            if experiment:
                experiment.log_metric("epoch_loss", avg_loss, step=epoch)
            print(f"Epoch {epoch + 1}/{num_epochs} completed. Average Loss: {avg_loss:.4f}")

            # Save model checkpoint every epoch
            model_name = f"model_epoch_{epoch + 1}.pt"
            model_path = save_model(retriever.module, epoch_folder_path, model_name)
            if experiment:
                experiment.log_model(f"model_epoch_{epoch}", model_path)
            print(f"Model saved to: {model_path}")

            # Evaluate the model after each epoch
            print(f"Starting evaluation for epoch {epoch + 1}...")

            # Create a temporary model for evaluation (without DDP wrapper)
            eval_model = retriever.module
            eval_results = evaluate_model(
                model=eval_model,
                test_data=public_test,
                document_embeddings=document_embeddings,
                image_base_path=image_base_path,
                k=10,
                similarity_threshold=0.8,
                device=device
            )

            # Print evaluation summary
            print_evaluation_summary(eval_results, epoch + 1)

            # Log evaluation metrics to Comet ML
            if experiment:
                eval_metrics = eval_results["overall_metrics"]
                experiment.log_metric("eval_precision", eval_metrics["avg_precision"], step=epoch)
                experiment.log_metric("eval_recall", eval_metrics["avg_recall"], step=epoch)
                experiment.log_metric("eval_f2_score", eval_metrics["avg_f2_score"], step=epoch)
                experiment.log_metric("eval_coverage_rate", eval_metrics["coverage_rate"], step=epoch)
                experiment.log_metric("eval_avg_retrieved_per_case", eval_metrics["avg_retrieved_per_case"], step=epoch)

            # Save evaluation results
            eval_results_path = os.path.join(eval_folder_path, f"evaluation_epoch_{epoch + 1}.json")
            with open(eval_results_path, "w", encoding="utf-8") as f:
                json.dump(eval_results, f, indent=2, ensure_ascii=False)
            print(f"Evaluation results saved to: {eval_results_path}")

        # Synchronize all processes
        dist.barrier()

    if rank == 0:
        print("Training completed!")

        # Save final model
        model_name = f"final_model.pt"
        final_model_path = save_model(retriever.module, folder_path, model_name)
        if experiment:
            experiment.log_model("final_model", final_model_path)

        # Final evaluation
        print("Starting final evaluation...")
        eval_model = retriever.module
        final_eval_results = evaluate_model(
            model=eval_model,
            test_data=public_test,
            document_embeddings=document_embeddings,
            image_base_path=image_base_path,
            k=5,
            similarity_threshold=0.7,
            device=device
        )

        print_evaluation_summary(final_eval_results)

        # Save final evaluation results
        final_eval_path = os.path.join(folder_path, "final_evaluation.json")
        with open(final_eval_path, "w", encoding="utf-8") as f:
            json.dump(final_eval_results, f, indent=2, ensure_ascii=False)

        if experiment:
            final_metrics = final_eval_results["overall_metrics"]
            experiment.log_metric("final_precision", final_metrics["avg_precision"])
            experiment.log_metric("final_recall", final_metrics["avg_recall"])
            experiment.log_metric("final_f2_score", final_metrics["avg_f2_score"])
            experiment.end()

        print(f"Final model saved to: {final_model_path}")
        print(f"Final evaluation results saved to: {final_eval_path}")

    # Clean up
    cleanup()


def train():
    """Main training function that spawns multiple processes"""
    load_dotenv()  # Load environment variables

    # Check available GPUs
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Multi-GPU training requires CUDA.")

    world_size = max(1, torch.cuda.device_count())  # Use available GPUs
    print(f"Starting multi-GPU training with {world_size} GPUs")

    # Spawn processes for distributed training
    mp.spawn(
        train_worker,
        args=(world_size,),
        nprocs=world_size,
        join=True
    )


if __name__ == "__main__":
    train()