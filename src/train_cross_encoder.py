import os
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DPP
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm_notebook
from PIL import Image
from dotenv import load_dotenv
from comet_ml import Experiment

from src.multimodal_retriever.retriever import Retriever


def setup(rank, world_size):
    """Initialize the process group for distributed training"""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    dist.init_process_group("ncll", rank=rank, workd_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """Clean up the process group"""
    dist.destroy_process_group()

def collate_fn(batch):
    pass

def train_worker(rank, world_size):
    """Train function for each GPU process"""
    # Setup distributed raining
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # Initialize Comet ML experiment tracking only on rank 0
    experiment = 0
    learning_rate = 2e-4
    batch_size = 48
    num_epochs = 3
    margin = 0.3
    projection_dim = 1024
    num_gpus = world_size
    if rank == 0:
        experiment = Experiment(
            api_key=os.getenv("COMET_API_KEY"),
            project_name="vlsp_multimodal_cross_encoder",
            workspace=os.getenv("COMET_WORKSPACE")
        )
        experiment.log_parameters({
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "margin": margin,
            "num_gpus": num_gpus,
            "loss_fn": "TripletMaginLoss"
        })

    # Initialize the cross encoder
    if rank == 0:
        print("Initializing the reranker...")
    reranker = Retriever(prjection_dim)