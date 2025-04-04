import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from models.model_wrappers import DistributedTrainingWrapper
from models.UNet import UNet
from customDatasets.datasets import CustomImageDataset
from models.processing_blocks import DataAugmentor
from torch.utils.data import DataLoader, DistributedSampler
import os

def setup_ddp():
    # Get rank and world_size from environment variables set by torchrun
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])

    # Set up CUDA device for this rank
    torch.cuda.set_device(local_rank)

    # Initialize the process group for DDP (NCCL backend for GPUs)
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

    return rank, world_size, local_rank

def cleanup_ddp():
    dist.destroy_process_group()

def main_ddp():
    # Setup DDP and get rank, world_size, local_rank
    rank, world_size, local_rank = setup_ddp()

    batch_size = 16
    augmentations_per_datapoint = 4
    num_workers = 0

    # Initialize the model and move it to the correct GPU
    model = UNet().to(local_rank)
    ddp_model = DDP(model, device_ids=[local_rank])

    # Set up the data augmentor
    data_augmentor = DataAugmentor(augmentations_per_datapoint)

    # Set up datasets and dataloaders
    train_dataset = CustomImageDataset(split="train", augmentations_per_datapoint=augmentations_per_datapoint)
    validation_dataset = CustomImageDataset(split="validation", augmentations_per_datapoint=0)

    # DistributedSampler ensures each process gets a different subset of the data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    val_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    # Initialize the training wrapper
    trainer = DistributedTrainingWrapper(rank, ddp_model, train_dataloader, val_dataloader, data_augmentor, num_workers=num_workers, batch_size=batch_size)
    trainer.train(2)

    # Cleanup DDP
    cleanup_ddp()

if __name__ == "__main__":
    # Use torchrun to spawn multiple processes
    world_size = int(os.environ["WORLD_SIZE"])  # This will be set by torchrun automatically
    mp.spawn(main_ddp, nprocs=world_size, args=())

# Note: This script should be run using torchrun or similar command to handle distributed training.
# For example:
# torchrun --nproc_per_node=NUM_GPUS scripts/train_distributed.py
# where NUM_GPUS is the number of GPUs you want to use.
# Ensure that the environment variable LOCAL_RANK is set correctly for each process.
# This script assumes that the model and datasets are already set up for distributed training.
# The `DistributedTrainingWrapper` class should handle the distributed training logic.


