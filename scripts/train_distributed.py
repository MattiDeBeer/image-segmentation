import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
from models.model_wrappers import DistributedTrainingWrapper
from models.UNet import UNet
from customDatasets.datasets import CustomImageDataset
from models.processing_blocks import DataAugmentor
from torch.utils.data import DataLoader, DistributedSampler
import torch.nn as nn

def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def main_ddp():
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    setup(rank, world_size)



    model = UNet()
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    augmentations_per_datapoint = 4
    data_augmentor = DataAugmentor(augmentations_per_datapoint)

    batch_size = 10
    num_workers = 4

    train_dataset = CustomImageDataset(split="train", augmentations_per_datapoint=augmentations_per_datapoint)

    validation_dataset = CustomImageDataset(split="validation", augmentations_per_datapoint=0)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    val_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    trainer = DistributedTrainingWrapper(rank, ddp_model, train_dataloader, val_dataloader, data_augmentor, batch_size=batch_size)

    trainer.train(2)
    cleanup()

if __name__ == "__main__":
    main_ddp()

# Note: This script should be run using torchrun or similar command to handle distributed training.
# For example:
# torchrun --nproc_per_node=NUM_GPUS scripts/train_distributed.py
# where NUM_GPUS is the number of GPUs you want to use.
# Ensure that the environment variable LOCAL_RANK is set correctly for each process.
# This script assumes that the model and datasets are already set up for distributed training.
# The `DistributedTrainingWrapper` class should handle the distributed training logic.


