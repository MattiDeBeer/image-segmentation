import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from models.model_wrappers import DistributedTrainingWrapper
from models.UNet import UNet
from customDatasets.datasets import CustomImageDataset
from models.processing_blocks import DataAugmentor
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup_ddp():
    dist.destroy_process_group()

def main_ddp(rank, world_size):
    setup_ddp(rank, world_size)

    batch_size = 10
    augmentations_per_datapoint = 4
    num_workers = 0

    model = UNet.to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    DataAugmentor = DataAugmentor(augmentations_per_datapoint)

    train_dataset = CustomImageDataset(split="train",augmentations_per_datapoint=augmentations_per_datapoint)
    validation_dataset = CustomImageDataset(split="validation",augmentations_per_datapoint=0)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    val_sampler = DistributedSampler(validation_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(validation_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    trainer = DistributedTrainingWrapper(rank, ddp_model, train_dataloader, val_dataloader, num_workers=num_workers, batch_size=batch_size, augmentor=DataAugmentor)
    trainer.train(10)

    cleanup_ddp()

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(main_ddp, args=(world_size,), nprocs=world_size)

