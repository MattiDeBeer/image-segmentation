import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from customDatasets.datasets import CustomImageDataset
from models.UNet import UNet
from torch.utils.data import DataLoader
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header,log_loss_to_csv
from tqdm import tqdm 
from models.losses import HybridLoss, IoU, PixelAccuracy, Dice
from torch import autocast, GradScaler
from models.processing_blocks import DataAugmentor


#setup funciton
def setup(rank, world_size):
    """Initialize distributed process group"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def loss_function(outputs,targets,criterion):
    return criterion(outputs,targets)

def train(num_epochs, model, dataloader,rank, train_sampler,optimizer,criterion,save_location, validation_dataloader):
    gradscaler = GradScaler('cuda')
    for epoch in range(0,num_epochs):

        running_loss = 0
        model.train()

        train_sampler.set_epoch(epoch)
        for images, labels in tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False):
            images, labels = images.cuda(rank), labels.cuda(rank)

            optimizer.zero_grad()

            with autocast(device_type='cuda'):
                outputs = model(images)
                loss = loss_function(outputs,labels,criterion)


            gradscaler.scale(loss).backward()
            gradscaler.step(optimizer)
            gradscaler.update()
            
            running_loss += loss.item()
        
        val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss  = validate(model,validation_dataloader,criterion,rank)
        train_loss = running_loss / len(dataloader)

        tqdm.write(f"Rank {rank}, Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        dist.barrier()

        if rank == 0:
            log_loss_to_csv(epoch,train_loss,val_loss,avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss,save_location)
            torch.save(model.state_dict(), f'{save_location}model_{epoch+1}.pth')

            tqdm.write(f"Val IoU: {avg_iou_loss:.4f}")
            tqdm.write(f"Val Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
            tqdm.write(f"Val Dice: {avg_dice_loss:.4f}")

        

def validate(model, validation_dataloader, criterion, rank):
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    running_iou_loss = 0.0
    running_pixel_acc_loss = 0.0
    running_dice_loss = 0.0
    
    with torch.no_grad():  # No gradients needed during validation
        for inputs, targets in tqdm(validation_dataloader, desc=f"Validation",leave=False):
            inputs, targets = inputs.cuda(rank), targets.cuda(rank)  # Move data to device
            
            # Forward pass
            with autocast(device_type = 'cuda'):
                outputs = model(inputs)
            
                # Calculate different losses
                hybrid_loss = criterion(outputs, targets)
                iou_loss = IoU()(outputs, targets)
                pixel_acc_loss = PixelAccuracy()(outputs, targets)
                dice_loss = Dice()(outputs, targets)
            
            # Track the losses
            running_val_loss += hybrid_loss.item()
            running_iou_loss += iou_loss.item()
            running_pixel_acc_loss += pixel_acc_loss.item()
            running_dice_loss += dice_loss.item()
    
    # Calculate average validation losses
    avg_val_loss = running_val_loss / len(validation_dataloader)
    avg_iou_loss = running_iou_loss / len(validation_dataloader)
    avg_pixel_acc_loss = running_pixel_acc_loss / len(validation_dataloader)
    avg_dice_loss = running_dice_loss / len(validation_dataloader)

    return avg_val_loss, avg_pixel_acc_loss,avg_dice_loss,avg_iou_loss

#training loop
def train_distributed(rank, world_size):

    setup(rank, world_size)
   
    model = UNet(out_channels = 3)

    num_epochs = 200
    batch_size = 16

    model_save_file = "saved-models/Test"

    num_workers = 0

    # Clear unused memory from the cache
    torch.cuda.empty_cache()

    #load datasets
    train_dataset = CustomImageDataset(split='train', cache=True)
    val_dataset = CustomImageDataset(split='validation', cache=True)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=num_workers, pin_memory=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=num_workers, pin_memory=True)

    #compile model and wrap with DDP
    model = model.cuda(rank)  # Move to the correct device
    model = DDP(model, device_ids=[rank])

    

    #Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = HybridLoss()

    save_location = get_next_run_folder(model_save_file)

    if rank == 0:
        save_training_info(model,
                   optimizer,
                   criterion,
                   train_dataloader,
                   val_dataloader,
                   save_location, 
                   extra_params = {})

        write_csv_header(save_location)


    train(num_epochs,model,train_dataloader,rank, train_sampler,optimizer,criterion,save_location,val_dataloader)

    #cleanup
    dist.destroy_process_group()

#multi gpu entry point
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train_distributed, args=(world_size,), nprocs=world_size)

