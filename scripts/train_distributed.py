import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from customDatasets.datasets import ImageDatasetClasses
from models.UNet import UNet
from torch.utils.data import DataLoader
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header,log_loss_to_csv
from tqdm import tqdm  # For progress bar

#setup funciton
def setup(rank, world_size):
    """Initialize distributed process group"""
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def loss_function(outputs,targets,criterion):
    return criterion(outputs,targets)

def train(num_epochs, model, dataloader, rank, train_sampler,optimizer,criterion,save_location, validation_dataloader):
    for epoch in num_epochs:

        running_loss = 0
        model.train()

        train_sampler.set_epoch(epoch)
        for images, labels in dataloader:
            images, labels = images.cuda(rank), labels.cuda(rank)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs,labels,criterion)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
        val_loss = validate(model,validation_dataloader,)
        train_loss = running_loss / len(dataloader)

        print(f"Rank {rank}, Epoch {epoch}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        if rank == 1:
            log_loss_to_csv(epoch,train_loss,val_loss,save_location)
            torch.save(model.state_dict(), f'{save_location}model_{epoch+1}.pth')

        

def validate(model, dataloader, criterion, rank):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, labels in dataloader:

            images, labels = images.cuda(rank), labels.cuda(rank)
            outputs = model(images)
            loss = loss_function(outputs, labels, criterion)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)

    return avg_loss

#training loop
def train(rank, world_size):

    setup(rank, world_size)

    ### HYPERPARAMETERS ###
   
    model = UNet(out_channels = 3)

    num_epochs = 250
    batch_size = 16

    uncertianty_mask_coefficient = 0

    model_save_file = "saved-models/UNet"

    batch_size = 16

    ######################

    # Clear unused memory from the cache
    torch.cuda.empty_cache()

    #load datasets
    train_dataset = ImageDatasetClasses(dataset="mattidebeer/Oxford-IIIT-Pet-Augmented",split='train')
    val_dataset = ImageDatasetClasses(dataset="mattidebeer/Oxford-IIIT-Pet-Augmented",split='validation')

    #Use DistributedSampler to ensure each GPU gets different data
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler, num_workers=4, pin_memory=True)

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_dataloader = DataLoader(train_dataset, batch_size=batch_size, sampler=val_sampler, num_workers=4, pin_memory=True)

    #compile model and wrap with DDP
    model = torch.compile(model)
    model = DDP(model, device_ids=[rank])

    if rank == 1:
        save_location = get_next_run_folder(model_save_file)
        save_training_info(model,
                   optimizer,
                   criterion,
                   train_dataloader,
                   val_dataloader,
                   save_location, 
                   extra_params = {'uncertianty_mask_coefficient' : uncertianty_mask_coefficient})
        write_csv_header(save_location)

    #Optimizer and loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = torch.nn.CrossEntropyLoss()

    train(num_epochs,model,train_dataloader,rank, train_sampler,optimizer,criterion,val_dataloader)

    #cleanup
    dist.destroy_process_group()

#multi gpu entry point
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size)

