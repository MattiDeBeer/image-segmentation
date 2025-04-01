from customDatasets.datasets import ImageDatasetClasses
from models.UNet import UNet
from torch.utils.data import DataLoader
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header,log_loss_to_csv
import torch
import torch.optim as optim
from tqdm import tqdm
from models.losses import *


###### Hyperperameters ###########
model = UNet(out_channels = 3)

num_epochs = 250
batch_size = 120

uncertianty_mask_coefficient = 0

model_save_file = "saved-models/Unets"

## for Cluster
dataset_loc = 'Dataset/Oxford-IIIT-Pet-Augmented'

## for DICE computers
#dataset_loc = "mattidebeer/Oxford-IIIT-Pet-Augmented" #uncomment to load remote dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

tqdm_disable = False

torch.set_float32_matmul_precision('high')
##############################

# Clear unused memory from the cache
torch.cuda.empty_cache()

save_location = get_next_run_folder(model_save_file)

train_dataset = ImageDatasetClasses(dataset=dataset_loc,split='train')
validation_dataset = ImageDatasetClasses(dataset=dataset_loc,split = 'validation')

train_dataloader = DataLoader(train_dataset,batch_size = batch_size)
validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size)

print('Dataset loaded sucesfully')

model = torch.compile(model, mode="max-autotune")
model.to(device)  # Then move to GPU

optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = HybridLoss()

num_params = sum(p.numel() for p in model.parameters())

save_training_info(model,
                   optimizer,
                   criterion,
                   train_dataloader,
                   validation_dataloader,
                   save_location, 
                   extra_params = {'uncertianty_mask_coefficient' : uncertianty_mask_coefficient,
                                   'num_params' : num_params})


write_csv_header(save_location)

for epoch in tqdm( range(num_epochs), desc='Training', unit = 'Epoch', leave = False, disable=tqdm_disable):

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0

    # Training loop
    for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False, disable=tqdm_disable):
        inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
        optimizer.zero_grad()  # Zero gradients from the previous step

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)  # Compute the loss

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Track the loss
        running_loss += loss.item()

    # Calculate average training loss and accuracy
    avg_train_loss = running_loss / len(train_dataloader)

    # Validation loop
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    running_iou_loss = 0.0
    running_pixel_acc_loss = 0.0
    running_dice_loss = 0.0

    with torch.no_grad():  # No gradients needed during validation
        for inputs, targets in tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Validation",leave=False,disable=tqdm_disable):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device

            # Forward pass
            outputs = model(inputs)

            # Calculate different losses
            hybrid_loss = criterion(outputs, targets)
            iou_loss = IoULoss()(outputs, targets)
            pixel_acc_loss = PixelAccuracyLoss()(outputs, targets)
            dice_loss = DiceLoss()(outputs, targets)

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

    tqdm.write(f"Epoch: {epoch}")
    tqdm.write(f"Train Loss: {avg_train_loss:.4f}")
    tqdm.write(f"Validation Loss: {avg_val_loss:.4f}")
    tqdm.write(f"Val IoU: {avg_iou_loss:.4f}")
    tqdm.write(f"Val Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
    tqdm.write(f"Val Dice: {avg_dice_loss:.4f}")
    tqdm.write('\n')

    log_loss_to_csv(epoch,avg_train_loss,avg_val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss, save_location)

    torch.save(model.state_dict(), f'{save_location}model_{epoch+1}.pth')








