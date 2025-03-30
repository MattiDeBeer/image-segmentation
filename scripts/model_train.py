
from customDatasets.datasets import ImageDataset, DummyDataset
from models.UNet import UNet
from torch.utils.data import DataLoader
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header,log_loss_to_csv
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # For progress bar


###### Hyperperameters ###########
model = UNet()

num_epochs = 10
batch_size = 32

uncertianty_mask_coefficient = 0.5

model_save_file = "saved-models/UNets"
dataset_loc = '../../Datasets/Oxford-IIIT-Pet-Augmented'

#dataset_loc = "mattidebeer/Oxford-IIIT-Pet-Augmented" #uncomment to load remote dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##############################

save_location = get_next_run_folder(model_save_file)

train_dataset = ImageDataset(dataset=dataset_loc,split='test', uncertianty_mask_coeff=uncertianty_mask_coefficient)
validation_dataset = ImageDataset(dataset=dataset_loc,split='test', uncertianty_mask_coeff=0)
train_dataset = ImageDataset(dataset=dataset_loc,split='test', uncertianty_mask_coeff=uncertianty_mask_coefficient)
validation_dataset = ImageDataset(dataset=dataset_loc,split='test', uncertianty_mask_coeff=0)

train_dataloader = DataLoader(train_dataset,batch_size = batch_size)
validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size)

if torch.cuda.is_available():
    model = torch.compile(model)  # Compile first
    model.to(device)  # Then move to GPU

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

save_training_info(model,
                   optimizer,
                   criterion,train_dataloader,
                   validation_dataloader,
                   save_location, 
                   extra_params = {'uncertianty_mask_coefficient' : uncertianty_mask_coefficient})


write_csv_header(save_location)

for epoch in tqdm( range(num_epochs), desc='Training', unit = 'Epoch', leave = False):

    model.train()
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    # Training loop
    for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False):
        
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
    
    print(f"Train Loss: {avg_train_loss:.4f}")
    
    # Validation loop
    model.eval()  # Set the model to evaluation mode
    running_val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():  # No gradients needed during validation
        for inputs, targets in tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Validation",leave=False):
            inputs, targets = inputs.to(device), targets.to(device)  # Move data to device
            
            # Forward pass (no need for backward or optimizer step)
            outputs = model(inputs)
            loss = criterion(outputs, targets)  # Compute the loss
            
            # Track the loss
            running_val_loss += loss.item()
    
    # Calculate average validation loss and accuracy
    avg_val_loss = running_val_loss / len(validation_dataloader)
    
    print(f"Validation Loss: {avg_val_loss:.4f}")

    log_loss_to_csv(epoch,avg_train_loss,avg_val_loss,save_location)
    
    torch.save(model.state_dict(), f'{save_location}model_{epoch+1}.pth')







