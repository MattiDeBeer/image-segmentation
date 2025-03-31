#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 30 17:50:01 2025

@author: jamie
"""

from customDatasets.datasets import ImageDataset, DummyDataset
# Import the autoencoder model instead of UNet
from models.autoencoder import Autoencoder  
from torch.utils.data import DataLoader
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header, log_loss_to_csv
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm  # For progress bar
from torchsummary import summary

###### Hyperparameters ###########
# Instantiate the autoencoder with matching input/output channels (e.g., RGB: 3)
model = Autoencoder(in_channels=3, out_channels=3)
#summary(model, input_size=(3, 256, 256))

num_epochs = 200
batch_size = 16

# For autoencoder training, we can use a different save folder if desired
model_save_file = "/tmp/saved-models/Autoencoders"
#dataset_loc = 'Datasets/Oxford-IIIT-Pet-Augmented'
dataset_loc = "mattidebeer/Oxford-IIIT-Pet-Augmented" #uncomment to load remote dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
##############################

save_location = get_next_run_folder(model_save_file)

# Use the same dataset, but note: for autoencoder, the target is the input image.
# Here we assume ImageDataset returns (image, label) but we will ignore the label.
#train_dataset = DummyDataset()
#validation_dataset = DummyDataset()

train_dataset = ImageDataset(dataset=dataset_loc, split='train', uncertianty_mask_coeff=0)
validation_dataset = ImageDataset(dataset=dataset_loc, split='validation', uncertianty_mask_coeff=0)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size)
validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size)

if torch.cuda.is_available():
    model = torch.compile(model)  # Compile first (if using torch.compile)
    model.to(device)  # Then move to GPU

# Use MSELoss for reconstruction instead of BCE
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

save_training_info(model,
                   optimizer,
                   criterion,
                   train_dataloader,
                   validation_dataloader,
                   save_location,
                   extra_params = {})

write_csv_header(save_location)

for epoch in tqdm(range(num_epochs), desc='Training', unit='Epoch', leave=False):
    model.train()
    running_loss = 0.0
    
    # Training loop
    for inputs, _ in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False):
        optimizer.zero_grad()  # Zero gradients from the previous step
        
        inputs = inputs.to(device)
        # Autoencoder target is the input image itself
        outputs = model(inputs)
        loss = criterion(outputs, inputs)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
    
    avg_train_loss = running_loss / len(train_dataloader)
    print(f"Train Loss: {avg_train_loss:.4f}")
    
    # Validation loop
    model.eval()
    running_val_loss = 0.0
    
    with torch.no_grad():
        for inputs, _ in tqdm(validation_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
            inputs = inputs.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, inputs)
            running_val_loss += loss.item()
    
    avg_val_loss = running_val_loss / len(validation_dataloader)
    print(f"Validation Loss: {avg_val_loss:.4f}")
    
    log_loss_to_csv(epoch, avg_train_loss, avg_val_loss, save_location)
    
    torch.save(model.state_dict(), f'{save_location}/model_{epoch+1}.pth')
