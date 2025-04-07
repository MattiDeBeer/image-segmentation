#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 18:27:24 2025

@author: jamie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for prompt-based segmentation with late fusion at the bottleneck.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm
from models.losses import HybridLoss, IoULoss, PixelAccuracyLoss, DiceLoss
# Import your dataset class
from customDatasets.datasets import PromptImageDataset

# Import your model components:    
from models.prompt_segmentation import ClipUnetPrompt
num_epochs = 200
batch_size = 16
learning_rate = 0.001
weight_decay = 1e-4
num_classes = 3         
gaussian_sigma = 10.0     

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

train_dataset = PromptImageDataset(
    dataset_loc="mattidebeer/Oxford-IIIT-Pet-Augmented",
    split="train",
    gaussian_sigma=gaussian_sigma
)
val_dataset = PromptImageDataset(
    dataset_loc="mattidebeer/Oxford-IIIT-Pet-Augmented",
    split="validation",
    gaussian_sigma=gaussian_sigma
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

model = ClipUnetPrompt()
model.to(device)


optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = HybridLoss()
# Optionally, set up any logging, CSV writers, etc.
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header, log_loss_to_csv
save_location = get_next_run_folder("/tmp/saved-models/prompt_segmentation")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
save_training_info(model, optimizer, criterion, train_loader, val_loader, save_location, extra_params={'num_params': num_params})
write_csv_header(save_location)

for epoch in tqdm(range(num_epochs), desc='Training', unit='Epoch'):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for images, prompts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
        images = images.to(device, non_blocking=True)    
        prompts = prompts.to(device, non_blocking=True)  
        labels = labels.to(device, non_blocking=True)    
        
        optimizer.zero_grad()
        outputs = model(images, prompts)  
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        
    epoch_time = time.time() - start_time
    avg_train_loss = running_loss / len(train_loader)
    
    # Validation loop:
    model.eval()
    running_val_loss = 0.0
    with torch.no_grad():
        for images, prompts, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
            images = images.to(device)
            prompts = prompts.to(device)
            labels = labels.to(device)
            
            outputs = model(images, prompts)
            loss = criterion(outputs, labels.long())
            running_val_loss += loss.item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Time: {epoch_time:.2f}s")
    log_loss_to_csv(epoch, avg_train_loss, avg_val_loss, save_location)
    torch.save(model.state_dict(), f"{save_location}/model_epoch{epoch+1}.pth")
