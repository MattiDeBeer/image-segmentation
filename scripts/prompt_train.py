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

# Import your dataset class
from customDatasets.datasets import PromptImageDataset

# Import your model components:
from models.autoencoder import Autoencoder      
from models.prompt_encoder import PromptEncoder, SegmentationModelWithPrompt    
from models.pre_trained import SegmentationDecoderSkip
num_epochs = 50
batch_size = 16
learning_rate = 0.001
weight_decay = 1e-4
num_classes = 3         # e.g., [cat, dog, background] 
gaussian_sigma = 10.0     # Gaussian sigma for prompt heatmap (or None for binary)

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

autoencoder = Autoencoder(in_channels=3, out_channels=3)
ckpt = torch.load("saved-models/pre_trained_autoencoder_model/model_32.pth")
new_ckpt = {}

for key, val in ckpt.items():
    # If the key starts with "_orig_mod.", remove that prefix
    if key.startswith("_orig_mod."):
        new_key = key.replace("_orig_mod.", "")
    else:
        new_key = key

    new_ckpt[new_key] = val

autoencoder.load_state_dict(new_ckpt)
# Freeze image encoder
for param in autoencoder.encoder.parameters():
    param.requires_grad = False


prompt_encoder = PromptEncoder(out_channels=512)  # For example, output shape: [B, 512, H/8, W/8]

# Instantiate segmentation decoder (using skip connections variant)
decoder = SegmentationDecoderSkip(out_channels=num_classes)



model = SegmentationModelWithPrompt(
    image_encoder=autoencoder.encoder,
    prompt_encoder=prompt_encoder,
    decoder=decoder,
    fusion_method='concat'  # Could also use 'add' if dimensions align
)

model.to(device)

# ===== Optimizer and Loss =====
# Only optimize prompt_encoder and decoder parameters; image encoder is frozen.
optimizer = optim.Adam(
    list(model.prompt_encoder.parameters()) + list(model.decoder.parameters()),
    lr=learning_rate,
    weight_decay=weight_decay
)
# Assume labels are provided as class indices (shape [B, H, W]); use CrossEntropyLoss.
criterion = nn.CrossEntropyLoss()

# Optionally, set up any logging, CSV writers, etc.
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header, log_loss_to_csv
save_location = get_next_run_folder("saved-models/prompt_segmentation")

num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
save_training_info(model, optimizer, criterion, train_loader, val_loader, save_location, extra_params={'num_params': num_params})
write_csv_header(save_location)

# ===== Training Loop =====
for epoch in tqdm(range(num_epochs), desc='Training', unit='Epoch'):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for images, prompts, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
        images = images.to(device, non_blocking=True)    # [B, 3, 256, 256]
        prompts = prompts.to(device, non_blocking=True)  # [B, 1, 256, 256]
        labels = labels.to(device, non_blocking=True)    # if labels are [B, H, W] with class indices
        
        optimizer.zero_grad()
        outputs = model(images, prompts)  # Expected output: [B, num_classes, 256, 256]
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
