#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Training script for prompt-based segmentation with late fusion at the bottleneck,
using the new data augmentation pipeline and logging additional metrics.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
from tqdm import tqdm

# Loss and metric functions
from models.losses import HybridLossBinary, IoUBinary, PixelAccuracyBinary, DiceBinary
# Dataset class
from customDatasets.datasets import PromptImageDataset
# Model component (implements prompt fusion)
from models.prompt_segmentation import ClipUnetPrompt
# Helper functions for logging and saving info
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header, log_loss_to_csv
# New data augmentor for prompt-based segmentation
from models.processing_blocks import DataAugmentorPrompt

# Hyperparameters
num_epochs = 200
batch_size = 16
learning_rate = 0.001
weight_decay = 1e-4
gaussian_sigma = 10.0  # sigma for Gaussian prompts

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Initialize training and validation datasets
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

# Create DataLoader objects
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=True)

# Initialize the model and move it to device
model = ClipUnetPrompt()
model.to(device)

# Initialize optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = HybridLossBinary()

# Initialize metrics
iou_metric = IoUBinary()
pixel_acc_metric = PixelAccuracyBinary()
# For dice, we'll compute it from the IoU value, e.g., dice = 2 * IoU / (1 + IoU)

# Set up logging and checkpoint folder
save_location = get_next_run_folder("/tmp/saved-models/prompt_segmentation")
num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
save_training_info(model, optimizer, criterion, train_loader, val_loader, save_location, extra_params={'num_params': num_params})
write_csv_header(save_location)

# Initialize the data augmentor for prompt segmentation
augmentations_per_datapoint = 4
data_augmentor = DataAugmentorPrompt(augmentations_per_datapoint)
data_augmentor.to(device)

# Training loop
for epoch in tqdm(range(num_epochs), desc='Training', unit='Epoch'):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    # Training loop: note that PromptImageDataset returns (image, prompt_map, label)
    for images, prompt_maps, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} Training", leave=False):
        images = images.to(device, non_blocking=True)        # [B, 3, H, W]
        prompt_maps = prompt_maps.to(device, non_blocking=True)  # [B, 1, H, W]
        labels = labels.to(device, non_blocking=True)          # [B, 3, H, W]
        
        # Apply joint data augmentation: augment images, labels, and prompt maps
        images, labels, prompt_maps = data_augmentor(images, labels, prompt_maps)
        
        optimizer.zero_grad()
        outputs = model(images, prompt_maps)  # Expected output: [B, num_classes, H, W]
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    epoch_time = time.time() - start_time
    avg_train_loss = running_loss / len(train_loader)

    # Validation loop with additional metrics
    model.eval()
    running_val_loss = 0.0
    running_iou_loss = 0.0
    running_pixel_acc_loss = 0.0
    running_dice_loss = 0.0
    with torch.no_grad():
        for images, prompt_maps, labels in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} Validation", leave=False):
            images = images.to(device)
            prompt_maps = prompt_maps.to(device)
            labels = labels.to(device)
            
            outputs = model(images, prompt_maps)
            hybrid_loss = criterion(outputs, labels.long())
            iou_loss = iou_metric(outputs, labels)
            pixel_acc_loss = pixel_acc_metric(outputs, labels)
            # Compute dice loss as a function of IoU: for example, dice = 2*IoU/(1+IoU)
            dice_loss = 2 * iou_loss / (1 + iou_loss)
            
            running_val_loss += hybrid_loss.item()
            running_iou_loss += iou_loss.item()
            running_pixel_acc_loss += pixel_acc_loss.item()
            running_dice_loss += dice_loss.item()
            
    avg_val_loss = running_val_loss / len(val_loader)
    avg_iou_loss = running_iou_loss / len(val_loader)
    avg_pixel_acc_loss = running_pixel_acc_loss / len(val_loader)
    avg_dice_loss = running_dice_loss / len(val_loader)

    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {avg_train_loss:.4f} - Val Loss: {avg_val_loss:.4f} - Time: {epoch_time:.2f}s")
    print(f"Val IoU: {avg_iou_loss:.4f} - Val Pixel Acc: {avg_pixel_acc_loss:.4f} - Val Dice: {avg_dice_loss:.4f}")
    log_loss_to_csv(epoch, avg_train_loss, avg_val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss, save_location)
    # Save model weights every 5 epochs to save sapce
    if (epoch + 1) % 5 == 0:
        torch.save(model.state_dict(), f"{save_location}/model_epoch{epoch+1}.pth")
