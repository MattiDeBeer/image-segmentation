#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:25:11 2025

@author: jamie
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm

# Imports for your dataset, augmentor, and model
# from your_project.datasets import CustomImageDatasetPrompt
# from your_project.augmentors import DataAugmentor
# from your_project.models import ClipUnet
# from your_project.metrics import IoU, Dice, PixelAccuracy


class IoU(nn.Module):
    """
    Computes the Intersection over Union (IoU) for binary segmentation.
    Output is the average IoU over the batch.
    """
    def __init__(self, eps=1e-6, threshold=0.5):
        super().__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(self, logits, targets):
        """
        Args:
            logits:   (B, 1, H, W) raw model outputs (unnormalized)
            targets:  (B, H, W) binary ground truth {0,1}
        Returns:
            Average IoU across the batch (scalar).
        """
        # Apply sigmoid => probability in [0,1]
        probs = torch.sigmoid(logits)

        # Threshold => predictions in {0,1}
        preds = (probs >= self.threshold).float()

        # Flatten for easy intersection/union
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        ious = []
        for b in range(preds.size(0)):
            intersection = (preds[b] * targets[b]).sum()
            union = preds[b].sum() + targets[b].sum() - intersection
            iou = (intersection + self.eps) / (union + self.eps)
            ious.append(iou)

        return torch.mean(torch.stack(ious))
class PixelAccuracy(nn.Module):
    """
    For binary segmentation: fraction of pixels predicted correctly.
    """
    def __init__(self, threshold=0.5):
        super().__init__()
        self.threshold = threshold

    def forward(self, logits, targets):
        """
        Args:
            logits:   (B, 1, H, W) raw model outputs
            targets:  (B, H, W) binary ground truth {0,1}
        Returns:
            Average pixel accuracy across the batch (scalar).
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()

        # Flatten
        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        correct = (preds == targets).float().sum(dim=1)  # correct per sample
        total = preds.size(1)                           # pixels per sample
        accuracy_per_image = correct / total

        return accuracy_per_image.mean()  # average over batch
class Dice(nn.Module):
    """
    Computes Dice coefficient for binary segmentation.
    Equivalent to 2*TP / (2*TP + FP + FN).
    """
    def __init__(self, eps=1e-6, threshold=0.5):
        super().__init__()
        self.eps = eps
        self.threshold = threshold

    def forward(self, logits, targets):
        """
        Args:
            logits:   (B, 1, H, W) raw model outputs
            targets:  (B, H, W)    binary ground truth {0,1}
        Returns:
            Average Dice score across the batch (scalar).
        """
        probs = torch.sigmoid(logits)
        preds = (probs >= self.threshold).float()

        preds = preds.view(preds.size(0), -1)
        targets = targets.view(targets.size(0), -1).float()

        dice_scores = []
        for b in range(preds.size(0)):
            intersection = (preds[b] * targets[b]).sum()
            denom = preds[b].sum() + targets[b].sum()
            dice_score = (2.0 * intersection + self.eps) / (denom + self.eps)
            dice_scores.append(dice_score)

        return torch.mean(torch.stack(dice_scores))


def train_prompt_segmentation(
    train_dataset,
    val_dataset,
    model,
    data_augmentor,
    criterion,
    optimizer,
    num_epochs=10,
    batch_size=4,
    device=None
):
    """
    A simple training function for prompt-based segmentation.
    
    Args:
        train_dataset (Dataset): An instance of CustomImageDatasetPrompt for training.
        val_dataset   (Dataset): An instance of CustomImageDatasetPrompt for validation.
        model (nn.Module): The ClipUnet model with in_channels=4 and out_channels=1.
        data_augmentor (nn.Module): Instance of DataAugmentor that handles 3-channel image + 1-channel prompt + mask.
        criterion (nn.Module): Typically nn.BCEWithLogitsLoss() for binary segmentation.
        optimizer (torch.optim.Optimizer): e.g., optim.Adam(model.parameters(), lr=1e-3).
        num_epochs (int): How many epochs to train.
        batch_size (int): Batch size for DataLoader.
        device (torch.device): 'cuda' or 'cpu'. If None, auto-selects.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Set up DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False)
    
    # Move model + augmentor to correct device
    model.to(device)
    data_augmentor.to(device)
    
    # Metrics (assuming you have your own classes or functions)
    iou = IoU()
    dice = Dice()
    pixel_acc = PixelAccuracy()

    # AMP GradScaler
    scaler = GradScaler()

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        
        # --- Training Loop ---
        for images, prompts, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]"):
            # Move data to device
            images  = images.to(device)
            prompts = prompts.to(device)
            masks   = masks.to(device)

            # Apply augmentations (geometric + color)
            images, prompts, masks = data_augmentor(images, prompts, masks)

            # Concatenate [B,4,H,W] => (3 image channels + 1 prompt channel)
            x = torch.cat([images, prompts], dim=1)
            
            optimizer.zero_grad()
            
            with autocast(device_type=device.type):
                outputs = model(x)                        # [B, 1, H, W]
                loss    = criterion(outputs, masks.float())  # BCEWithLogitsLoss

            # Backprop + update
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)

        # --- Validation Loop ---
        model.eval()
        val_loss = 0.0
        val_iou = 0.0
        val_pixel_acc = 0.0
        val_dice = 0.0

        with torch.no_grad():
            for images, prompts, masks in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]"):
                images  = images.to(device)
                prompts = prompts.to(device)
                masks   = masks.to(device)

                # No augmentations for validation
                x = torch.cat([images, prompts], dim=1)

                with autocast(device_type=device.type):
                    outputs = model(x)
                    loss = criterion(outputs, masks.float())
                    val_loss += loss.item()
                    
                    # Compute metrics
                    iou_val  = iou(outputs, masks)
                    dice_val = dice(outputs, masks)
                    pix_acc  = pixel_acc(outputs, masks)

                    val_iou        += iou_val.item()
                    val_dice       += dice_val.item()
                    val_pixel_acc  += pix_acc.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_iou  = val_iou / len(val_loader)
        avg_val_dice = val_dice / len(val_loader)
        avg_val_pacc = val_pixel_acc / len(val_loader)

        print(f"\n[Epoch {epoch+1}/{num_epochs}]")
        print(f"Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"Val IoU: {avg_val_iou:.4f} | Val Dice: {avg_val_dice:.4f} | Val PixAcc: {avg_val_pacc:.4f}\n")

        # (Optional) Save model each epoch
        torch.save(model.state_dict(), f"clipunet_prompt_epoch_{epoch+1}.pth")


# -------------- EXAMPLE USAGE --------------
if __name__ == "__main__":
    # 1) Create Datasets
    train_dataset = CustomImageDatasetPrompt(
        split="train",
        augmentations_per_datapoint=4,
        cache=True
    )
    val_dataset = CustomImageDatasetPrompt(
        split="validation",
        augmentations_per_datapoint=0,
        cache=True
    )

    # 2) Create Model (in_channels=4, out_channels=1 for binary segmentation)
    model = ClipUnet(in_channels=4, out_channels=1)

    # 3) Create DataAugmentor
    data_augmentor = DataAugmentor(augmentations_per_datapoint=4)

    # 4) Create Loss and Optimizer
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 5) Train
    train_prompt_segmentation(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        data_augmentor=data_augmentor,
        criterion=criterion,
        optimizer=optimizer,
        num_epochs=10,
        batch_size=8  # or any you like
    )
