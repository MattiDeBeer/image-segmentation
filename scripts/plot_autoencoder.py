#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 31 18:09:44 2025

@author: jamie
"""

import torch
import matplotlib.pyplot as plt
from torchvision import transforms
from models.autoencoder import Autoencoder  
from customDatasets.datasets import CustomImageDataset  

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. Load the model and weights
model = Autoencoder(in_channels=3, out_channels=3)  # or your segmentation model
state_dict = torch.load("saved-models/Autoencoders/model_3.pth", map_location=device)
new_state_dict = {}
for key, value in state_dict.items():
    new_key = key.replace("_orig_mod.", "")
    new_state_dict[new_key] = value
model.load_state_dict(new_state_dict)

model.to(device)
model.eval()

# 2. Get a test sample (modify according to your dataset structure)
# Example using a single sample from your test dataset:
test_dataset = CustomImageDataset(split="test",augmentations_per_datapoint=0)
test_img, test_mask = test_dataset[1]

# Convert to float
test_img = test_img.float() / 255.0

test_img_batch = test_img.unsqueeze(0).to(device)
with torch.no_grad():
    pred = model(test_img_batch)


# If it's multi-class segmentation:
pred_mask = torch.argmax(pred, dim=1).cpu().squeeze().numpy()
# For binary segmentation, you might use:
# pred_mask = (pred > 0.5).cpu().squeeze().numpy()

# Also, convert test_mask to a NumPy array if needed
# (adjust this if your ground truth is stored differently)
if isinstance(test_mask, torch.Tensor):
    true_mask = test_mask.cpu().numpy()
else:
    true_mask = test_mask

# 4. Convert the input image for plotting
# If the image is normalized (e.g. [0,1]), this might be enough:
img_np = test_img.cpu().permute(1, 2, 0).numpy()

# 5. Plot the results side-by-side
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Instead of using test_mask as ground truth, use test_img.
fig, axes = plt.subplots(1, 2, figsize=(10, 5))

axes[0].imshow(img_np)
axes[0].set_title("Input Image")
axes[0].axis("off")

# The reconstructed image from the autoencoder
recon_img = pred.cpu().squeeze().permute(1, 2, 0).numpy()
axes[1].imshow(recon_img)
axes[1].set_title("Reconstructed Image")
axes[1].axis("off")

plt.tight_layout()
plt.show()

