#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 16:45:00 2025

@author: jamie
"""

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from customDatasets.datasets import PromptImageDataset  # or wherever your class is defined

# Instantiate your dataset (adjust parameters if needed)
dataset = PromptImageDataset(
    dataset_loc="mattidebeer/Oxford-IIIT-Pet-Augmented",
    split="train",
    gaussian_sigma=5.0,
)

# Fetch a single sample
image, prompt_map, label = dataset[1]

print("Image shape:", image.shape)      # Expected: [3, 256, 256]
print("Prompt shape:", prompt_map.shape)    # Expected: [1, 256, 256]
print("Label shape:", label.shape)        # Expected: [3, 256, 256] (for 3-class mask)

# Visualize the sample
def visualize_sample(image, prompt, label):
    # Convert image to numpy and transpose for plotting (H x W x C)
    img_np = image.detach().cpu().numpy().transpose(1, 2, 0)
    prompt_np = prompt.detach().cpu().numpy()[0]  # single channel
    label_np = label.detach().cpu().numpy()         # shape: [3, 256, 256]
    # Create composite label by taking argmax across channels
    composite_label = np.argmax(label_np, axis=0)
    
    # Determine the prompt location (coordinates of maximum value in prompt map)
    cy, cx = np.unravel_index(np.argmax(prompt_np), prompt_np.shape)
    
    # Determine which class is predicted at that coordinate
    class_at_prompt = composite_label[cy, cx]
    class_map = {0: 'cat', 1: 'dog', 2: 'background'}
    
    print(f"Prompt located at pixel ({cy}, {cx}) belongs to class: {class_map.get(class_at_prompt, 'unknown')}")
    
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Image")
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(prompt_np, cmap='hot')
    plt.title("Prompt Heatmap")
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    # For the label, you can show each channel separately or merge them.
    # Here we'll show a composite by taking the argmax across channels.
    composite_label = np.argmax(label_np, axis=0)
    plt.imshow(composite_label, cmap='viridis')
    plt.title("Composite Label")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

visualize_sample(image, prompt_map, label)

# Alternatively, check the DataLoader
dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
for batch in dataloader:
    (imgs, prompts), labels = batch
    print("Batch image shape:", imgs.shape)    # e.g., [2, 3, 256, 256]
    print("Batch prompt shape:", prompts.shape)  # e.g., [2, 1, 256, 256]
    print("Batch label shape:", labels.shape)    # e.g., [2, 3, 256, 256]
    break  # just check one batch
