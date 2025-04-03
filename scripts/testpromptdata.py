#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 17:28:44 2025

@author: jamie
"""
import numpy as np
from customDatasets.datasets import ImageDatasetPrompt
# Create the dataset
dataset = ImageDatasetPrompt(
    dataset="Datasets/Oxford-IIIT-Pet-Augmented",
    split="train",
    sigma=10.0
)

# Grab a random item
img, prompt, lbl = dataset[0]

print("Image shape:", img.shape)      # Should be [3, 256, 256]
print("Prompt shape:", prompt.shape)  # Should be [1, 256, 256]
print("Label shape:", lbl.shape)      # If 3-class: [3, 256, 256]

import matplotlib.pyplot as plt

def show_sample(img, prompt, lbl):
    # Convert to CPU NumPy for plotting
    img_np = img.detach().cpu().numpy()
    prompt_np = prompt.detach().cpu().numpy()[0]  # single channel
    lbl_np = lbl.detach().cpu().numpy()           # shape [3, 256, 256]

    # Show the image (transpose for [H,W,C])
    plt.figure()
    plt.imshow(np.transpose(img_np, (1,2,0)))
    plt.title("Image")

    # Show the prompt heatmap
    plt.figure()
    plt.imshow(prompt_np, cmap='hot')
    plt.title("Prompt Heatmap")

    # Show label channels separately (cat, dog, background)
    cat_ch = lbl_np[0]
    dog_ch = lbl_np[1]
    bg_ch  = lbl_np[2]

    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(cat_ch, cmap='gray')
    plt.title("Cat Mask")

    plt.subplot(1,3,2)
    plt.imshow(dog_ch, cmap='gray')
    plt.title("Dog Mask")

    plt.subplot(1,3,3)
    plt.imshow(bg_ch, cmap='gray')
    plt.title("Background Mask")

    plt.show()

# Example usage:
show_sample(img, prompt, lbl)
