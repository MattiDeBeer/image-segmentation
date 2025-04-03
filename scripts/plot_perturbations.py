#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 16:28:11 2025

@author: jamie
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

# Import your base dataset & the new perturbation class
from customDatasets.datasets import CustomImageDataset
from customDatasets.perturbations import SaltPepperNoise, GaussianPixelNoise, GaussianBlur, ContrastIncrease, ContrastDecrease, BrightnessIncrease, BrightnessDecrease, OcclusionIncrease

def plot_example(original_image, perturbed_image):
    """
    original_image, perturbed_image: Tensors in [C,H,W] (float [0..1]).
    Displays them side-by-side using matplotlib.
    """
    # Convert from [C,H,W] to [H,W,C]
    orig_np = original_image.permute(1, 2, 0).cpu().numpy()
    pert_np = perturbed_image.permute(1, 2, 0).cpu().numpy()
    
    # For plotting, clamp to [0..1]
    orig_np = np.clip(orig_np, 0, 1)
    pert_np = np.clip(pert_np, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    
    axes[0].imshow(orig_np)
    axes[0].set_title("Original Image")
    axes[0].axis('off')
    
    axes[1].imshow(pert_np)
    axes[1].set_title("Perturbed Image")
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.show()

def main():
    # 1) Load your base test dataset (images in [0..1], no random augmentation)
    base_test_dataset = CustomImageDataset(
        split='test', 
        augmentations_per_datapoint=0
    )
    
    # 2) Wrap it with the perturbation you want to test
    #    For example: SaltPepperNoise with amount=0.1 (10% of pixels are salt/pepper)
    sp_dataset = GaussianPixelNoise(base_test_dataset, standard_deviation = 18)
    
    # 3) Grab a single sample from the base dataset & from the perturbation dataset
    sample_idx = 0
    original_img, _ = base_test_dataset[sample_idx]   # shape [C,H,W] in [0..1]
    perturbed_img, _ = sp_dataset[sample_idx]
    
    # 4) Plot side-by-side
    plot_example(original_img, perturbed_img)

if __name__ == "__main__":
    main()
