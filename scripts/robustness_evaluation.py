#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 17:22:03 2025

@author: jamie
"""

import torch
from torch.utils.data import DataLoader
import csv
import random
import numpy as np

#import  classes:
from customDatasets.datasets import CustomImageDataset
from customDatasets.perturbations import (
    GaussianPixelNoise, GaussianBlur, ContrastIncrease, ContrastDecrease,
     BrightnessIncrease, BrightnessDecrease, OcclusionIncrease, SaltPepperNoise)
#import dice loss
from models.losses import Dice
#import model
from models.CLIP_models import ClipUnet
from tqdm import tqdm


def main():
    
  
    device = "cuda" if torch.cuda.is_available() else "cpu"

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    # 1) Recreate the architecture
    model = ClipUnet()
 
    
    ckpt = torch.load("saved-models/ClipUnet/model.pth", map_location=device)
    
    new_ckpt = {}
    for key, val in ckpt.items():
        new_key = key.replace("_orig_mod.", "") if key.startswith("_orig_mod.") else key
        new_ckpt[new_key] = val
    
    model.load_state_dict(new_ckpt, strict=False)
    
    model.to(device)
    model.eval()
    dice_fn = Dice()





    base_test_dataset = CustomImageDataset(
         split="test",
         augmentations_per_datapoint=0)
    perturbations = {
        "gaussian_noise": {
            "class": GaussianPixelNoise,
            "params": [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]
        },
        "gaussian_blur": {
            "class": GaussianBlur,
            "params": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        },
        "contrast_increase": {
            "class": ContrastIncrease,
            "params": [1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.2, 1.25]
        },
        "contrast_decrease": {
            "class": ContrastDecrease,
            "params": [1.0, 0.95, 0.9, 0.85, 0.8, 0.6, 0.4, 0.3, 0.2, 0.1]
        },
        "brightness_increase": {
            "class": BrightnessIncrease,
            "params": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        },
        "brightness_decrease": {
            "class": BrightnessDecrease,
            "params": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        },
        "occlusion": {
            "class": OcclusionIncrease,
            "params": [0, 5, 10, 15, 20, 25, 30, 35, 40, 45]
        },
        "salt_pepper_noise": {
            "class": SaltPepperNoise,
            "params": [0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18]
        }
    }



    results_file = "results/robustness_scores.csv"
    with open(results_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["perturbation_type", "param_value", "mean_dice"])

    


 
        for p_name, p_info in tqdm(perturbations.items()):
             p_class = p_info["class"]
             for param_val in tqdm(p_info["params"]):
                 # 5a) Create the perturbed dataset
                 perturbed_dataset = p_class(base_test_dataset, param_val)
    
                 # 5b) Build a DataLoader
                 loader = DataLoader(perturbed_dataset, batch_size=8, shuffle=False)
    
                 # 5c) Evaluate
                 total_dice = 0.0
                 for images, masks in loader:
                     images, masks = images.to(device), masks.to(device)
                     with torch.no_grad():
                         preds = model(images)
                     # compute dice for the batch
                     batch_dice =dice_fn(preds, masks)
                     total_dice += batch_dice.item()  
    
                 mean_dice = total_dice / len(loader)
    
                 # 5d) Write results to CSV
                 writer.writerow([p_name, param_val, f"{mean_dice:.4f}"])
                 print(f"{p_name} param={param_val}, Dice={mean_dice:.4f}")

    print("Evaluation complete. Results saved to CSV.")

if __name__ == "__main__":
    main()
