#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 15:56:22 2025

@author: jamie
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
print(os.path.exists("saved-models/Autoencoders/model_179/loss.csv"))




# Load CSV file
df = pd.read_csv(
    "saved-models/Autoencoders/model_179/loss.csv",
    engine="python",
    encoding="utf-8-sig"
)

print(df.head())
# Plot
plt.figure(figsize=(10, 6))
plt.plot(df["Epoch"], df["Train Loss"], label="Train Loss")
plt.plot(df["Epoch"], df["Validation Loss"], label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Train vs Validation Loss")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
