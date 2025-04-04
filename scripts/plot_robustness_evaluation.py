#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 18:52:49 2025

@author: jamie
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  3 XX:XX:XX 2025

Author: jamie

Script: plot_robustness_curves.py
Reads the CSV with columns [perturbation_type, param_value, mean_dice]
and produces a separate plot for each perturbation type.
"""

import csv
import matplotlib.pyplot as plt
from collections import defaultdict
import os

def main():
    csv_path = "results/robustness_scores.csv"  # Path to your CSV file
    output_dir = "results/plots/"  # Where to save the plots

    # Make sure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # 1) Read CSV into a dictionary:
    #    data[ptype] = list of (param, dice)
    data = defaultdict(list)
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ptype = row["perturbation_type"]
            param_str = row["param_value"]
            dice_str = row["mean_dice"]

            # Convert param and dice to float
            # (param could be int (0,2,4...) or float (0.02, 0.04...), so we parse as float)
            param_val = float(param_str)
            dice_val = float(dice_str)
            
            data[ptype].append((param_val, dice_val))

    # 2) For each ptype, sort data by param and generate a plot
    for ptype, values in data.items():
        # Sort by param_value so the line plot is in ascending order
        values.sort(key=lambda x: x[0])  # x: (param, dice)

        # Separate into X, Y
        x_vals = [v[0] for v in values]
        y_vals = [v[1] for v in values]

        # 3) Plot
        plt.figure()
        plt.plot(x_vals, y_vals, marker='o')
        plt.title(f"{ptype} - Dice vs. Param")
        plt.xlabel("Perturbation Parameter")
        plt.ylabel("Mean Dice Score")
        plt.grid(True)

        # 4) Save the plot
        out_path = os.path.join(output_dir, f"{ptype}_robustness_plot.png")
        plt.savefig(out_path)
        plt.close()

    print(f"Plots saved to: {output_dir}")

if __name__ == "__main__":
    main()
