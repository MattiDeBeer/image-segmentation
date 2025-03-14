#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 17:55:32 2025

@author: matti
"""
import h5py
import numpy as np
from tqdm import tqdm

"""
This script formats the train labels.
It puts each mask in its own channel, depending on its class.
It also adds the unceritanty regions, which can be used in training
"""

with h5py.File('augmented-dataset/TrainVal.h5','r') as data_file:
    with h5py.File('augmented-dataset/TrainValFormatted.h5','w') as out_file:
        images = data_file['images']
        labels = data_file['labels']
        
        out_file.create_dataset('images', images.shape, dtype=np.uint8)
        out_file.create_dataset('labels', (labels.shape[0],2,labels.shape[1],labels.shape[2]), dtype=np.uint8)
        out_file.create_dataset('uncertian_regions', (labels.shape), dtype=np.uint8)
        
        for i in tqdm( range(0,len(images)), desc ='Formatting labels', unit= ' Images'):
            out_file['images'][i] = images[i]
            out_file['uncertian_regions'][i] = np.array(labels[i] == 255, dtype = np.uint8)
            
            out_file['labels'][i] = np.array([ labels[i] == 1, labels[i] == 2], dtype = np.uint8)
            
            
    
    