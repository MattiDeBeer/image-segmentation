import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
import numpy as np
from torchvision import transforms
from datasets import load_dataset
import copy
from scripts.dataset_downloader import download_huggingface_dataset
import random
    
class CustomImageDataset(Dataset):

    def __init__(self,dataset_loc = 'Data/Oxford-IIIT-Pet-Augmented', augmentations_per_datapoint = 2, split='validation'):
        
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")
        
        assert isinstance(augmentations_per_datapoint,int) and augmentations_per_datapoint > 0, f"You must choose a positive integer for augmentations per datapoint, you choose: {augmentations_per_datapoint}"

        try:
            self.dataset = load_dataset(dataset_loc, split=split)
        except Exception as e: 
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                print(f"Error: The dataset was not found locally")
                download_huggingface_dataset("mattidebeer/Oxford-IIIT-Pet-Augmented",dataset_loc,split=split)
                self.dataset = load_dataset(dataset_loc, split=split)
            else:
                print(f"An unexpected error occurred: {e}")

        self.augmentations_per_datapoint = augmentations_per_datapoint + 1

        self.image_transform  = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),

            ### Image only transforms ###
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(21)
        )
        
        self.mask_transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
        )

    def __len__(self):
        return(len(self.dataset) * self.augmentations_per_datapoint)
    
    def _transform_datapoint(self,image,mask):

        #Generate a random seed
        seed = torch.randint(0, 2**32, (1,)).item()

        random.seed(seed)
        torch.manual_seed(seed)
        image = self.image_transform(image)

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask.unsqueeze(0)).squeeze(0)

        return image, mask
    
    def _deserialize_datapoint(self,datapoint):
        image = self._deserialize_numpy(datapoint['image'])
        mask = self._deserialize_numpy(datapoint['mask'],shape=(256,256))

        image = torch.from_numpy(image).permute(2,0,1)
        mask = torch.from_numpy(np.where(mask == 255, 0, np.where(mask == 38, 1, np.where(mask == 75, 2, 0)))).long()

        return image, mask
    
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))
    
    def __getitem__(self, idx):

        image_index = idx // self.augmentations_per_datapoint

        datapoint = self.dataset[image_index]
        image, mask = self._deserialize_datapoint(datapoint)

        if idx % self.augmentations_per_datapoint != 0:
            image, mask = self._transform_datapoint(image,mask)

        return image, mask


        
    
        
