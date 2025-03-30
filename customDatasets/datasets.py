import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import numpy as np
import torchvision.transforms as transforms
from datasets import load_dataset
import copy


# Dataset class for handling image data with buffered loading
class ImageDataset(Dataset):
    # Initialize the dataset with specified parameters
    def __init__(self, dataset="mattidebeer/Oxford-IIIT-Pet-Augmented", split='train', transform=None, uncertianty_mask_coeff = 0):
        
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")
        
        self.split = split

        assert uncertianty_mask_coeff >= 0 and uncertianty_mask_coeff <= 1, f"Uncertianty mask coefficient must be a value between 0 and 1, you inputted {uncertianty_mask_coeff}"
        
        self.uncertianty_mask_coeff = uncertianty_mask_coeff

        self.dataset = self.dataset = load_dataset(dataset,split=split)

        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    # Return the total length of the dataset
    def __len__(self):
        return len(self.dataset)
    
    # Format a single datapoint by processing image and mask
    def _format_datapoint(self,datapoint):
        image = self._deserialize_numpy(datapoint['image'])
        image = image.astype(np.float32) / 255.0
        mask = self._deserialize_numpy(datapoint['mask'],shape=(256,256))
        cat_mask = torch.from_numpy(np.where(mask == 38,1,0)).unsqueeze(0) #cat
        dog_mask = torch.from_numpy(np.where(mask == 75,1,0)).unsqueeze(0) #dog

        cat_mask = cat_mask.to(self.device)
        dog_mask = dog_mask.to(self.device)

        if self.uncertianty_mask_coeff != 0:

            uncertianty_mask = torch.from_numpy(np.where(mask == 255,1,0)).unsqueeze(0)
            uncertianty_mask = uncertianty_mask.to(self.device)

            if cat_mask.any():
                cat_mask = cat_mask + self.uncertianty_mask_coeff * uncertianty_mask
            else:
                dog_mask = dog_mask + self.uncertianty_mask_coeff * uncertianty_mask


        image = torch.from_numpy(image).to(self.device).permute(2,0,1)
        label = torch.cat([cat_mask,dog_mask],dim=0)

        return image, label
    
    # Helper function to deserialize numpy arrays from byte data
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))

    # Get a single item from the dataset
    def __getitem__(self, idx):

        # Get and format datapoint
        datapoint = self.dataset[idx]
        image, label = self._format_datapoint(datapoint)

        return image, label
    
class DummyDataset:

    def __init__(self,image_channels=3,width=256,height=256,label_channels=2):
        self.image_channels = image_channels
        self.width = width
        self.height = height
        self.label_channels = label_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return 100
    
    def __getitem__(self,idx):

        image = torch.rand(self.image_channels,self.width,self.height).to(self.device)
        label = torch.rand(self.label_channels,self.width,self.height).to(self.device)

        return image, label