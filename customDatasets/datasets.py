import torch
from torch.utils.data import Dataset
from PIL import Image
import os
import sys
import numpy as np
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
import copy
from scripts.dataset_downloader import download_huggingface_dataset
import random
    
class CustomImageDataset(Dataset):

    def __init__(self,dataset_loc = 'Data/Oxford-IIIT-Pet-Augmented', augmentations_per_datapoint = 2, split='validation'):
        
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")
        
        assert isinstance(augmentations_per_datapoint,int) and augmentations_per_datapoint >= 0, f"You must choose a positive integer for augmentations per datapoint, you choose: {augmentations_per_datapoint}"

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

        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0

        background_mask = np.where(mask == 255, 1,0)
        cat_mask = np.where(mask == 38, 1, 0)
        dog_mask = np.where(mask == 75,2,0)

        if np.sum(cat_mask) > 0:
            mask = cat_mask + background_mask
        else:
            mask = dog_mask + 2* background_mask

        return image, torch.tensor(mask)
    
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))
    
    def __getitem__(self, idx):

        image_index = idx // self.augmentations_per_datapoint

        datapoint = self.dataset[image_index]
        image, mask = self._deserialize_datapoint(datapoint)

        if idx % self.augmentations_per_datapoint != 0:
            image, mask = self._transform_datapoint(image,mask)

        return image, mask


class DummyDataset:
 
     def __init__(self,image_channels=3,width=256,height=256,label_channels=2,length = 100):
         self.image_channels = image_channels
         self.width = width
         self.height = height
         self.label_channels = label_channels
         self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
         self.length = length
 
     def __len__(self):
         return self.length
 
     def __getitem__(self,idx):
 
        image = torch.rand(self.image_channels,self.width,self.height)
        if self.label_channels == 1:
            label = torch.randint(0, 3, (self.width, self.height)).long()
        else:
             label = torch.softmax(torch.rand(self.label_channels,self.width,self.height),dim=0)

        return image, label


class CustomImageDatasetRobust(Dataset):

    def __init__(self,dataset_loc = 'Data/Oxford-IIIT-Pet-Augmented', augmentations_per_datapoint = 2, split='validation'):
        
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")
        
        assert isinstance(augmentations_per_datapoint,int) and augmentations_per_datapoint >= 0, f"You must choose a positive integer for augmentations per datapoint, you choose: {augmentations_per_datapoint}"

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

        image = torch.from_numpy(image).permute(2,0,1).float() 

        background_mask = np.where(mask == 255, 1,0)
        cat_mask = np.where(mask == 38, 1, 0)
        dog_mask = np.where(mask == 75,2,0)

        if np.sum(cat_mask) > 0:
            mask = cat_mask + background_mask
        else:
            mask = dog_mask + 2* background_mask

        return image, torch.tensor(mask)
    
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))
    
    def __getitem__(self, idx):

        image_index = idx // self.augmentations_per_datapoint

        datapoint = self.dataset[image_index]
        image, mask = self._deserialize_datapoint(datapoint)

        if idx % self.augmentations_per_datapoint != 0:
            image, mask = self._transform_datapoint(image,mask)

        return image, mask


        
class ClassImageDataset(Dataset):

    def __init__(self,dataset_loc = 'Data/Oxford-IIIT-Pet-Augmented', augmentations_per_datapoint = 2, split='validation'):
        
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")
        
        assert isinstance(augmentations_per_datapoint,int) and augmentations_per_datapoint >= 0, f"You must choose a positive integer for augmentations per datapoint, you choose: {augmentations_per_datapoint}"

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

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.image_transform  = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),

            ### Image only transforms ###
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(21)
        ).to(self.device)
        
        self.mask_transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
        ).to(self.device)

    def __len__(self):
        return(len(self.dataset) * self.augmentations_per_datapoint)
    
    def _transform_datapoint(self,image,mask):

        #Generate a random seed
        seed = torch.randint(0, 2**32, (1,)).item()

        random.seed(seed)
        torch.manual_seed(seed)
        image = self.image_transform(image.to(self.device)).to('cpu')

        random.seed(seed)
        torch.manual_seed(seed)
        mask = self.mask_transform(mask.unsqueeze(0).to(self.device)).squeeze(0).to('cpu')

        return image, mask
    
    def _deserialize_datapoint(self,datapoint):
        image = self._deserialize_numpy(datapoint['image'])
        mask = self._deserialize_numpy(datapoint['mask'],shape=(256,256))

        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0

        background_mask = np.where(mask == 255, 1,0)
        cat_mask = np.where(mask == 38, 1,0)
        dog_mask = np.where(mask == 75,1,0)

        if np.sum(cat_mask) > 0:
            label = 0
        else:
            label = 1

        segment_mask = cat_mask + dog_mask + background_mask

        return image, torch.tensor(segment_mask).float(), torch.tensor([label]).float()
    
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))
    
    def __getitem__(self, idx):

        image_index = idx // self.augmentations_per_datapoint

        datapoint = self.dataset[image_index]
        image, mask, label = self._deserialize_datapoint(datapoint)

        if idx % self.augmentations_per_datapoint != 0:
            image, mask = self._transform_datapoint(image,mask)

        return image, (mask, label)
        

class ClassImageDatasetGPU(Dataset):

    def __init__(self,dataset_loc = 'Data/Oxford-IIIT-Pet-Augmented', augmentations_per_datapoint = 2, split='validation'):
        
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")
        
        assert isinstance(augmentations_per_datapoint,int) and augmentations_per_datapoint >= 0, f"You must choose a positive integer for augmentations per datapoint, you choose: {augmentations_per_datapoint}"

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

    def __len__(self):
        return(len(self.dataset) * self.augmentations_per_datapoint)
    
    
    def _deserialize_datapoint(self,datapoint):
        image = self._deserialize_numpy(datapoint['image'])
        mask = self._deserialize_numpy(datapoint['mask'],shape=(256,256))

        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0

        background_mask = np.where(mask == 255, 1,0)
        cat_mask = np.where(mask == 38, 1,0)
        dog_mask = np.where(mask == 75,1,0)

        if np.sum(cat_mask) > 0:
            label = 0
        else:
            label = 1

        segment_mask = cat_mask + dog_mask + background_mask

        return image, torch.tensor(segment_mask).float(), torch.tensor([label]).float()
    
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))
    
    def __getitem__(self, idx):

        image_index = idx // self.augmentations_per_datapoint

        datapoint = self.dataset[image_index]
        image, mask, label = self._deserialize_datapoint(datapoint)

        return image, (mask, label)


class CustomCollateFn:
    def __init__(self, augmentations_per_datapoint):
    
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.augmentations_per_datapoint = 2

        self.image_transform  = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),

            ### Image only transforms ###
            transforms.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2),
            transforms.GaussianBlur(21)
        ).to(self.device)
        
        self.mask_transform = torch.nn.Sequential(
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(90),
        ).to(self.device)

    def __call__(self, batch):

        images, labels = zip(*batch)  # Unpack batch

        images = torch.stack(images)
        masks, class_labels = zip(*labels)

        masks = torch.stack(masks)
        class_labels = torch.stack(class_labels)

        masks = masks.pin_memory().to(self.device, non_blocking=True)
        images = images.pin_memory().to(self.device, non_blocking=True)
        class_labels = class_labels.pin_memory().to(self.device, non_blocking=True)

        saved_images = images[::self.augmentations_per_datapoint+1]
        saved_masks = masks[::self.augmentations_per_datapoint+1]

        #Generate a random seed
        seed = torch.randint(0, 2**32, (1,)).item()

        random.seed(seed)
        torch.manual_seed(seed)
        transformed_images = self.image_transform(images)

        random.seed(seed)
        torch.manual_seed(seed)
        transformed_masks = self.mask_transform(masks.unsqueeze(1)).squeeze(1)

        transformed_images[::self.augmentations_per_datapoint+1] = saved_images
        transformed_masks[::self.augmentations_per_datapoint+1] = saved_masks

        labels = (transformed_masks,class_labels)

        return transformed_images, labels
