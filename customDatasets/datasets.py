import torch
from torch.utils.data import Dataset
import torchvision.transforms.v2 as transforms
from datasets import load_dataset
from scripts.dataset_downloader import download_huggingface_dataset
from tqdm import tqdm
import numpy as np
import torch
import random
import copy
import os
import warnings
    
class CustomImageDataset(Dataset):

    def __init__(self,dataset_loc = 'Data/Oxford-IIIT-Pet-Augmented', augmentations_per_datapoint = 0, split='validation', cache=False):
        
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

        self.cache = cache

        self.dataset_length = len(self.dataset) * self.augmentations_per_datapoint

        if self.cache:
            cache_file = os.path.join(dataset_loc, f"{split}_dataset.pt")
            if os.path.exists(cache_file):
                self.dataset_cache = torch.load(cache_file, weights_only=True)
            else:
                print(f"Cache not found. Creating and saving dataset cache at {cache_file}")
                self.dataset_cache = []
                for datapoint in tqdm(self.dataset, desc=f"Caching {split} dataset:", leave=False, total=len(self.dataset)):
                    self.dataset_cache.append(self._deserialize_datapoint(datapoint))
                    torch.save(self.dataset_cache, cache_file)
                    self.dataset_cache = torch.load(cache_file)

            del self.dataset

    def __len__(self):
        return self.dataset_length

    
    def _deserialize_datapoint(self,datapoint):
        image = self._deserialize_numpy(datapoint['image'])
        mask = self._deserialize_numpy(datapoint['mask'],shape=(256,256))

        image = torch.from_numpy(image).permute(2,0,1).float() / 255.0

        cat_mask = np.where(mask == 38, 1,0)
        dog_mask = np.where(mask == 75, 2,0)
        uncertianty_mask = np.where(mask == 255,1,0)

        if cat_mask.sum() > 0:
            mask = cat_mask + uncertianty_mask
        else:
            mask = dog_mask + 2*uncertianty_mask

        return image, torch.tensor(mask)
    
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))
    
    def __getitem__(self, idx):

        image_index = idx // self.augmentations_per_datapoint

        if self.cache:
            image, mask = self.dataset_cache[image_index]
        else:
            datapoint = self.dataset[image_index]
            image, mask = self._deserialize_datapoint(datapoint)

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
    
class PromptImageDataset(Dataset):
    def __init__(
        self,
        dataset_loc="Data/Oxford-IIIT-Pet-Augmented",
        split="train",
        gaussian_sigma=None,
    ):
        """
        dataset_loc: Hugging Face dataset identifier or local path.
        split: 'train', 'validation', or 'test'.
        prompt_gaussian_sigma: if not None, use a Gaussian with this sigma for the prompt heatmap;
                               if None, just use a single binary point.
        """
        if split not in ["train", "validation", "test"]:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")

        # Attempt to load the dataset
        try:
            self.dataset = load_dataset(dataset_loc, split=split)
        except Exception as e:
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                print("Error: The dataset was not found locally. Downloading it now...")
                download_huggingface_dataset(
                    "mattidebeer/Oxford-IIIT-Pet-Augmented", dataset_loc, split=split
                )
                self.dataset = load_dataset(dataset_loc, split=split)
            else:
                print(f"An unexpected error occurred: {e}")
                raise e

        self.gaussian_sigma = gaussian_sigma

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        """Return (image, prompt_map, label)."""

        # 1) Get the raw data
        datapoint = self.dataset[idx]
        image, cat_mask, dog_mask, background_mask = self._deserialize(datapoint)

        # 2) Randomly pick whether to sample from cat, dog, or background
        chosen_class = self._choose_class(cat_mask, dog_mask, background_mask)

        # 3) Generate the prompt map
        prompt_map = self._create_prompt_map(chosen_class, cat_mask, dog_mask, background_mask)

        # 4) Build the segmentation label (3 channels: cat, dog, background)
        cat_tensor = cat_mask.unsqueeze(0)  # [1, H, W]
        dog_tensor = dog_mask.unsqueeze(0)  # [1, H, W]
        bg_tensor  = background_mask.unsqueeze(0)
        label = torch.cat([cat_tensor, dog_tensor, bg_tensor], dim=0)  # [3, H, W]

        return image, prompt_map, label

    def _deserialize(self, datapoint):
        """Read image + masks and return (image_t, cat_mask, dog_mask, bg_mask)."""
        image_np = self._deserialize_numpy(datapoint["image"], shape=(256,256,3))
        mask_np  = self._deserialize_numpy(datapoint["mask"], shape=(256,256))

        # Convert to torch
        image_t = torch.from_numpy(image_np).permute(2,0,1).float() / 255.0

        cat_mask = torch.from_numpy((mask_np == 38).astype(np.float32))
        dog_mask = torch.from_numpy((mask_np == 75).astype(np.float32))
        bg_mask  = 1.0 - (cat_mask + dog_mask)

        return image_t, cat_mask, dog_mask, bg_mask

    def _deserialize_numpy(self, byte_data, shape=(256,256,3), dtype=np.uint8):
        arr = np.frombuffer(byte_data, dtype=dtype).reshape(shape)
        return arr.copy()

    def _choose_class(self, cat_mask, dog_mask, bg_mask):
        """
        Decide which class to prompt (cat, dog, or background).
        Weighted by number of pixels in each region.
        """
        cat_pixels = int(cat_mask.sum().item())
        dog_pixels = int(dog_mask.sum().item())
        bg_pixels  = int(bg_mask.sum().item())

        total = cat_pixels + dog_pixels + bg_pixels
        if total == 0:
            return "bg"

        choice = random.choices(
            population=["cat", "dog", "bg"],
            weights=[cat_pixels, dog_pixels, bg_pixels],
            k=1
        )[0]
        return choice

    def _create_prompt_map(self, chosen_class, cat_mask, dog_mask, background_mask):
        """Pick a random pixel in the chosen mask and create a binary or Gaussian heatmap."""
        if chosen_class == "cat":
            coords = torch.nonzero(cat_mask)
        elif chosen_class == "dog":
            coords = torch.nonzero(dog_mask)
        else:
            coords = torch.nonzero(background_mask)

        # fallback if no pixels found
        if coords.shape[0] == 0:
            cy, cx = 128, 128
        else:
            idx = random.randint(0, coords.shape[0] - 1)
            cy, cx = coords[idx].tolist()

        h, w = cat_mask.shape
        if self.gaussian_sigma is not None:
            heatmap = self._create_gaussian((cy, cx), (h, w), sigma=self.gaussian_sigma)
        else:
            # Just binary
            heatmap = torch.zeros((h, w), dtype=torch.float32)
            heatmap[cy, cx] = 1.0

        return heatmap.unsqueeze(0)  # shape [1, H, W]

    def _create_gaussian(self, center_xy, shape, sigma=10.0):
        """Create a 2D Gaussian heatmap."""
        cy, cx = center_xy
        h, w = shape
        yv, xv = torch.meshgrid(
            torch.arange(h, dtype=torch.float32),
            torch.arange(w, dtype=torch.float32),
            indexing='ij'
        )
        dist_sq = (xv - cx)**2 + (yv - cy)**2
        heatmap = torch.exp(-dist_sq / (2.0 * sigma**2))
        return heatmap

    
