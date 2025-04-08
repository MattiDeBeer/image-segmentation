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
    """
    A custom PyTorch Dataset class for loading and processing an image segmentation dataset.
    This dataset is designed to work with augmented datasets and supports caching for faster access.
    Attributes:
        dataset (Dataset): The loaded dataset, either from a local directory or downloaded from Hugging Face Hub.
        augmentations_per_datapoint (int): The number of augmentations to apply per datapoint, including the original datapoint.
        cache (bool): Whether to cache the dataset in memory for faster access.
        dataset_length (int): The total length of the dataset, accounting for augmentations.
        dataset_cache (list): A cached version of the dataset, if caching is enabled.
    Methods:
        __len__(): Returns the total number of datapoints in the dataset, including augmentations.
        __getitem__(idx): Retrieves the image and mask for the given index, applying augmentations if necessary.
        _deserialize_datapoint(datapoint): Converts a serialized datapoint into a PyTorch tensor for the image and mask.
        _deserialize_numpy(byte_data, shape, dtype): Converts serialized byte data into a NumPy array with the specified shape and data type.
    """
    def __init__(self,dataset_loc = 'Data/Oxford-IIIT-Pet-Augmented', augmentations_per_datapoint = 0, split='validation', cache=False):
        
        # Ensure the split is valid
        if split not in ['train', 'validation', 'test']:
            raise ValueError(f"split must be one of: 'train', 'validation', 'test'. You selected {split}")
        
        # Validate that augmentations_per_datapoint is a non-negative integer
        assert isinstance(augmentations_per_datapoint, int) and augmentations_per_datapoint >= 0, \
            f"You must choose a positive integer for augmentations per datapoint, you choose: {augmentations_per_datapoint}"

        try:
            # Attempt to load the dataset from the specified location
            self.dataset = load_dataset(dataset_loc, split=split)
        except Exception as e: 
            # Handle the case where the dataset is not found locally or on the Hugging Face Hub
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                print(f"Error: The dataset was not found locally")
                # Download the dataset from the Hugging Face Hub
                download_huggingface_dataset("mattidebeer/Oxford-IIIT-Pet-Augmented", dataset_loc, split=split)
                # Reload the dataset after downloading
                self.dataset = load_dataset(dataset_loc, split=split)
            else:
                # Raise any other unexpected errors
                print(f"An unexpected error occurred: {e}")

        # Increment augmentations_per_datapoint to include the original datapoint
        self.augmentations_per_datapoint = augmentations_per_datapoint + 1

        # Enable caching if specified
        self.cache = cache

        # Calculate the total dataset length, accounting for augmentations
        self.dataset_length = len(self.dataset) * self.augmentations_per_datapoint

        if self.cache:
            # Define the cache file path
            cache_file = os.path.join(dataset_loc, f"{split}_dataset.pt")
            
            # Check if the cache file already exists
            if os.path.exists(cache_file):
                print(f"Loading dataset cache from {cache_file}")
                # Load the cached dataset from the file
                self.dataset_cache = torch.load(cache_file, weights_only=True)
            else:
                print(f"Cache not found. Creating and saving dataset cache at {cache_file}")
                # Create a new cache by deserializing each datapoint
                self.dataset_cache = []
                for datapoint in tqdm(self.dataset, desc=f"Caching {split} dataset:", leave=False, total=len(self.dataset)):
                    self.dataset_cache.append(self._deserialize_datapoint(datapoint))

                # Save the newly created cache to the file
                torch.save(self.dataset_cache, cache_file)
                # Reload the cache to ensure consistency
                self.dataset_cache = torch.load(cache_file, weights_only=True)

            # Remove the original dataset from memory to save space
            del self.dataset

    def __len__(self):
        return self.dataset_length

    
    def _deserialize_datapoint(self,datapoint):
        """
        Deserialize a single datapoint into a PyTorch tensor for the image and a processed mask.

        Args:
            datapoint (dict): A dictionary containing:
                - 'image': Serialized numpy array representing the image.
                - 'mask': Serialized numpy array representing the segmentation mask.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The image tensor with shape (C, H, W) and values normalized to [0, 1].
                - mask (torch.Tensor): The processed mask tensor with integer values representing categories:
                    - 1 for cat regions.
                    - 2 for dog regions.
                    - Additional values for uncertainty regions based on the original mask.
        """
        # Deserialize the image from the serialized numpy array
        image = self._deserialize_numpy(datapoint['image'])
        # Deserialize the mask from the serialized numpy array with the specified shape
        mask = self._deserialize_numpy(datapoint['mask'], shape=(256, 256))

        # Convert the image to a PyTorch tensor, permute dimensions to (C, H, W), and normalize to [0, 1]
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Create binary masks for different regions in the segmentation mask
        cat_mask = np.where(mask == 38, 1, 0)  # Mask for cat regions
        dog_mask = np.where(mask == 75, 2, 0)  # Mask for dog regions
        uncertianty_mask = np.where(mask == 255, 1, 0)  # Mask for uncertain regions

        # Combine masks based on the presence of cat or dog regions
        if cat_mask.sum() > 0:
            # If cat regions exist, combine cat mask with uncertainty mask
            mask = cat_mask + uncertianty_mask
        else:
            # Otherwise, combine dog mask with uncertainty mask (scaled by 2)
            mask = dog_mask + 2 * uncertianty_mask

        # Return the processed image and mask as PyTorch tensors
        return image, torch.tensor(mask)
    
    def _deserialize_numpy(self,byte_data, shape=(256,256,3), dtype=np.uint8):
        # Deserialize the byte data into a NumPy array with the specified shape and data type
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))
    
    def __getitem__(self, idx):
        """
        Retrieve the image and mask for the given index, applying augmentations if necessary.

        Args:
            idx (int): The index of the datapoint to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The image tensor with shape (C, H, W).
                - mask (torch.Tensor): The segmentation mask tensor.
        """
        # Calculate the index of the original image in the dataset
        image_index = idx // self.augmentations_per_datapoint

        if self.cache:
            # If caching is enabled, retrieve the image and mask from the cache
            image, mask = self.dataset_cache[image_index]
        else:
            # Otherwise, deserialize the datapoint from the dataset
            datapoint = self.dataset[image_index]
            image, mask = self._deserialize_datapoint(datapoint)

        return image, mask
    



class DummyDataset:
    """
    A dummy dataset class for generating random image and label data.
    Attributes:
        image_channels (int): Number of channels in the generated images. Default is 3.
        width (int): Width of the generated images. Default is 256.
        height (int): Height of the generated images. Default is 256.
        label_channels (int): Number of channels in the generated labels. If 1, labels are 
                            generated as integer class indices. Otherwise, labels are 
                            generated as softmax probabilities. Default is 2.
        length (int): Number of samples in the dataset. Default is 100.
        device (torch.device): The device on which tensors are created (CPU or GPU).
    Methods:
        __len__():
            Returns the number of samples in the dataset.
        __getitem__(idx):
            Generates a random image and label pair for the given index.
            Args:
                idx (int): Index of the sample to retrieve.
            Returns:
                tuple: A tuple containing:
                    - image (torch.Tensor): A tensor of shape 
                    (image_channels, width, height) representing the random image.
                    - label (torch.Tensor): A tensor representing the random label. If 
                    label_channels is 1, the shape is (width, height) with integer 
                    class indices. Otherwise, the shape is 
                    (label_channels, width, height) with softmax probabilities.
    """
 
    def __init__(self, image_channels=3, width=256, height=256, label_channels=2, length=100):
        """
        Initialize the DummyDataset with the specified parameters.

        Args:
            image_channels (int): Number of channels in the generated images. Default is 3.
            width (int): Width of the generated images. Default is 256.
            height (int): Height of the generated images. Default is 256.
            label_channels (int): Number of channels in the generated labels. If 1, labels are 
                                  generated as integer class indices. Otherwise, labels are 
                                  generated as softmax probabilities. Default is 2.
            length (int): Number of samples in the dataset. Default is 100.
        """
        self.image_channels = image_channels
        self.width = width
        self.height = height
        self.label_channels = label_channels
        self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
        self.length = length

    def __len__(self):
        """
        Return the number of samples in the dataset.

        Returns:
            int: The total number of samples in the dataset.
        """
        return self.length

    def __getitem__(self, idx):
        """
        Generate a random image and label pair for the given index.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): A tensor of shape 
                  (image_channels, width, height) representing the random image.
                - label (torch.Tensor): A tensor representing the random label. If 
                  label_channels is 1, the shape is (width, height) with integer 
                  class indices. Otherwise, the shape is 
                  (label_channels, width, height) with softmax probabilities.
        """
        # Generate a random image tensor with the specified dimensions
        image = torch.rand(self.image_channels, self.width, self.height)

        if self.label_channels == 1:
            # Generate a random label tensor with integer class indices
            label = torch.randint(0, 3, (self.width, self.height)).long()
        else:
            # Generate a random label tensor with softmax probabilities
            label = torch.softmax(torch.rand(self.label_channels, self.width, self.height), dim=0)

        return image, label


    
class CustomImageDatasetNew(Dataset):

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
                print(f"Loading dataset cache from {cache_file}")
                self.dataset_cache = torch.load(cache_file, weights_only=True)
            else:
                print(f"Cache not found. Creating and saving dataset cache at {cache_file}")
                self.dataset_cache = []
                for datapoint in tqdm(self.dataset, desc=f"Caching {split} dataset:", leave=False, total=len(self.dataset)):
                    self.dataset_cache.append(self._deserialize_datapoint(datapoint))

                torch.save(self.dataset_cache, cache_file)
                self.dataset_cache = torch.load(cache_file, weights_only=True)

            del self.dataset

    def __len__(self):
        return self.dataset_length

    
    def _deserialize_datapoint(self,datapoint):
        image = self._deserialize_numpy(datapoint['image'])
        mask = self._deserialize_numpy(datapoint['mask'],shape=(256,256))

        image = torch.from_numpy(image).permute(2,0,1).float() 

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

         # 4) Build the segmentation label as a single-channel tensor with class indices.
        # Assign: 0 for cat, 1 for dog, and 2 for background.
        label = torch.zeros_like(cat_mask, dtype=torch.float32)
        if chosen_class == "cat":
            label[cat_mask == 1] = 1.0
        elif chosen_class == "dog":
            label[dog_mask == 1] = 1.0
        else:
            label[background_mask == 1] = 1.0
        label = label.unsqueeze(0)
    

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

    
