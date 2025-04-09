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
class PromptImageDataset(Dataset):
    """
    A dataset for prompt-based segmentation:
      - (image, gaussian_heatmap, binary_mask_for_that_point)

    The mask is binary:
      - 1 = region of the clicked pixel's class
      - 0 = everything else
    """

    def __init__(self,
                 dataset_loc='Data/Oxford-IIIT-Pet-Augmented',
                 augmentations_per_datapoint=0,
                 split='validation',
                 cache=False,
                 gaussian_sigma=5,
                 ignore_uncertainty=True):
        """
        Args:
            dataset_loc (str): Path to the dataset folder.
            augmentations_per_datapoint (int): Number of augmentations. 
            split (str): 'train', 'validation', or 'test'.
            cache (bool): If True, load/save a cached version of (image, mask).
            gaussian_sigma (int): Std. dev. for Gaussian peak around the prompt point.
            ignore_uncertainty (bool): Whether to skip uncertain pixels when sampling a point.
        """
        if split not in ['train', 'validation', 'test']:
            raise ValueError(
                f"split must be one of: 'train', 'validation', 'test'. Got {split}."
            )

        if not isinstance(augmentations_per_datapoint, int) or augmentations_per_datapoint < 0:
            raise ValueError(f"augmentations_per_datapoint must be a nonnegative integer, got {augmentations_per_datapoint}.")

        # Attempt to load the dataset
        try:
            self.dataset = load_dataset(dataset_loc, split=split)
        except Exception as e:
            if "doesn't exist on the Hub or cannot be accessed" in str(e):
                print("Error: The dataset was not found locally. Downloading...")
                download_huggingface_dataset("mattidebeer/Oxford-IIIT-Pet-Augmented",
                                             dataset_loc,
                                             split=split)
                self.dataset = load_dataset(dataset_loc, split=split)
            else:
                print(f"An unexpected error occurred: {e}")
                raise e

        self.augmentations_per_datapoint = augmentations_per_datapoint + 1
        self.gaussian_sigma = gaussian_sigma
        self.ignore_uncertainty = ignore_uncertainty

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
                for datapoint in tqdm(self.dataset,
                                      desc=f"Caching {split} dataset:",
                                      leave=False,
                                      total=len(self.dataset)):
                    self.dataset_cache.append(self._deserialize_datapoint(datapoint))

                torch.save(self.dataset_cache, cache_file)
                self.dataset_cache = torch.load(cache_file, weights_only=True)

            del self.dataset
        else:
            self.dataset_cache = None

    def __len__(self):
        return self.dataset_length

    def _deserialize_datapoint(self, datapoint):
        """Convert the stored byte arrays to torch Tensors."""
        image = self._deserialize_numpy(datapoint['image'])
        mask = self._deserialize_numpy(datapoint['mask'], shape=(256, 256))

        # Convert image
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        # Convert mask to a torch tensor
        mask = torch.from_numpy(mask)

        return image, mask

    def _deserialize_numpy(self,
                           byte_data,
                           shape=(256, 256, 3),
                           dtype=np.uint8):
        """Helper to convert the stored byte data into a numpy array."""
        return copy.deepcopy(np.frombuffer(byte_data, dtype=dtype).reshape(shape))

    def __getitem__(self, idx):
        """
        Returns:
            image: Tensor [3, H, W]
            prompt_heatmap: Tensor [1, H, W]
            target_mask: Tensor [H, W] (binary) -> 1 if pixel belongs to the clicked point's class, else 0
        """
        # Figure out which datapoint
        image_index = idx // self.augmentations_per_datapoint

        if self.cache:
            image, mask = self.dataset_cache[image_index]
        else:
            datapoint = self.dataset[image_index]
            image, mask = self._deserialize_datapoint(datapoint)

        # mask has values:
        #   0 or 255 => background or uncertain
        #   38 => cat
        #   75 => dog
        # or, in your old logic, you might have cat=1, dog=2, etc.
        # Adjust the logic below to whatever your stored mask values actually are.

        # Convert to simpler labeling:
        #   1 => cat
        #   2 => dog
        #   0 => background
        #   255 => uncertainty
        cat_mask = (mask == 38).long()
        dog_mask = (mask == 75).long()
        # If the original background was 0, it remains 0
        # If 255 is uncertain -> handle or ignore below

        # Combine to a single label mask: 
        #    1 = cat, 2 = dog, 0 = background, 255 = uncertain
        label_mask = (1 * cat_mask) + (2 * dog_mask)
        # If there's 255, set that for uncertain
        label_mask[mask == 255] = 255

        # Now pick a random pixel for the prompt
        # Option A: skip uncertain (255) if ignore_uncertainty=True
        valid_pixels = (label_mask != 255) if self.ignore_uncertainty else torch.ones_like(label_mask, dtype=torch.bool)
        valid_indices = valid_pixels.nonzero(as_tuple=False)  # shape: [N, 2]
        # Choose 1 random pixel
        chosen_idx = valid_indices[torch.randint(len(valid_indices), size=(1,))]
        y, x = chosen_idx[0].item(), chosen_idx[1].item()

        # Figure out the label at that point
        chosen_label = label_mask[y, x].item()  # 0, 1, or 2

        # Build a binary mask: 1 if label_mask == chosen_label, else 0
        # For background => chosen_label=0 => all 0-labeled pixels become 1
        binary_mask = (label_mask == chosen_label).long()

        # Create a Gaussian heatmap channel of size [H, W]
        prompt_heatmap = self._create_gaussian_heatmap(
            height=label_mask.shape[0],
            width=label_mask.shape[1],
            center=(y, x),
            sigma=self.gaussian_sigma
        )

        return image, prompt_heatmap, binary_mask

    def _create_gaussian_heatmap(self, height, width, center, sigma=5):
        """
        Creates a 2D Gaussian heatmap with a peak at `center`.
        
        center: (y, x)
        sigma: Standard deviation of the Gaussian
        """
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float32),
            torch.arange(width, dtype=torch.float32),
            indexing='ij'
        )
        # center_y, center_x
        cy, cx = center
        # Calculate squared distance from center
        dist_sq = (x - cx)**2 + (y - cy)**2
        # Gaussian = exp(-dist / (2 * sigma^2))
        heatmap = torch.exp(-dist_sq / (2 * sigma**2))

        # scale to [0,1], though not strictly necessary
        heatmap /= heatmap.max()
        
        # unsqueeze to make [1, H, W]
        return heatmap.unsqueeze(0)

    
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

    
