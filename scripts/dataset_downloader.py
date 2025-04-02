from datasets import load_dataset
import os

def download_huggingface_dataset(dataset_name, save_path, split="train"):
    """
    Downloads a dataset from Hugging Face and saves it to a specified directory.
    
    Args:
        dataset_name (str): The name of the dataset on Hugging Face.
        save_path (str): The directory where the dataset should be stored.
        split (str): The dataset split to download (e.g., 'train', 'validation', 'test').
    """

    print(f"fetching Dataset {dataset_name}")

    os.makedirs(save_path, exist_ok=True)

    dataset = load_dataset(dataset_name, split=split, cache_dir=save_path)

    print(f"Dataset {dataset_name} downloaded to {save_path}")

if __name__ == '__main__':
    dataset_name = "mattidebeer/Oxford-IIIT-Pet-Augmented"
    save_path = "Data/Oxford-IIIT-Pet-Augmented"

    download_huggingface_dataset(dataset_name, save_path, split="train")
    download_huggingface_dataset(dataset_name, save_path, split="validation")
    download_huggingface_dataset(dataset_name, save_path, split="train")