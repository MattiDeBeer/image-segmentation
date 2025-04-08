
import matplotlib.pyplot as plt
import os
import json
import csv
from matplotlib.patches import Patch
import torch
import numpy as np

def save_training_info(model, optimizer, loss_fn, train_dataloader, val_dataloader, file_path, extra_params=None):
    """
    Saves the training configuration and model structure to a JSON file.
    This function extracts and saves the structure of the model, optimizer, loss function, 
    and dataloaders, along with any additional parameters provided, into a JSON file for 
    reproducibility and debugging purposes.
    Args:
        model (torch.nn.Module): The PyTorch model whose structure is to be saved.
        optimizer (torch.optim.Optimizer): The optimizer used for training, including its parameter groups.
        loss_fn (torch.nn.Module): The loss function used during training.
        train_dataloader (torch.utils.data.DataLoader): The DataLoader for the training dataset.
        val_dataloader (torch.utils.data.DataLoader): The DataLoader for the validation dataset.
        file_path (str): The directory path where the JSON file will be saved. The file will be named 'model_settings.json'.
        extra_params (dict, optional): Additional parameters to include in the saved JSON file. Defaults to None.
    Returns:
        None: The function saves the configuration to a JSON file and does not return any value.
    Notes:
        - The model structure is saved as a dictionary containing layer types and their hyperparameters.
        - The optimizer structure includes learning rate, momentum, and weight decay for each parameter group.
        - The loss function is saved as its class name.
        - The dataloader information includes batch size and dataset size.
        - The JSON file is saved with an indentation of 4 for readability.
    Example:
        save_training_info(
            model=my_model,
            optimizer=my_optimizer,
            loss_fn=my_loss_fn,
            train_dataloader=train_loader,
            val_dataloader=val_loader,
            file_path="/path/to/save/",
            extra_params={"epochs": 50, "learning_rate_schedule": "step"}
        )
    """
    # Save the model structure (layers) as a dictionary
    model_structure = {}
    for name, module in model.named_modules():  # Use named_modules instead of named_children to get all layers
        layer_info = {
            'type': str(module.__class__.__name__)
        }
        
        # Add specific hyperparameters based on layer type
        if hasattr(module, 'in_channels'):
            layer_info['in_channels'] = module.in_channels
        if hasattr(module, 'out_channels'):
            layer_info['out_channels'] = module.out_channels
        if hasattr(module, 'kernel_size'):
            layer_info['kernel_size'] = module.kernel_size if isinstance(module.kernel_size, int) else list(module.kernel_size)
        if hasattr(module, 'num_heads'):
            layer_info['num_heads'] = module.num_heads
        if hasattr(module, 'dropout'):
            layer_info['dropout'] = module.dropout
        if hasattr(module, 'padding'):
            layer_info['padding'] = module.padding if isinstance(module.padding, int) else list(module.padding)
        # Linear layer parameters
        if hasattr(module, 'in_features'):
            layer_info['in_features'] = module.in_features
        if hasattr(module, 'out_features'):
            layer_info['out_features'] = module.out_features
        # Multi-head attention parameters
        if hasattr(module, 'embed_dim'):
            layer_info['embed_dim'] = module.embed_dim
        if hasattr(module, 'head_dim'):
            layer_info['head_dim'] = module.head_dim
        if hasattr(module, 'kdim'):
            layer_info['key_dim'] = module.kdim
        if hasattr(module, 'vdim'):
            layer_info['value_dim'] = module.vdim
        
        model_structure[name] = layer_info
    
    # Save the optimizer structure (param_groups, etc.)
    optimizer_structure = {
        'param_groups': []
    }

    for param_group in optimizer.param_groups:
        # For each param group, we only save the config (lr, momentum, etc.)
        param_group_info = {
            'lr': param_group['lr'],
            'momentum': param_group.get('momentum', None),  # Some optimizers may not have momentum
            'weight_decay': param_group.get('weight_decay', None)
        }
        optimizer_structure['param_groups'].append(param_group_info)
    
    # Save the loss function type (human-readable)
    loss_fn_name = str(loss_fn.__class__.__name__)

    # Save dataloader information (batch size, dataset length, etc.)
    train_dataloader_info = {
        'batch_size': train_dataloader.batch_size,
        'dataset_size': len(train_dataloader.dataset)  # Assuming the dataset is accessible via 'dataset'
    }

    val_dataloader_info = {
        'batch_size': val_dataloader.batch_size,
        'dataset_size': len(val_dataloader.dataset)
    }

    # Create a checkpoint dictionary
    checkpoint = {
        'model_structure': model_structure,
        'optimizer_structure': optimizer_structure,
        'loss_function': loss_fn_name,
        'train_dataloader': train_dataloader_info,
        'val_dataloader': val_dataloader_info
    }

    # If extra_params is provided, add them to the checkpoint
    if extra_params:
        checkpoint['extra_params'] = extra_params
    
    # Write the checkpoint to a JSON file
    with open(file_path+'model_settings.json', 'w') as f:
        json.dump(checkpoint, f, indent=4)

    print(f"Model settings saved to {file_path}model_settings.json")

def get_next_run_folder(base_path):
    """
    Creates and returns the path to the next sequentially numbered folder 
    (e.g., 'run-001', 'run-002', etc.) within the specified base directory. 
    If the folder already exists, increments the counter until a non-existing 
    folder is found and created.
    Args:
        base_path (str): The base directory where the new folder will be created.
    Returns:
        str: The path to the newly created folder, ending with a '/'.
    """

    i = 1
    while True:
        # Generate a folder name with a zero-padded counter
        folder_name = f"run-{i:03d}"
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder does not exist
        if not os.path.isdir(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            # Return the path of the newly created folder
            return folder_path + '/'

        # If folder exists, increment the counter and check again
        i += 1

def write_csv_header(csv_path):
    """
    Creates a CSV file with a header row for logging training and validation metrics 
    if the file does not already exist.
    Args:
        csv_path (str): The directory path where the CSV file will be created.
    """
    # Define the path to the CSV file for logging loss
    csv_file = csv_path + 'loss.csv'
    
    # Check if the CSV file does not exist
    if not os.path.exists(csv_file):
        # Create and open the CSV file in write mode
        with open(csv_file, mode='w', newline='') as file:
            # Initialize a CSV writer
            writer = csv.writer(file)
            # Write the header row with column names
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 
                           'Val Pixel Accuracy',
                           'Val Mean Dice',
                           'Val IoU'])

def log_loss_to_csv(epoch, train_loss, val_loss, val_pixel_acc, val_dice, val_iou, csv_path):
    """
    Logs training and validation metrics to a CSV file.

    This function appends a row of metrics, including epoch number, training loss,
    validation loss, validation pixel accuracy, validation Dice coefficient, and 
    validation Intersection over Union (IoU), to a CSV file.

    Args:
        epoch (int): The current epoch number.
        train_loss (float): The training loss for the current epoch.
        val_loss (float): The validation loss for the current epoch.
        val_pixel_acc (float): The pixel accuracy on the validation set.
        val_dice (float): The Dice coefficient on the validation set.
        val_iou (float): The Intersection over Union (IoU) on the validation set.
        csv_path (str): The directory path where the CSV file is located. The file
                        will be named 'loss.csv' and appended to this path.

    Returns:
        None
    """  
    # Define the path to the CSV file
    csv_file = csv_path + 'loss.csv'
    # Open the CSV file in append mode
    with open(csv_file, mode='a', newline='') as file:
        # Initialize a CSV writer
        writer = csv.writer(file)
        # Write a row with the provided metrics
        writer.writerow([epoch, train_loss, val_loss,
                        val_pixel_acc,
                        val_dice,
                        val_iou])
        
def plot_segmentations(images, predictions, class_colors=None, alpha=0.5, n_cols=4):
    """
    Visualize image segmentations by overlaying predictions on input images.
    Args:
        images (torch.Tensor or np.ndarray): Batch of input images, shape (N, C, H, W) or (N, H, W, C).
        predictions (torch.Tensor or np.ndarray): Batch of segmentation predictions, shape (N, H, W).
        class_colors (dict, optional): Mapping of class indices to RGB colors. Defaults to predefined colors.
        alpha (float, optional): Transparency level for overlay blending. Defaults to 0.5.
        n_cols (int, optional): Number of columns in the plot grid. Defaults to 4.
    Returns:
        None: Displays the segmentation visualization.
    """
    class_labels = ['Background', 'Cat', 'Dog']

    # Convert images and predictions to NumPy arrays
    if isinstance(images, torch.Tensor):
        images = images.cpu().numpy().transpose(0, 2, 3, 1)  # Convert to (N, H, W, C)
    if isinstance(predictions, torch.Tensor):
        predictions = torch.nn.functional.softmax(predictions, dim=1).argmax(dim=1).cpu().numpy()  # Convert to (N, H, W)

    # Define default colors if none provided
    if class_colors is None:
        class_colors = {
            0: (0, 0, 0),       # Background (transparent)
            1: (0, 0, 255),     # Cat (blue)
            2: (0, 255, 0)      # Dog (green)
        }

    # Determine the number of rows for the plot
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols

    # Create the figure
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    axes = axes.flatten()

    for i, (image, prediction) in enumerate(zip(images, predictions)):
        # Create overlay mask
        overlay = np.zeros_like(image)
        for cls, color in class_colors.items():
            overlay[prediction == cls] = np.array(color) / 255.0  # Normalize colors to [0,1]

        # Blend the overlay with the image
        blended = (1 - alpha) * image + alpha * overlay

        # Plot the result
        ax = axes[i]
        ax.imshow(blended)
        ax.axis("off")


    # Hide any unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.show()





    
    




