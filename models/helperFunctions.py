
import matplotlib.pyplot as plt
import os
import json
import csv
from matplotlib.patches import Patch
import torch
import numpy as np

def save_training_info(model, optimizer, loss_fn, train_dataloader, val_dataloader, file_path, extra_params=None):
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
    i = 1
    while True:
        folder_name = f"run-{i:03d}"
        folder_path = os.path.join(base_path, folder_name)
        
        # Check if the folder does not exist
        if not os.path.isdir(folder_path):
            # Create the folder
            os.makedirs(folder_path)
            return folder_path+'/'  # Return the path of the newly created folder

        i += 1  # If folder exists, increment the counter and check again

def write_csv_header(csv_path):
    csv_file = csv_path + 'loss.csv'
    if not os.path.exists(csv_file):
        with open(csv_file, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss', 
                           'Val Pixel Accuracy',
                           'Val Mean Dice',
                           'Val IoU'])

def log_loss_to_csv(epoch, train_loss, val_loss, val_pixel_acc, 
                    val_dice, val_iou, csv_path):
    csv_file = csv_path + 'loss.csv'
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss,
                        val_pixel_acc,
                        val_dice,
                        val_iou])

def show_datapoint(datapoint):

    image = datapoint[0]
    label = datapoint[1]
    
    plt.figure(figsize=(12,4))

    plt.subplot(141)
    plt.imshow(image.permute(1,2,0).cpu())
    plt.title('Original Image')

    plt.subplot(142)
    plt.imshow(label[0].cpu(), cmap='gray')
    plt.title('Cat Mask')

    plt.subplot(143)
    plt.imshow(label[1].cpu(), cmap='gray')
    plt.title('Dog Mask')

    if label.size()[0] >=3:
        plt.subplot(144)
        plt.imshow(label[2].cpu(), cmap='gray')
        plt.title('Background Mask')

    plt.show()

def show_datapoint_classes(datapoint):

    image = datapoint[0]
    label = datapoint[1]
    
    plt.figure(figsize=(12,4))

    plt.subplot(141)
    plt.imshow(image.permute(1,2,0).cpu())
    plt.title('Original Image')

    plt.subplot(142)
    plt.imshow((label.cpu() == 1).float(), cmap='gray')
    plt.title('Cat Mask')

    plt.subplot(143)
    plt.imshow((label.cpu() == 2).float(), cmap='gray')
    plt.title('Dog Mask')

    plt.subplot(144)
    plt.imshow((label.cpu() == 0).float(), cmap='gray')
    plt.title('Background mask')

    plt.show()

def show_prediciton(input,output):

    image = input[0]
    label = torch.nn.functional.softmax(output[0],dim=0)
    
    plt.figure(figsize=(12,4))

    plt.subplot(141)
    plt.imshow(image.permute(1,2,0).cpu())
    plt.title('Original Image')

    plt.subplot(142)
    plt.imshow(label[0].cpu(), cmap='gray')
    plt.title('Background Mask')

    plt.subplot(143)
    plt.imshow(label[1].cpu(), cmap='gray')
    plt.title('Cat Mask')

    plt.subplot(144)
    plt.imshow((label[2]).cpu(), cmap='gray')
    plt.title('Dog Mask')



    plt.show()

def show_detailed_prediction(input,output):
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))  # 3 rows, 4 columns

    image = input[0]
    label = output[0]

    # First row
    axes[0, 0].imshow(image.permute(1, 2, 0).cpu())
    axes[0, 0].set_title('Original Image')

    axes[0, 1].imshow(label[0].cpu(), cmap='gray')
    axes[0, 1].set_title('Cat Mask')

    axes[0, 2].imshow(label[1].cpu(), cmap='gray')
    axes[0, 2].set_title('Dog Mask')

    if label.size()[0] >= 3:
        axes[0, 3].imshow((label[2]).cpu(), cmap='gray')
        axes[0, 3].set_title('Background mask')
    else:
        axes[0, 3].imshow((label[0] + label[1]).cpu(), cmap='gray')
        axes[0, 3].set_title('Sum of Masks')

    # Second row (Modify these based on your additional data)
    axes[1, 0].imshow((label[0].cpu() > 0.25).float(), cmap='gray')
    axes[1, 0].set_title('Cat uncertianty > 0.25')

    axes[1, 1].imshow((label[0].cpu() > 0.5).float(), cmap='gray')
    axes[1, 1].set_title('Cat uncertianty > 0.5')

    axes[1, 2].imshow((label[0].cpu() > 0.75).float(), cmap='gray')
    axes[1, 2].set_title('Cat uncertianty > 0.75')

    axes[1, 3].imshow((label[0].cpu() > 0.9).float(), cmap='gray')
    axes[1, 3].set_title('Cat uncertianty > 0.95')

    # Third row (Modify these based on your additional data)
    axes[2, 0].imshow((label[1].cpu() > 0.25).float(), cmap='gray')
    axes[2, 0].set_title('Dog uncertianty > 0.25')

    axes[2, 1].imshow((label[1].cpu() > 0.5).float(), cmap='gray')
    axes[2, 1].set_title('Dog uncertianty > 0.5')

    axes[2, 2].imshow((label[1].cpu() > 0.75).float(), cmap='gray')
    axes[2, 2].set_title('Dog uncertianty > 0.75')

    axes[2, 3].imshow((label[1].cpu() > 0.9).float(), cmap='gray')
    axes[2, 3].set_title('Dog uncertianty > 0.9')

    # Remove axis labels
    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()


def plot_segmentation(image, prediction, class_colors=None, alpha=0.5):

    class_labels = ['Background', 'Cat', 'Dog']

    # Remove batch dimension
    image = image[0]
    prediction = prediction[0]

    # Convert image tensor to NumPy (HxWxC)
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy().transpose(1, 2, 0)

    # Convert prediction tensor to class index map
    if isinstance(prediction, torch.Tensor):
        prediction = torch.nn.functional.softmax(prediction, dim=0).argmax(dim=0).cpu().numpy()

    # Define default colors if none provided
    num_classes = max(prediction.max() + 1, len(class_labels))  # Ensure it covers all classes
    if class_colors is None:
        class_colors = {i: (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255)) for i in range(num_classes)}

    # Create overlay mask
    overlay = np.zeros_like(image)
    for cls, color in class_colors.items():
        overlay[prediction == cls] = np.array(color) / 255.0  # Normalize colors to [0,1]

    # Blend the overlay with the image
    blended = (1 - alpha) * image + alpha * overlay

    # Plot the result
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(blended)
    ax.axis("off")

    # Create legend
    legend_patches = [Patch(color=np.array(color) / 255, label=class_labels[cls]) for cls, color in class_colors.items() if cls < len(class_labels)]
    ax.legend(handles=legend_patches, loc='upper right')

    plt.show()

def show_segment_class_datapoint(datapoint):
    image = datapoint[0]
    segment_mask = datapoint[1][0]
    segment_class = datapoint[1][1]

    print(f"Segment Class: {segment_class}")

    plt.figure(figsize=(12,4))

    plt.subplot(141)
    plt.imshow(image.permute(1,2,0).cpu())
    plt.title('Original Image')

    plt.subplot(142)
    plt.imshow(segment_mask.cpu(), cmap='gray')
    plt.title(f"Mask for class: {segment_class}")

    plt.show()

def convert_prediciton(masks, classes):

    B, C, H, W = masks.size()

    det_masks = (masks > 0.5).float()
    det_classes = (classes > 0.5).float()

    bg_masks = 1 - det_masks

    cat_masks = det_masks * (1 - det_classes.unsqueeze(-1).unsqueeze(-1))
    dog_masks = det_masks * det_classes.unsqueeze(-1).unsqueeze(-1)

    out_masks = torch.cat([bg_masks,cat_masks,dog_masks], dim=1)

    return out_masks

def convert_targets(targets):
    masks, classes = targets

    B, C, H, W = masks.size()

    cat_masks = masks * (1 - classes.unsqueeze(-1).unsqueeze(-1))
    dog_masks = masks * classes.unsqueeze(-1).unsqueeze(-1)

    out_masks = cat_masks + 2 * dog_masks
    out_masks = out_masks.long()

    return out_masks




    
    




