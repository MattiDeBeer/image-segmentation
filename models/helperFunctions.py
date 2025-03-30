
import matplotlib.pyplot as plt
import os
import json
import csv

def save_training_info(model, optimizer, loss_fn, train_dataloader, val_dataloader, file_path, extra_params=None):
    # Save the model structure (layers) as a dictionary
    model_structure = {}
    for name, module in model.named_children():  # Iterate over each child module/layer
        model_structure[name] = str(module.__class__.__name__)  # Save layer type (e.g., Conv2d, Linear, etc.)
    
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
            writer.writerow(['Epoch', 'Train Loss', 'Validation Loss'])

def log_loss_to_csv(epoch, train_loss, val_loss, csv_path):
    csv_file = csv_path + 'loss.csv'
    with open(csv_file, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([epoch, train_loss, val_loss])

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

    plt.show()

