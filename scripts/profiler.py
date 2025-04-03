from customDatasets.datasets import ClassImageDatasetGPU, CustomCollateFn, ClassImageDataset
from models.UNet import *
from models.CLIP_models import *
from torch.utils.data import DataLoader
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header,log_loss_to_csv, convert_prediciton
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from models.losses import HybridLoss, IoU, PixelAccuracy, Dice
import sys
import torch.multiprocessing as mp
import os
import time

core_num = 2

os.environ["OMP_NUM_THREADS"] = str(core_num)
os.environ["MKL_NUM_THREADS"] = str(core_num)

if __name__ == '__main__':

    mp.set_start_method('spawn', force=True)

    ###### Hyperperameters ###########
    model = ClipResSegmentationClassification()

    num_epochs = 200
    batch_size = 80
    num_workers = core_num - 2

    model_save_file = "saved-models/SegmentClassifierClipRes"

    augmentations_per_datapoint = 2

    ##############################

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Clear unused memory from the cache
    torch.cuda.empty_cache()

    save_location = get_next_run_folder(model_save_file)

    train_dataset = ClassImageDatasetGPU(split='train',augmentations_per_datapoint=4)
    validation_dataset = ClassImageDataset(split='validation',augmentations_per_datapoint=0)

    train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle=False, num_workers=num_workers, collate_fn=CustomCollateFn(augmentations_per_datapoint))
    validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=num_workers)

    #model = torch.compile(model, mode="max-autotune")
    model.to(device)  # Then move to GPU

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    num_params = sum(p.numel() for p in model.parameters())

    iou = IoU()
    pixel_acc = PixelAccuracy()
    dice = Dice()

    save_training_info(model,
                    optimizer,
                    criterion,
                    train_dataloader,
                    validation_dataloader,
                    save_location, 
                    extra_params = {'num_params' : num_params})


    write_csv_header(save_location)

    gradscaler = torch.GradScaler('cuda')

    with torch.profiler.profile(
    activities=[
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA
    ],
    on_trace_ready=torch.profiler.tensorboard_trace_handler('./profile-log'),
    record_shapes=True,
    profile_memory=True
    ) as prof:

        for epoch in tqdm( range(num_epochs), desc='Training', unit = 'Epoch', leave = False):
            
            model.train()
            running_loss = 0.0
            correct_train = 0
            total_train = 0

            start_time = time.time()
            
            # Training loop
            for inputs, targets in tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False):

                mask_targets, class_targets = targets[0].unsqueeze(1), targets[1]
                
                optimizer.zero_grad()  # Zero gradients from the previous step
                
                # Forward pass
                with torch.autocast('cuda'):
                    pred_masks, pred_labels = model(inputs)
                    loss = criterion(pred_masks, mask_targets)  + criterion(pred_labels, class_targets)
                
                # Backward pass and optimization
                gradscaler.scale(loss).backward()
                gradscaler.step(optimizer)
                gradscaler.update()
                
                # Track the loss
                running_loss += loss.item()

            end_time = time.time()
            
            # Calculate time per datapoint
            time_per_epoch = end_time - start_time
            total_datapoints = len(train_dataloader) * batch_size
            rate = total_datapoints / time_per_epoch

            # Calculate average training loss and accuracy
            avg_train_loss = running_loss / len(train_dataloader)

            print(f"Train loss")

            prof.step()
            
            