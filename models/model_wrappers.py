from customDatasets.datasets import CustomImageDataset
from processing_blocks import DataAugmentor
from models.UNet import UNet
from torch.utils.data import DataLoader
from models.helperFunctions import get_next_run_folder, save_training_info, write_csv_header,log_loss_to_csv
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from models.losses import HybridLoss, IoULoss, PixelAccuracyLoss, DiceLoss
import sys
import torch.multiprocessing as mp
import os
import time

class TrainingWrapper:

    def __init__(self,
                 num_workers = 12,
                 batch_size = 100,
                 model = UNet(),
                 save_location = "saved-models/Test",
                 train_dataset_class = CustomImageDataset,
                 validation_dataset_class = CustomImageDataset,
                 train_dataset_args = {'split':'train','augmentations_per_datapoint' : 4},
                 validation_dataset_args = {'split':'validation','augmentations_per_datapoint' : 0},       
                 model_compilation_args = {'mode' : 'max-autotune'},    
                 criterion_class = HybridLoss, 
                 optimizer_class = optim.Adam,
                 optimizer_args = {'lr': 0.001, 'weight_decay' : 1e-4},
                 data_augmentor_class = DataAugmentor,
                 ):
    

        os.environ["OMP_NUM_THREADS"] = str(num_workers)
        os.environ["MKL_NUM_THREADS"] = str(num_workers)

        mp.set_start_method('spawn', force=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.cuda.empty_cache()

        self.save_location = get_next_run_folder(save_location)

        self.batch_size = batch_size

        augmentations_per_datapoint = train_dataset_args.get('augmentations_per_datapoint', 0)

        train_dataset = train_dataset_class(**train_dataset_args)
        validation_dataset = validation_dataset_class(**validation_dataset_args)

        self.train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        self.data_augmentor = data_augmentor_class(augmentations_per_datapoint)

        model = torch.compile(model, **model_compilation_args)
        self.model = model.to(self.device)  # Then move to GPU

        self.optimizer = optimizer_class(model.parameters(), **optimizer_args)
        self.criterion = criterion_class()

        num_params = sum(p.numel() for p in self.model.parameters())


        save_training_info(model,
                        self.optimizer,
                        self.criterion,
                        self.train_dataloader,
                        self.validation_dataloader,
                        save_location, 
                        extra_params = {'num_params' : num_params})


        write_csv_header(save_location)

    def train(self, num_epochs):

        gradscaler = torch.GradScaler('cuda')

        for epoch in tqdm( range(num_epochs), desc='Training', unit = 'Epoch', leave = False):
            
            self.model.train()
            running_loss = 0.0

            start_time = time.time()
            
            # Training loop
            for inputs, targets in tqdm(self.train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                inputs, targets = self.data_augmentor(inputs,targets)
                
                self.optimizer.zero_grad()  # Zero gradients from the previous step
                
                # Forward pass
                with torch.autocast('cuda'):
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)  # Compute the loss
                
                # Backward pass and optimization
                gradscaler.scale(loss).backward()
                gradscaler.step(self.optimizer)
                gradscaler.update()
                
                # Track the loss
                running_loss += loss.item()

            end_time = time.time()
            
            # Calculate time per datapoint
            time_per_epoch = end_time - start_time
            total_datapoints = len(self.train_dataloader) * self.batch_size
            rate = total_datapoints / time_per_epoch

            # Calculate average training loss and accuracy
            avg_train_loss = running_loss / len(self.train_dataloader)
            
            # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            running_val_loss = 0.0
            running_iou_loss = 0.0
            running_pixel_acc_loss = 0.0
            running_dice_loss = 0.0
        
        with torch.no_grad():  # No gradients needed during validation
            for inputs, targets in tqdm(self.validation_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} Validation",leave=False):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                with torch.autocast('cuda'):
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    # Calculate different losses
                    hybrid_loss = self.criterion(outputs, targets)
                    iou_loss = IoULoss()(outputs, targets)
                    pixel_acc_loss = PixelAccuracyLoss()(outputs, targets)
                    dice_loss = DiceLoss()(outputs, targets)
                
                # Track the losses
                running_val_loss += hybrid_loss.item()
                running_iou_loss += iou_loss.item()
                running_pixel_acc_loss += pixel_acc_loss.item()
                running_dice_loss += dice_loss.item()
        
        # Calculate average validation losses
        avg_val_loss = running_val_loss / len(self.validation_dataloader)
        avg_iou_loss = running_iou_loss / len(self.validation_dataloader)
        avg_pixel_acc_loss = running_pixel_acc_loss / len(self.validation_dataloader)
        avg_dice_loss = running_dice_loss / len(self.validation_dataloader)
        
        tqdm.write(f"Epoch: {epoch}")
        tqdm.write(f"Rate: {rate:.1f} datapoints/s")
        tqdm.write(f"Train Loss: {avg_train_loss:.4f}")
        tqdm.write(f"Validation Loss: {avg_val_loss:.4f}")
        tqdm.write(f"Val IoU: {avg_iou_loss:.4f}")
        tqdm.write(f"Val Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
        tqdm.write(f"Val Dice: {avg_dice_loss:.4f}")
        tqdm.write('\n')

        log_loss_to_csv(epoch,avg_train_loss,avg_val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss, self.save_location)
        
        # Delete previous model save if it exists (keep only latest)
        if epoch > 0:
            prev_model_path = f'{self.save_location}model_{epoch}.pth'
            if os.path.exists(prev_model_path):
                os.remove(prev_model_path)

        # Save current model
        torch.save(self.model.state_dict(), f'{self.save_location}model_{epoch+1}.pth')