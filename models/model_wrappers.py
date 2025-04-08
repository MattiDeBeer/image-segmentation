from customDatasets.datasets import CustomImageDataset
from models.processing_blocks import DataAugmentor
from models.UNet import UNet
from torch.utils.data import DataLoader
from models.helperFunctions import *
import torch
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
from models.losses import HybridLoss, IoU, PixelAccuracy, Dice
import torch.multiprocessing as mp
import os
import time
from collections import OrderedDict

class TrainingWrapper:

    def __init__(self,
                 num_workers = 12,
                 batch_size = 100,
                 model_class = UNet,
                 model_arguments = {},
                 save_location = None,
                 train_dataset_class = CustomImageDataset,
                 validation_dataset_class = CustomImageDataset,
                 train_dataset_args = {'split':'train','augmentations_per_datapoint' : 4},
                 validation_dataset_args = {'split':'validation','augmentations_per_datapoint' : 0},       
                 model_compilation_args = {},    
                 criterion_class = HybridLoss, 
                 optimizer_class = optim.Adam,
                 optimizer_args = {'lr': 0.001, 'weight_decay' : 1e-4},
                 data_augmentor_class = DataAugmentor,
                 ):
        
    

        if num_workers > 0:
            os.environ["OMP_NUM_THREADS"] = str(num_workers)
            os.environ["MKL_NUM_THREADS"] = str(num_workers)

        mp.set_start_method('fork', force=True)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        torch.cuda.empty_cache()

        if save_location is None:
            save_location = "saved-models/" + model_class.__name__

        self.save_location = get_next_run_folder(save_location)

        self.batch_size = batch_size

        augmentations_per_datapoint = train_dataset_args.get('augmentations_per_datapoint', 0)

        train_dataset = train_dataset_class(**train_dataset_args)
        validation_dataset = validation_dataset_class(**validation_dataset_args)

        self.train_dataloader = DataLoader(train_dataset,batch_size = batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.validation_dataloader = DataLoader(validation_dataset,batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        data_augmentor = data_augmentor_class(augmentations_per_datapoint)
        self.data_augmentor = data_augmentor.to(self.device)

        model = model_class(**model_arguments)
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
                        self.save_location, 
                        extra_params = {'num_params' : num_params})

    def train(self, num_epochs):

        write_csv_header(self.save_location)

        gradscaler = torch.GradScaler(self.device.type)

        iou = IoU()
        dice = Dice()
        pixel_acc = PixelAccuracy()

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
                with torch.autocast(self.device.type):
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

                    with torch.autocast(self.device.type):
                        # Forward pass
                        outputs = self.model(inputs)
                        
                        # Calculate different losses
                        hybrid_loss = self.criterion(outputs, targets)
                        iou_loss = iou(outputs, targets)
                        pixel_acc_loss = pixel_acc(outputs, targets)
                        dice_loss = 2 * iou(outputs, targets) / (1 + iou(outputs, targets))
                    
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

            if torch.cuda.is_available():
                device = torch.cuda.current_device()
                tqdm.write(f"Current device: {torch.cuda.get_device_name(device)}")
                tqdm.write(f"Device index: {device}")
                tqdm.write(f"Current GPU memory usage: {torch.cuda.memory_allocated(device) / 1e9} GB")
                tqdm.write(f"Max GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1e9} GB")
                tqdm.write(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9} GB")
                tqdm.write('\n')

            log_loss_to_csv(epoch,avg_train_loss,avg_val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss, self.save_location)
            
            # Save current model
            torch.save(self.model.state_dict(), f'{self.save_location}model_{epoch+1}.pth')

class TestWrapper:

    def __init__(self,
                 model,
                 test_dataset_class = CustomImageDataset,
                 test_dataset_args = {'split':'test','augmentations_per_datapoint' : 0, 'cache' : False},
                 batch_size = 10,
                 model_load_location = None,
                 ):
        
        assert model_load_location is not None, "Model load location must be specified"

        self.test_dataloader = DataLoader(test_dataset_class(**test_dataset_args),batch_size = batch_size, shuffle=False, pin_memory=True)

        # Load state_dict
        state_dict = torch.load(model_load_location, map_location="cpu", weights_only=True)

        # Remove "_orig_mod." prefix
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")  # Remove prefix
            new_state_dict[new_key] = v

        # Load into model
        model.load_state_dict(new_state_dict)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = model.to(self.device)  # Then move to GPU

    
    def test(self):
            
        iou = IoU()
        dice = Dice()
        pixel_acc = PixelAccuracy()


       # Validation loop
        self.model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0
        running_iou_loss = 0.0
        running_pixel_acc_loss = 0.0
        running_dice_loss = 0.0
    
        with torch.no_grad():  # No gradients needed during validation
            for inputs, targets in tqdm(self.test_dataloader, desc=f"Evaluating model test performance",leave=False):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                
                # Forward pass
                outputs = self.model(inputs)
                
                iou_loss = iou(outputs, targets)
                pixel_acc_loss = pixel_acc(outputs, targets)
                dice_loss = 2 * iou(outputs, targets) / (1 + iou(outputs, targets))
                
                running_iou_loss += iou_loss.item()
                running_pixel_acc_loss += pixel_acc_loss.item()
                running_dice_loss += dice_loss.item()
        
        # Calculate average validation losses
        avg_iou_loss = running_iou_loss / len(self.test_dataloader)
        avg_pixel_acc_loss = running_pixel_acc_loss / len(self.test_dataloader)
        avg_dice_loss = running_dice_loss / len(self.test_dataloader)

        tqdm.write(f"IoU: {avg_iou_loss:.4f}")
        tqdm.write(f"Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
        tqdm.write(f"Dice: {avg_dice_loss:.4f}")
        tqdm.write('\n')


    
    def test_augmentation(self,augmentation, param_val):
            
            iou = IoU()
            dice = Dice()
            pixel_acc = PixelAccuracy()


        # Validation loop
            self.model.eval()  # Set the model to evaluation mode
            running_iou_loss = 0.0
            running_pixel_acc_loss = 0.0
            running_dice_loss = 0.0

            augmentation.to(self.device)
        
            with torch.no_grad():  # No gradients needed during validation
                for inputs, targets in tqdm(self.test_dataloader, desc=f"Evaluating performance metric {augmentation.__class__.__name__} for paramater value {param_val}", leave=False):
                    inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                    # Apply the augmentation
                    inputs, targets = augmentation(inputs, targets)
                    
                    # Forward pass
                    outputs = self.model(inputs)
                    
                    iou_loss = iou(outputs, targets)
                    pixel_acc_loss = pixel_acc(outputs, targets)
                    dice_loss = 2 * iou(outputs, targets) / (1 + iou(outputs, targets))
                    
                    running_iou_loss += iou_loss.item()
                    running_pixel_acc_loss += pixel_acc_loss.item()
                    running_dice_loss += dice_loss.item()
            
            # Calculate average validation losses
            avg_iou_loss = running_iou_loss / len(self.test_dataloader)
            avg_pixel_acc_loss = running_pixel_acc_loss / len(self.test_dataloader)
            avg_dice_loss = dice_loss / len(self.test_dataloader)

            return avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss
        
    def log_results_to_csv(results, filename='augmentation_results.csv'):
        """
        Logs the results of the augmentation tests into a CSV file.
        
        Arguments:
        results (list of tuples): Each tuple contains (augmentation_name, parameter_value, IoU, Pixel Accuracy, Dice)
        filename (str): The name of the output CSV file
        """
        fieldnames = ['Augmentation', 'Parameter Value', 'Avg IoU Loss', 'Avg Pixel Accuracy Loss', 'Avg Dice Loss']
        
        # Open the file in append mode (or create it if it doesn't exist)
        with open(filename, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            
            # Write headers only if the file is empty
            if file.tell() == 0:
                writer.writeheader()
            
            # Write each row to the CSV file
            for result in results:
                writer.writerow({
                    'Augmentation': result[0],
                    'Parameter Value': result[1],
                    'Avg IoU Loss': result[2],
                    'Avg Pixel Accuracy Loss': result[3],
                    'Avg Dice Loss': result[4]
                })


    def test_gaussian_pixel_noise(self, parameter_values):
        """
        Tests Gaussian Pixel Noise augmentation with different parameter values.
        
        Arguments:
        parameter_values (list): List of standard deviation values for the Gaussian noise
        """
        results = []
        for std in parameter_values:
            augmentation = GaussianPixelNoise(std)
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, std)
            results.append(('Gaussian Pixel Noise', std, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV
        self.log_results_to_csv(results, filename='gaussian_pixel_noise_results.csv')


    def test_repeated_blur(self, parameter_values):
        """
        Tests Repeated Blur augmentation with different parameter values.
        
        Arguments:
        parameter_values (list): List of values for the number of blur repetitions
        """
        results = []
        for times in parameter_values:
            augmentation = RepeatedBlur(times)
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, times)
            results.append(('Repeated Blur', times, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV
        self.log_results_to_csv(results, filename='repeated_blur_results.csv')


    def test_contrast_change(self, parameter_values):
        """
        Tests Contrast Change augmentation with different parameter values.
        
        Arguments:
        parameter_values (list): List of contrast factor values
        """
        results = []
        for factor in parameter_values:
            augmentation = ContrastChange(factor)
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, factor)
            results.append(('Contrast Change', factor, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV
        self.log_results_to_csv(results, filename='contrast_change_results.csv')


    def test_brightness_change(self, parameter_values):
        """
        Tests Brightness Change augmentation with different parameter values.
        
        Arguments:
        parameter_values (list): List of brightness offset values
        """
        results = []
        for offset in parameter_values:
            augmentation = BrightnessChange(offset)
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, offset)
            results.append(('Brightness Change', offset, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV
        self.log_results_to_csv(results,filename='brightness_change_results.csv')


    def test_occlusion(self, parameter_values):
        """
        Tests Occlusion augmentation with different parameter values.
        
        Arguments:
        parameter_values (list): List of occlusion square sizes
        """
        results = []
        for size in parameter_values:
            augmentation = Occlusion(size)  
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, size)
            results.append(('Occlusion', size, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV
        self.log_results_to_csv(results, filename='occlusion_results.csv')


    def test_salt_and_pepper(self, parameter_values):
        """
        Tests Salt and Pepper Noise augmentation with different parameter values.
        
        Arguments:
        parameter_values (list): List of noise strength values
        """
        results = []
        for amount in parameter_values:
            augmentation = SaltAndPepper(amount)
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, amount)
            results.append(('Salt and Pepper', amount, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV
        self.log_results_to_csv(results)


    def test_robustness(self):
        self.test_gaussian_pixel_noise([0.01, 0.05, 0.1])

class DistributedTrainingWrapper:

    def __init__(self,
                 rank,
                 model,
                 train_dataloader,
                 validation_dataloader,
                 data_augmentor,
                 batch_size = 100,
                 save_location = None,   
                 criterion_class = HybridLoss, 
                 optimizer_class = optim.Adam,
                 optimizer_args = {'lr': 0.001, 'weight_decay' : 2e-4},
                 ):
    

        if rank == 0:
            if save_location is None:
                save_location = "saved-models/" + model.module.__class__.__name__

            self.save_location = get_next_run_folder(save_location)

        self.batch_size = batch_size

        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.data_augmentor = data_augmentor
        self.rank = rank

        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_args)
        self.criterion = criterion_class()

        num_params = sum(p.numel() for p in self.model.parameters())

        if self.rank == 0:
            save_training_info(model,
                            self.optimizer,
                            self.criterion,
                            self.train_dataloader,
                            self.validation_dataloader,
                            self.save_location, 
                            extra_params = {'num_params' : num_params})



    def train(self, num_epochs):

        if self.rank == 0:
            write_csv_header(self.save_location)

        gradscaler = torch.GradScaler('cuda')

        iou = IoU()
        dice = Dice()
        pixel_acc = PixelAccuracy()

        if self.rank == 0:
            disable = False
        else:
            disable = True

        for epoch in tqdm( range(num_epochs), desc='Training', unit = 'Epoch', leave = False, disable=disable ):

            self.train_dataloader.sampler.set_epoch(epoch)
            
            self.model.train()
            running_loss = 0.0

            start_time = time.time()
            
            # Training loop
            for inputs, targets in tqdm(self.train_dataloader, desc=f"[Rank:{self.rank}] Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False):
                inputs, targets = inputs.to(self.rank, non_blocking=True), targets.to(self.rank, non_blocking=True)

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

            torch.distributed.barrier()
            
            with torch.no_grad():  # No gradients needed during validation
                for inputs, targets in tqdm(self.validation_dataloader, desc=f"[Rank:{self.rank}] Epoch {epoch+1}/{num_epochs} Validation",leave=False):
                    inputs, targets = inputs.to(self.rank, non_blocking=True), targets.to(self.rank, non_blocking=True)

                    with torch.autocast('cuda'):
                        # Forward pass
                        outputs = self.model(inputs)
                        
                        # Calculate different losses
                        hybrid_loss = self.criterion(outputs, targets)
                        iou_loss = iou(outputs, targets)
                        pixel_acc_loss = pixel_acc(outputs, targets)
                        dice_loss = 2 * iou(outputs, targets) / (1 + iou(outputs, targets))
                    
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
            
            tqdm.write(f"[Rank:{self.rank}] Epoch: {epoch}")
            tqdm.write(f"[Rank:{self.rank}] Rate: {rate:.1f} datapoints/s")
            tqdm.write(f"[Rank:{self.rank}] Train Loss: {avg_train_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Validation Loss: {avg_val_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Val IoU: {avg_iou_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Val Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Val Dice: {avg_dice_loss:.4f}")
            tqdm.write('\n')

            if self.rank == 0:

                log_loss_to_csv(epoch,avg_train_loss,avg_val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss, self.save_location)

                # Save current model
                torch.save(self.model.state_dict(), f'{self.save_location}model_{epoch+1}.pth')

            torch.distributed.barrier()
            