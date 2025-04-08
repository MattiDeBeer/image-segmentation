from customDatasets.datasets import CustomImageDataset
from models.processing_blocks import *
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
    
    """
    TrainingWrapper is a utility class designed to streamline the training and validation process for deep learning models. 
    It provides a structured framework for initializing datasets, data loaders, models, optimizers, and loss functions, 
    while also supporting data augmentation and mixed precision training. The class handles device management, 
    model saving, and logging of training metrics, making it easier to train and evaluate models for tasks such as 
    image segmentation.
        device (torch.device): The device (CPU or GPU) used for computation.
        save_location (str): Directory where the model and training information are saved.
        data_augmentor (DataAugmentor): Instance of the data augmentor, moved to the appropriate device.
        model (torch.nn.Module): The initialized and compiled model, moved to the appropriate device.
    """

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
        """
        Initializes the model wrapper with specified configurations for training and validation.
        Parameters:
            num_workers (int): Number of worker threads for data loading. Default is 12.
            batch_size (int): Batch size for training and validation. Default is 100.
            model_class (type): Class of the model to be used. Default is UNet.
            model_arguments (dict): Arguments to initialize the model. Default is an empty dictionary.
            save_location (str): Directory to save the model. If None, a default location is used. Default is None.
            train_dataset_class (type): Class for the training dataset. Default is CustomImageDataset.
            validation_dataset_class (type): Class for the validation dataset. Default is CustomImageDataset.
            train_dataset_args (dict): Arguments for initializing the training dataset. Default is {'split': 'train', 'augmentations_per_datapoint': 4}.
            validation_dataset_args (dict): Arguments for initializing the validation dataset. Default is {'split': 'validation', 'augmentations_per_datapoint': 0}.
            model_compilation_args (dict): Arguments for compiling the model (if supported). Default is an empty dictionary.
            criterion_class (type): Class for the loss function. Default is HybridLoss.
            optimizer_class (type): Class for the optimizer. Default is optim.Adam.
            optimizer_args (dict): Arguments for initializing the optimizer. Default is {'lr': 0.001, 'weight_decay': 1e-4}.
            data_augmentor_class (type): Class for the data augmentor. Default is DataAugmentor.
        Attributes:
            device (torch.device): Device to be used for computation (GPU if available, otherwise CPU).
            save_location (str): Directory where the model and training information will be saved.
            batch_size (int): Batch size for training and validation.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            validation_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            data_augmentor (DataAugmentor): Data augmentor instance moved to the appropriate device.
            model (torch.nn.Module): Initialized and compiled model moved to the appropriate device.
            optimizer (torch.optim.Optimizer): Optimizer initialized with model parameters.
            criterion (torch.nn.Module): Loss function instance.
        """
        
        # Set the number of threads for OpenMP and MKL if num_workers is greater than 0
        if num_workers > 0:
            os.environ["OMP_NUM_THREADS"] = str(num_workers)
            os.environ["MKL_NUM_THREADS"] = str(num_workers)

        # Set the multiprocessing start method to 'fork' (forcefully)
        mp.set_start_method('fork', force=True)

        # Determine the device to use (GPU if available, otherwise CPU)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Clear the GPU memory cache
        torch.cuda.empty_cache()

        # Set the save location for the model if not provided
        if save_location is None:
            save_location = "saved-models/" + model_class.__name__

        # Get the next available folder for saving the model
        self.save_location = get_next_run_folder(save_location)

        # Store the batch size
        self.batch_size = batch_size

        # Get the number of augmentations per datapoint from the training dataset arguments
        augmentations_per_datapoint = train_dataset_args.get('augmentations_per_datapoint', 0)

        # Initialize the training and validation datasets
        train_dataset = train_dataset_class(**train_dataset_args)
        validation_dataset = validation_dataset_class(**validation_dataset_args)

        # Create DataLoaders for training and validation datasets
        self.train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        self.validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

        # Initialize the data augmentor and move it to the appropriate device
        data_augmentor = data_augmentor_class(augmentations_per_datapoint)
        self.data_augmentor = data_augmentor.to(self.device)

        # Initialize the model with the specified arguments
        model = model_class(**model_arguments)

        # Compile the model for optimization (if supported by the backend)
        model = torch.compile(model, **model_compilation_args)

        # Move the model to the appropriate device
        self.model = model.to(self.device)

        # Initialize the optimizer with the model parameters and specified arguments
        self.optimizer = optimizer_class(model.parameters(), **optimizer_args)

        # Initialize the loss criterion
        self.criterion = criterion_class()

        # Calculate the total number of parameters in the model
        num_params = sum(p.numel() for p in self.model.parameters())


        save_training_info(model,
                        self.optimizer,
                        self.criterion,
                        self.train_dataloader,
                        self.validation_dataloader,
                        self.save_location, 
                        extra_params = {'num_params' : num_params})

    def train(self, num_epochs):

        # Write the header for the CSV file to store training and validation metrics
        write_csv_header(self.save_location)

        # Initialize a gradient scaler for mixed precision training
        gradscaler = torch.GradScaler(self.device.type)

        # Initialize evaluation metrics for IoU, Dice coefficient, and Pixel Accuracy
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
                
                # Forward pass with mixed precision inference
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

            #verbose the validation results
            tqdm.write(f"Epoch: {epoch}")
            tqdm.write(f"Rate: {rate:.1f} datapoints/s")
            tqdm.write(f"Train Loss: {avg_train_loss:.4f}")
            tqdm.write(f"Validation Loss: {avg_val_loss:.4f}")
            tqdm.write(f"Val IoU: {avg_iou_loss:.4f}")
            tqdm.write(f"Val Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
            tqdm.write(f"Val Dice: {avg_dice_loss:.4f}")
            tqdm.write('\n')

            # If CUDA is available, log GPU details and memory usage
            if torch.cuda.is_available():
                device = torch.cuda.current_device()  # Get the current CUDA device
                tqdm.write(f"Current device: {torch.cuda.get_device_name(device)}")  # Log the device name
                tqdm.write(f"Device index: {device}")  # Log the device index
                tqdm.write(f"Current GPU memory usage: {torch.cuda.memory_allocated(device) / 1e9} GB")  # Log current memory usage
                tqdm.write(f"Max GPU memory usage: {torch.cuda.max_memory_allocated(device) / 1e9} GB")  # Log max memory usage
                tqdm.write(f"Total GPU memory: {torch.cuda.get_device_properties(device).total_memory / 1e9} GB")  # Log total memory
                tqdm.write('\n')  # Add a newline for better readability

            # Log the training and validation losses to a CSV file
            log_loss_to_csv(epoch, avg_train_loss, avg_val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss, self.save_location)
            
            # Save the current model's state_dict to a file
            torch.save(self.model.state_dict(), f'{self.save_location}model_{epoch+1}.pth')

class TestWrapper:
    """
    TestWrapper is a utility class designed to facilitate the evaluation of image segmentation models. 
    It provides methods for testing model performance on a test dataset, applying various augmentations 
    to assess robustness, and logging results for further analysis. The class supports evaluation metrics 
    such as Intersection over Union (IoU), Dice coefficient, and Pixel Accuracy, and includes functionality 
    for visualizing predictions.
    Key Features:
    - Initialize with a PyTorch model and test dataset configuration.
    - Evaluate model performance on the test dataset using standard metrics.
    - Test the impact of various augmentations (e.g., Gaussian noise, blurring, contrast changes, brightness changes, occlusion, and salt-and-pepper noise).
    - Log augmentation results to CSV files for analysis.
    - Visualize model predictions for selected test dataset samples.
    - test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
    - test_dataset (torch.utils.data.Dataset): The test dataset instance.
    - device (torch.device): The device (CPU or GPU) on which the model will run.
    - model (torch.nn.Module): The model loaded with the specified state_dict and moved to the appropriate device.
    Methods:
    - test(): Evaluate the model's performance on the test dataset.
    - test_augmentation(augmentation, param_val): Evaluate the model's performance with a specific augmentation.
    - log_results_to_csv(results, filename): Log augmentation results to a CSV file.
    - test_gaussian_pixel_noise(parameter_values): Test the effect of Gaussian pixel noise on model performance.
    - test_repeated_blur(parameter_values): Test the effect of repeated blurring on model performance.
    - test_contrast_change(parameter_values, increase): Test the effect of contrast changes on model performance.
    - test_brightness_change(parameter_values, increase): Test the effect of brightness changes on model performance.
    - test_occlusion(parameter_values): Test the effect of occlusion on model performance.
    - test_salt_and_pepper(parameter_values): Test the effect of salt-and-pepper noise on model performance.
    - test_robustness(): Evaluate the model's robustness across multiple augmentations.
    - plot_predictions(indices): Visualize model predictions for selected test dataset samples.
    """

    def __init__(self,
                 model,
                 test_dataset_class = CustomImageDataset,
                 test_dataset_args = {'split':'test','augmentations_per_datapoint' : 0, 'cache' : False},
                 batch_size = 10,
                 model_load_location = None,
                 ):
        """
        Initializes the model wrapper with the specified model, test dataset, and configurations.
        Args:
            model (torch.nn.Module): The PyTorch model to be used for inference.
            test_dataset_class (type, optional): The dataset class to be used for creating the test dataset. 
                Defaults to CustomImageDataset.
            test_dataset_args (dict, optional): Arguments to be passed to the test dataset class. 
                Defaults to {'split': 'test', 'augmentations_per_datapoint': 0, 'cache': False}.
            batch_size (int, optional): The batch size for the test DataLoader. Defaults to 10.
            model_load_location (str): The file path to the model's state_dict to be loaded.
        Raises:
            AssertionError: If `model_load_location` is not provided.
        Attributes:
            test_dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
            test_dataset (torch.utils.data.Dataset): The test dataset instance.
            device (torch.device): The device (CPU or GPU) on which the model will run.
            model (torch.nn.Module): The model loaded with the specified state_dict and moved to the appropriate device.
        """
                 
        # Ensure that the model load location is provided
        assert model_load_location is not None, "Model load location must be specified"

        # Initialize the test dataloader with the specified test dataset and arguments
        self.test_dataloader = DataLoader(
            test_dataset_class(**test_dataset_args), 
            batch_size=batch_size, 
            shuffle=False, 
            pin_memory=True
        )

        # Initialize the test dataset
        self.test_dataset = test_dataset_class(**test_dataset_args)

        # Load the model's state_dict from the specified location
        state_dict = torch.load(model_load_location, map_location="cpu", weights_only=True)

        # Remove the "_orig_mod." prefix from the state_dict keys (if present)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_key = k.replace("_orig_mod.", "")  # Remove prefix
            new_state_dict[new_key] = v

        # Load the modified state_dict into the model
        model.load_state_dict(new_state_dict)

        # Set the device to GPU if available, otherwise use CPU
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Move the model to the appropriate device
        self.model = model.to(self.device)

    
    def test(self):
        """
        Evaluate the model's performance on the test dataset using various metrics.
        This method performs a forward pass through the model for each batch in the 
        test dataloader and computes evaluation metrics including Intersection over Union (IoU), 
        Dice coefficient, and Pixel Accuracy. The results are averaged over the entire test dataset.
        Metrics:
            - IoU: Measures the overlap between predicted and ground truth segmentation.
            - Dice: Measures the similarity between predicted and ground truth segmentation.
            - Pixel Accuracy: Measures the percentage of correctly classified pixels.
        Steps:
            1. Set the model to evaluation mode.
            2. Iterate through the test dataloader without computing gradients.
            3. Compute the IoU, Dice, and Pixel Accuracy for each batch.
            4. Accumulate the metrics and compute their averages.
            5. Log the averaged metrics to the console.
        Note:
            - This method assumes that the model, test dataloader, and device are already initialized.
            - The IoU, Dice, and PixelAccuracy classes/functions should be implemented and compatible 
              with the model's outputs and targets.
        Outputs:
            Logs the average IoU, Pixel Accuracy, and Dice metrics to the console.
        """
        
        # Initialize evaluation metrics
        iou = IoU()  # Intersection over Union metric
        dice = Dice()  # Dice coefficient metric
        pixel_acc = PixelAccuracy()  # Pixel accuracy metric

        # Validation loop
        self.model.eval()  # Set the model to evaluation mode
        running_val_loss = 0.0  # Accumulator for validation loss
        running_iou_loss = 0.0  # Accumulator for IoU loss
        running_pixel_acc_loss = 0.0  # Accumulator for pixel accuracy loss
        running_dice_loss = 0.0  # Accumulator for Dice loss
    
        with torch.no_grad():  # No gradients needed during validation
            for inputs, targets in tqdm(self.test_dataloader, desc=f"Evaluating model test performance",leave=False):
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                
                # Perform a forward pass through the model
                outputs = self.model(inputs)
                
                # Compute evaluation metrics
                iou_loss = iou(outputs, targets)  # Calculate Intersection over Union (IoU) loss
                pixel_acc_loss = pixel_acc(outputs, targets)  # Calculate pixel accuracy loss
                dice_loss = 2 * iou(outputs, targets) / (1 + iou(outputs, targets))  # Calculate Dice loss
                
                # Accumulate the metrics for averaging later
                running_iou_loss += iou_loss.item()
                running_pixel_acc_loss += pixel_acc_loss.item()
                running_dice_loss += dice_loss.item()
        
        # Calculate average validation losses
        avg_iou_loss = running_iou_loss / len(self.test_dataloader)  # Compute the average IoU loss over the test dataset
        avg_pixel_acc_loss = running_pixel_acc_loss / len(self.test_dataloader)  # Compute the average pixel accuracy loss
        avg_dice_loss = running_dice_loss / len(self.test_dataloader)  # Compute the average Dice loss

        # Log the computed metrics to the console
        tqdm.write(f"IoU: {avg_iou_loss:.4f}")  # Log the average IoU loss
        tqdm.write(f"Pixel Accuracy: {avg_pixel_acc_loss:.4f}")  # Log the average pixel accuracy loss
        tqdm.write(f"Dice: {avg_dice_loss:.4f}")  # Log the average Dice loss
        tqdm.write('\n')  # Add a newline for better readability


    
    def test_augmentation(self,augmentation, param_val):
        """
        Evaluates the performance of the model using a specified augmentation and parameter value.
        This method applies a given augmentation to the test dataset, performs inference using the model, 
        and computes evaluation metrics including Intersection over Union (IoU), Pixel Accuracy, and Dice coefficient.
        Args:
            augmentation (torch.nn.Module): The augmentation to be applied to the input data.
            param_val (float): The parameter value associated with the augmentation being tested.
        Returns:
            tuple: A tuple containing the average IoU, Pixel Accuracy, and Dice coefficient over the test dataset.
        Notes:
            - The model is set to evaluation mode during this process.
            - The augmentation and data are moved to the appropriate device (e.g., GPU or CPU).
            - Evaluation is performed without gradient computation to save memory and improve performance.
            - Metrics are logged using tqdm for progress tracking.
        """
        
        # Initialize evaluation metrics
        iou = IoU()
        dice = Dice()
        pixel_acc = PixelAccuracy()

        # Set the model to evaluation mode
        self.model.eval()
        running_iou_loss = 0.0
        running_pixel_acc_loss = 0.0
        running_dice_loss = 0.0

        # Move the augmentation to the appropriate device
        augmentation.to(self.device)
        
        # Perform evaluation without gradient computation
        with torch.no_grad():
            for inputs, targets in tqdm(
            self.test_dataloader, 
            desc=f"Evaluating performance metric {augmentation.__class__.__name__} for parameter value {param_val}", 
            leave=False
            ):
                # Move inputs and targets to the appropriate device
                inputs, targets = inputs.to(self.device, non_blocking=True), targets.to(self.device, non_blocking=True)

                # Apply the augmentation to the inputs
                inputs = augmentation(inputs)
                
                # Perform a forward pass through the model
                outputs = self.model(inputs)
                
                # Compute evaluation metrics
                iou_loss = iou(outputs, targets)
                pixel_acc_loss = pixel_acc(outputs, targets)
                dice_loss = 2 * iou(outputs, targets) / (1 + iou(outputs, targets))
                
                # Accumulate the metrics
                running_iou_loss += iou_loss.item()
                running_pixel_acc_loss += pixel_acc_loss.item()
                running_dice_loss += dice_loss.item()
            
        # Calculate average metrics over the dataset
        avg_iou_loss = running_iou_loss / len(self.test_dataloader)
        avg_pixel_acc_loss = running_pixel_acc_loss / len(self.test_dataloader)
        avg_dice_loss = running_dice_loss / len(self.test_dataloader)

        # Log the results for the current augmentation and parameter value
        tqdm.write(f"Augmentation: {augmentation.__class__.__name__}, Parameter Value: {param_val}")
        tqdm.write(f"IoU: {avg_iou_loss:.4f}")
        tqdm.write(f"Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
        tqdm.write(f"Dice: {avg_dice_loss:.4f}")
        tqdm.write('\n')

        # Return the average metrics
        return avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss
        
    def log_results_to_csv(self,results, filename= None):
        """
        Logs the results of augmentations to a CSV file.
        This method appends the provided results to a CSV file. If the file does not exist, 
        it will be created, and headers will be written to the file. Each result is written 
        as a row in the CSV file.
        Args:
            results (list of tuples): A list where each tuple contains the following:
                - Augmentation (str): The name of the augmentation.
                - Parameter Value (float or int): The value of the parameter used for the augmentation.
                - Avg IoU Loss (float): The average Intersection over Union (IoU) loss.
                - Avg Pixel Accuracy Loss (float): The average pixel accuracy loss.
                - Avg Dice Loss (float): The average Dice loss.
            filename (str, optional): The name of the CSV file to write to. Defaults to None. 
                If provided, it will be prefixed with 'augmentation-results/'.
        Raises:
            IOError: If there is an issue opening or writing to the file.
        Notes:
            - The file is opened in append mode, so new results will be added to the end of the file.
            - Headers are written only if the file is empty.
        """
        
        filename = 'augmentation-results/' + filename
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
        Tests the effect of Gaussian pixel noise augmentation on model performance.
        This method evaluates the model's performance metrics (IoU loss, pixel accuracy loss, 
        and Dice loss) when subjected to Gaussian pixel noise with varying standard deviations.
        Args:
            parameter_values (list of float): A list of standard deviation values for the 
                Gaussian noise to be applied.
        Returns:
            None: The results are logged to a CSV file named 'gaussian_pixel_noise_results.csv'.
        The results include:
            - Augmentation name ('Gaussian Pixel Noise')
            - Standard deviation of the Gaussian noise
            - Average IoU loss
            - Average pixel accuracy loss
            - Average Dice loss
        """
        
        # Initialize an empty list to store the results
        results = []
        
        # Iterate over each parameter value (standard deviation of Gaussian noise)
        for std in parameter_values:
            # Create an instance of the GaussianPixelNoise augmentation with the specified standard deviation
            augmentation = GaussianPixelNoise(std)
            
            # Test the augmentation and retrieve the performance metrics
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, std)
            
            # Append the results (augmentation name, parameter value, and metrics) to the results list
            results.append(('Gaussian Pixel Noise', std, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV file
        self.log_results_to_csv(results, filename='gaussian_pixel_noise_results.csv')


    def test_repeated_blur(self, parameter_values):
        """
        Tests the performance of the RepeatedBlur augmentation with varying parameter values.
        This method evaluates the effect of applying the RepeatedBlur augmentation multiple times
        on the model's performance. It iterates over a list of parameter values, where each value
        represents the number of blur repetitions, and computes performance metrics for each case.
        Args:
            parameter_values (list of int): A list of integers specifying the number of blur 
                repetitions to test.
        Returns:
            None: The results, including the augmentation name, parameter value, and performance 
            metrics (average IoU loss, average pixel accuracy loss, and average Dice loss), are 
            logged to a CSV file named 'repeated_blur_results.csv'.
        """
        
        # Initialize an empty list to store the results
        results = []
        
        # Iterate over each parameter value (number of blur repetitions)
        for times in parameter_values:
            # Create an instance of the RepeatedBlur augmentation with the specified number of repetitions
            augmentation = RepeatedBlur(times)
            
            # Test the augmentation and retrieve the performance metrics
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, times)
            
            # Append the results (augmentation name, parameter value, and metrics) to the results list
            results.append(('Repeated Blur', times, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV file
        self.log_results_to_csv(results, filename='repeated_blur_results.csv')


    def test_contrast_change(self, parameter_values, increase=True):
        """
        Tests the effect of contrast change on the model's performance by applying 
        a ContrastChange augmentation with specified parameter values.
        Args:
            parameter_values (list of float): A list of contrast factors to test. 
                Each factor represents the level of contrast adjustment.
            increase (bool, optional): A flag indicating whether the contrast is 
                being increased or decreased. Defaults to True.
        Returns:
            None: The results of the test are logged to a CSV file. If `increase` 
            is True, results are saved to 'contrast_increase_results.csv'. If 
            `increase` is False, results are saved to 'contrast_decrease_results.csv'.
        Notes:
            - The method iterates over the provided contrast factors, applies the 
              ContrastChange augmentation, and evaluates the model's performance 
              using metrics such as IoU loss, pixel accuracy loss, and Dice loss.
            - The results are stored in a CSV file for further analysis.
        """
        
        # Initialize an empty list to store the results
        results = []
        
        # Iterate over each parameter value (contrast factor)
        for factor in parameter_values:
            # Create an instance of the ContrastChange augmentation with the specified factor
            augmentation = ContrastChange(factor)
            
            # Test the augmentation and retrieve the performance metrics
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, factor)
            
            # Append the results (augmentation name, parameter value, and metrics) to the results list
            results.append(('Contrast Change', factor, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV file
        if increase:
            # If contrast is being increased, log results to 'contrast_increase_results.csv'
            self.log_results_to_csv(results, filename='contrast_increase_results.csv')
        else:
            # If contrast is being decreased, log results to 'contrast_decrease_results.csv'
            self.log_results_to_csv(results, filename='contrast_decrease_results.csv')


    def test_brightness_change(self, parameter_values, increase=True):
        """
        Tests the effect of brightness change on image segmentation performance.
        This method evaluates the performance of a BrightnessChange augmentation
        by applying it with various parameter values (brightness offsets) and 
        measuring the impact on segmentation metrics such as IoU loss, pixel 
        accuracy loss, and Dice loss.
        Args:
            parameter_values (list of float): A list of brightness offset values 
                to test the augmentation with.
            increase (bool, optional): A flag indicating whether the brightness 
                is being increased or decreased. Defaults to True.
        Logs:
            - Results are logged to a CSV file:
              - 'brightness_increase_results.csv' if `increase` is True.
              - 'brightness_decrease_results.csv' if `increase` is False.
        Returns:
            None
        """
        
        # Initialize an empty list to store the results
        results = []
        
        # Iterate over each parameter value (brightness offset)
        for offset in parameter_values:
            # Create an instance of the BrightnessChange augmentation with the specified offset
            augmentation = BrightnessChange(offset)
            
            # Test the augmentation and retrieve the performance metrics
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, offset)
            
            # Append the results (augmentation name, parameter value, and metrics) to the results list
            results.append(('Brightness Change', offset, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV file
        if increase:
            # If brightness is being increased, log results to 'brightness_increase_results.csv'
            self.log_results_to_csv(results, filename='brightness_increase_results.csv')
        else:
            # If brightness is being decreased, log results to 'brightness_decrease_results.csv'
            self.log_results_to_csv(results, filename='brightness_decrease_results.csv')


    def test_occlusion(self, parameter_values):
        """
        Tests the effect of occlusion augmentation on the model's performance.
        This method iterates over a list of occlusion sizes, applies the occlusion
        augmentation to the input data, and evaluates the model's performance using
        metrics such as IoU loss, pixel accuracy loss, and Dice loss. The results
        are logged to a CSV file for further analysis.
        Args:
            parameter_values (list): A list of occlusion sizes to test.
        Returns:
            None
        """
        # Initialize an empty list to store the results
        results = []
        
        # Iterate over each parameter value (occlusion size)
        for size in parameter_values:
            # Create an instance of the Occlusion augmentation with the specified size
            augmentation = Occlusion(size)  
            
            # Test the augmentation and retrieve the performance metrics
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, size)
            
            # Append the results (augmentation name, parameter value, and metrics) to the results list
            results.append(('Occlusion', size, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV file
        self.log_results_to_csv(results, filename='occlusion_results.csv')


    def test_salt_and_pepper(self, parameter_values):
        """
        Tests the effect of the Salt and Pepper augmentation on the model's performance.
        This method applies the Salt and Pepper augmentation with varying amounts 
        specified in the `parameter_values` list. For each amount, it evaluates the 
        model's performance using the `test_augmentation` method and logs the results 
        to a CSV file.
        Args:
            parameter_values (list): A list of float values representing the amount 
                        of Salt and Pepper noise to apply.
        Returns:
            None: The results are logged to a CSV file named 'salt_and_pepper_results.csv'.
        """
       
        # Initialize an empty list to store the results
        results = []
        
        # Iterate over each parameter value (amount of Salt and Pepper noise)
        for amount in parameter_values:
            # Create an instance of the SaltAndPepper augmentation with the specified amount
            augmentation = SaltAndPepper(amount)
            
            # Test the augmentation and retrieve the performance metrics
            avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss = self.test_augmentation(augmentation, amount)
            
            # Append the results (augmentation name, parameter value, and metrics) to the results list
            results.append(('Salt and Pepper', amount, avg_iou_loss, avg_pixel_acc_loss, avg_dice_loss))
        
        # Log the results to a CSV
        self.log_results_to_csv(results, filename='salt_and_pepper_results.csv')

    def test_robustness(self):

        # a) Gaussian pixel noise
        self.test_gaussian_pixel_noise([1e-6, 2, 4, 6, 8, 10, 12, 14, 16, 18])

        # b) Gaussian blurring
        self.test_repeated_blur([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

        # c) Image Contrast Increase
        self.test_contrast_change([1.0, 1.01, 1.02, 1.03, 1.04, 1.05, 1.1, 1.15, 1.20, 1.25], True)

        # d) Image Contrast Decrease
        self.test_contrast_change([1.0, 0.95, 0.90, 0.85, 0.80, 0.60, 0.40, 0.30, 0.20, 0.10], False)

        # e) Image Brightness Increase
        self.test_brightness_change([0, 5, 10, 15, 20, 25, 30, 35, 40, 45], True)

        # f) Image Brightness Decrease
        self.test_brightness_change([0, -5, -10, -15, -20, -25, -30, -35, -40, -45], True)

        # g) Occlusion of the Image Increase
        self.test_occlusion([0, 5, 10, 15, 20, 25, 30, 35, 40, 45])

        # h) Salt and Pepper Noise
        self.test_salt_and_pepper([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16])

    
    def plot_predicitons(self, indicies = None):
        """
        Plots predictions of the model for a set of images from the test dataset.

        Arguments:
        indicies (list, optional): A list of indices specifying which images from the test dataset 
                                   to use for generating predictions. If None, random indices are selected.

        This method fetches the specified images and their corresponding targets from the test dataset, 
        propagates the images through the model to generate predictions, and prepares the data for visualization.
        """
        if indicies is None:
            # Select random indices if none are provided
            indicies = [int(i) for i in torch.randint(0, len(self.test_dataset), (4,))]

        # Fetch images and their corresponding targets from the test dataset
        images, targets = zip(*[self.test_dataloader.dataset[i] for i in indicies])
        images = torch.stack(images).to(self.device)
        targets = torch.stack(targets).to(self.device)

        # Set the model to evaluation mode and generate predictions
        self.model.eval()
        with torch.no_grad():
            predictions = self.model(images)

        plot_segmentations(images, predictions, n_cols=2)


class DistributedTrainingWrapper:
    """
    DistributedTrainingWrapper is a utility class designed to facilitate distributed training 
    of deep learning models using PyTorch. It provides an abstraction for managing the training 
    and validation processes in a distributed environment, including support for mixed precision 
    training, data augmentation, and evaluation metrics.
    The class handles the initialization of the model, optimizer, and loss function, as well as 
    the setup of distributed training utilities such as gradient scaling and synchronization. 
    It also provides functionality for logging training and validation metrics, saving model 
    checkpoints, and ensuring proper shuffling of data in distributed settings.
    Key Features:
    - Distributed training support with PyTorch's `torch.distributed` module.
    - Mixed precision training using `torch.autocast` and `torch.GradScaler`.
    - Automatic handling of data augmentation during training.
    - Logging of training and validation metrics, including loss, IoU, Dice coefficient, and Pixel Accuracy.
    - Automatic saving of model checkpoints and training information.
    - Progress bar visualization for the main process (rank 0) using `tqdm`.
    - save_location (str): Directory where the model and training information are saved.
    - batch_size (int): Batch size for training and validation.
    - model (torch.nn.Module): The model being trained and validated.
    - train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
    - validation_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
    - data_augmentor (callable): Data augmentation function or object.
    - rank (int): Rank of the current process in the distributed setup.
    - optimizer (torch.optim.Optimizer): Optimizer initialized with the model parameters.
    - criterion (torch.nn.Module): Loss function initialized for training.
    Methods:
    - __init__: Initializes the wrapper with the model, dataloaders, optimizer, and other configurations.
    - train: Trains the model for a specified number of epochs, logging metrics and saving checkpoints.
    """
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
        
        """
        Initializes the model wrapper for training and validation.
        Args:
            rank (int): The rank of the current process. Used for distributed training.
            model (torch.nn.Module): The model to be trained and validated.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            validation_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            data_augmentor (callable): A callable object or function for data augmentation.
            batch_size (int, optional): Batch size for training and validation. Defaults to 100.
            save_location (str, optional): Directory to save the model and training information. 
                                            Defaults to None, which generates a default path.
            criterion_class (type, optional): Loss function class to be used. Defaults to HybridLoss.
            optimizer_class (type, optional): Optimizer class to be used. Defaults to optim.Adam.
            optimizer_args (dict, optional): Arguments for the optimizer. Defaults to 
                                                {'lr': 0.001, 'weight_decay': 2e-4}.
        Attributes:
            save_location (str): Path to the directory where the model and training information are saved.
            batch_size (int): Batch size for training and validation.
            model (torch.nn.Module): The model being trained and validated.
            train_dataloader (torch.utils.data.DataLoader): DataLoader for the training dataset.
            validation_dataloader (torch.utils.data.DataLoader): DataLoader for the validation dataset.
            data_augmentor (callable): Data augmentation function or object.
            rank (int): Rank of the current process.
            optimizer (torch.optim.Optimizer): Optimizer initialized with the model parameters.
            criterion (torch.nn.Module): Loss function initialized for training.
        """
        # If the current process is the main process (rank 0), set up the save location
        if rank == 0:
            if save_location is None:
                # Default save location based on the model's class name
                save_location = "saved-models/" + model.module.__class__.__name__

            # Get the next available folder for saving the model
            self.save_location = get_next_run_folder(save_location)

        # Store the batch size
        self.batch_size = batch_size

        # Assign the model, dataloaders, and data augmentor to instance variables
        self.model = model
        self.train_dataloader = train_dataloader
        self.validation_dataloader = validation_dataloader
        self.data_augmentor = data_augmentor
        self.rank = rank  # Store the rank of the current process

        # Initialize the optimizer with the model parameters and specified arguments
        self.optimizer = optimizer_class(self.model.parameters(), **optimizer_args)

        # Initialize the loss criterion
        self.criterion = criterion_class()

        # Calculate the total number of parameters in the model
        num_params = sum(p.numel() for p in self.model.parameters())

        # If the current process is the main process (rank 0), save training information
        if self.rank == 0:
            save_training_info(model,
                    self.optimizer,
                    self.criterion,
                    self.train_dataloader,
                    self.validation_dataloader,
                    self.save_location, 
                    extra_params = {'num_params': num_params})



    def train(self, num_epochs):
        """
        Trains the model for a specified number of epochs.
        Args:
            num_epochs (int): The number of epochs to train the model.
        Description:
            This method performs distributed training of the model using mixed precision 
            and data augmentation. It logs training and validation metrics, including 
            loss, IoU, Dice coefficient, and Pixel Accuracy, for each epoch. The method 
            also saves the model's state_dict and logs metrics to a CSV file for the 
            main process (rank 0).
        Workflow:
            1. Initializes gradient scaler for mixed precision training.
            2. Sets up evaluation metrics (IoU, Dice, Pixel Accuracy).
            3. Iterates through the specified number of epochs:
                - Trains the model using the training dataloader.
                - Computes training loss and performance metrics.
                - Validates the model using the validation dataloader.
                - Computes validation loss and performance metrics.
            4. Logs metrics and saves the model for the main process (rank 0).
            5. Synchronizes all processes to ensure consistency in distributed training.
        Notes:
            - Mixed precision training is enabled using `torch.autocast` and `torch.GradScaler`.
            - Progress bars are displayed only for the main process (rank 0).
            - Distributed training requires proper setup of `torch.distributed`.
        Raises:
            RuntimeError: If distributed training is not properly initialized.
        """

        # If the current process is the main process (rank 0), write the CSV header for logging
        if self.rank == 0:
            write_csv_header(self.save_location)

        # Initialize a gradient scaler for mixed precision training on CUDA
        gradscaler = torch.GradScaler('cuda')

        # Initialize evaluation metrics for IoU, Dice coefficient, and Pixel Accuracy
        iou = IoU()
        dice = Dice()
        pixel_acc = PixelAccuracy()

        # Determine whether to disable tqdm progress bars based on the process rank
        if self.rank == 0:
            disable = False  # Enable progress bars for the main process
        else:
            disable = True  # Disable progress bars for other processes

        for epoch in tqdm( range(num_epochs), desc='Training', unit = 'Epoch', leave = False, disable=disable ):

            # Set the epoch for the sampler to ensure proper shuffling in distributed training
            self.train_dataloader.sampler.set_epoch(epoch)
            
            # Set the model to training mode
            self.model.train()
            running_loss = 0.0  # Initialize the running loss accumulator

            # Record the start time for performance measurement
            start_time = time.time()
            
            # Training loop
            for inputs, targets in tqdm(self.train_dataloader, desc=f"[Rank:{self.rank}] Epoch {epoch+1}/{num_epochs} Training", unit=' batch', leave=False):
                inputs, targets = inputs.to(self.rank, non_blocking=True), targets.to(self.rank, non_blocking=True)

                # Apply data augmentation
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

                    #enable mixed precision
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
            
            # Log validation metrics for the current epoch
            tqdm.write(f"[Rank:{self.rank}] Epoch: {epoch}")
            tqdm.write(f"[Rank:{self.rank}] Rate: {rate:.1f} datapoints/s")
            tqdm.write(f"[Rank:{self.rank}] Train Loss: {avg_train_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Validation Loss: {avg_val_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Val IoU: {avg_iou_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Val Pixel Accuracy: {avg_pixel_acc_loss:.4f}")
            tqdm.write(f"[Rank:{self.rank}] Val Dice: {avg_dice_loss:.4f}")
            tqdm.write('\n')

            # Perform rank-specific operations (only rank 0 handles logging and saving)
            if self.rank == 0:
                # Log training and validation losses to a CSV file
                log_loss_to_csv(epoch, avg_train_loss, avg_val_loss, avg_pixel_acc_loss, avg_dice_loss, avg_iou_loss, self.save_location)

                # Save the current model's state_dict to a file
                torch.save(self.model.state_dict(), f'{self.save_location}model_{epoch+1}.pth')

            # Synchronize all processes to ensure consistency
            torch.distributed.barrier()
            