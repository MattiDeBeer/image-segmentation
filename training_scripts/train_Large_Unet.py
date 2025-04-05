from models.model_wrappers import TrainingWrapper
from models.UNet import LargeUNet
from customDatasets.datasets import DummyDataset, CustomImageDataset

if __name__ == '__main__':
    
    Trainer = TrainingWrapper(
        model_class= LargeUNet,
        model_arguments={'out_channels' : 3},
        model_compilation_args = {'mode' : 'max-autotune'},
        num_workers = 12,
        batch_size = 200,
        train_dataset_class = CustomImageDataset,
        train_dataset_args = {'split' : 'test', 'augmentations_per_datapoint' : 4},
        validation_dataset_class= CustomImageDataset,
        validation_dataset_args = {'split' : 'validation'},
    )

    Trainer.train(200)