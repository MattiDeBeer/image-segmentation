from models.model_wrappers import TrainingWrapper
from models.UNet import LargeUNet
from customDatasets.datasets import DummyDataset, CustomImageDataset

if __name__ == '__main__':
    
    Trainer = TrainingWrapper(
        model_class= LargeUNet,
        model_arguments={'out_channels' : 3},
        model_compilation_args = {'mode' : 'max-autotune'},
        num_workers = 0,
        batch_size = 150,
        train_dataset_class = CustomImageDataset,
        train_dataset_args = {'split' : 'test', 'augmentations_per_datapoint' : 4, 'cache' : True},
        validation_dataset_class= CustomImageDataset,
        validation_dataset_args = {'split' : 'validation', 'cache' : True},
    )

    Trainer.train(200)