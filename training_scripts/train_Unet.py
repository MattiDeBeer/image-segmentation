from models.model_wrappers import TrainingWrapper
from models.UNet import UNet
from customDatasets.datasets import DummyDataset, CustomImageDataset

if __name__ == '__main__':
    
    Trainer = TrainingWrapper(
        model_class= UNet,
        model_arguments={'out_channels' : 3},
        model_compilation_args = {'max_autotune' : True},
        num_workers = 4,
        batch_size = 100,
        train_dataset_class = CustomImageDataset,
        train_dataset_args = {'split' : 'test', 'cache' : True, 'augmentations_per_datapoint' : 4},
        validation_dataset_class= CustomImageDataset,
        validation_dataset_args = {'split' : 'validation', 'cache' : True},
    )

    Trainer.train(200)