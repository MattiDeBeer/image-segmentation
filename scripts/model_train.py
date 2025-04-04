from models.model_wrappers import TrainingWrapper
from models.UNet import UNet
from customDatasets.datasets import DummyDataset, CustomImageDataset

Trainer = TrainingWrapper(
    model_class= UNet,
    model_arguments={'out_channels' : 3},
    model_compilation_args = {},
    num_workers= 0,
    batch_size = 10,
    train_dataset_class = CustomImageDataset,
    train_dataset_args = {'split' : 'validation'},
    validation_dataset_class= CustomImageDataset,
    validation_dataset_args = {'split' : 'validation'},
)

Trainer.train(1)









