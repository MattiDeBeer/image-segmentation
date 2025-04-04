from models.model_wrappers import TrainingWrapper
from models.CLIP_models import ClipUnet
from customDatasets.datasets import DummyDataset, CustomImageDataset

if __name__ == '__main__':
    
    Trainer = TrainingWrapper(
        model_class= ClipUnet,
        model_arguments={'out_channels' : 3},
        model_compilation_args = {'max-autotune' : True},
        num_workers = 4,
        batch_size = 80,
        train_dataset_class = CustomImageDataset,
        train_dataset_args = {'split' : 'test', 'cache' : True, 'augmentations_per_datapoint' : 4},
        validation_dataset_class= CustomImageDataset,
        validation_dataset_args = {'split' : 'validation', 'cache' : True},
    )

    Trainer.train(200)