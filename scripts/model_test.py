import torch
import torch.nn as nn
from models.model_wrappers import TestWrapper
from models.pre_trained import SegmentationModel, SegmentationDecoderSkip
from customDatasets.datasets import CustomImageDataset
from models.autoencoder import Autoencoder 

autoencoder = Autoencoder()
autoencoder.load_state_dict(torch.load("saved-models/Autoencoders/model_179/model_179.pth"), map_location="cpu")
# Freeze encoder weights
for param in autoencoder.encoder.parameters():
    param.requires_grad = False

tester = TestWrapper(
    model=SegmentationModel(autoencoder.encoder, SegmentationDecoderSkip(out_channels=2)),
    test_dataset_class=CustomImageDataset,
    batch_size=16,
    model_load_location="saved-models/pre_trained_autoencoder/pre_trained_autoencoder_best_model.pth",
    test_dataset_args = {'split':'test','augmentations_per_datapoint' : 0, 'cache' : False},
)

tester.test_robustness()