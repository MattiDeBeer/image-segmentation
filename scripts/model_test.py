
from models.model_wrappers import TestWrapper
from models.UNet import UNet, LargeUNet
from models.CLIP_models import ClipUnet, ClipResSegmentationModel, ClipAutoencoder
from customDatasets.datasets import CustomImageDataset

tester = TestWrapper(
    model=LargeUNet(),
    test_dataset_class=CustomImageDataset,
    batch_size=16,
    model_load_location="results/models/LargeUNet/best_model.pth",
    test_dataset_args = {'split':'test','augmentations_per_datapoint' : 0, 'cache' : False},
)

tester.test_robustness()