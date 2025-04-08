
from models.model_wrappers import TestWrapper
from models.UNet import UNet, LargeUNet
from customDatasets.datasets import CustomImageDataset

tester = TestWrapper(
    model=LargeUNet(),
    test_dataset_class=CustomImageDataset,
    batch_size=16,
    model_load_location="results/models/LargeUNet/best_model.pth",

)

tester.test()