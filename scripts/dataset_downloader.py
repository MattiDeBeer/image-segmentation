from customDatasets.datasets import ImageDataset, DummyDataset, ImageDataset3Mask
from torch.utils.data import DataLoader

train_dataset = ImageDatasetClasses(dataset="mattidebeer/Oxford-IIIT-Pet-Augmented",split='train')
val_dataset = ImageDatasetClasses(dataset="mattidebeer/Oxford-IIIT-Pet-Augmented",split='validation')