from customDatasets.datasets import ImageDataset, DummyDataset, ImageDataset3Mask
from models.UNet import UNet
from torch.utils.data import DataLoader
from models.helperFunctions import show_datapoint, show_prediciton, show_detailed_prediction
import torch
from collections import OrderedDict
import torch.nn.functional as F


dataset_loc = '../../Datasets/Oxford-IIIT-Pet-Augmented'
#dataset_loc = "mattidebeer/Oxford-IIIT-Pet-Augmented" #uncomment to load remote dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

validation_dataset = ImageDataset3Mask(dataset=dataset_loc,split='validation', uncertianty_mask_coeff=0.7)


# Load state_dict
state_dict = torch.load("saved-models/UNetCrossEntropy/trained_model.pth", map_location="cpu")

# Remove "_orig_mod." prefix
new_state_dict = OrderedDict()
for k, v in state_dict.items():
    new_key = k.replace("_orig_mod.", "")  # Remove prefix
    new_state_dict[new_key] = v

# Load into model
model = UNet(out_channels=3)  # Ensure architecture matches
model.load_state_dict(new_state_dict)

test_point = validation_dataset[100]
test_image = test_point[0].unsqueeze(0)
test_label = test_point[1].unsqueeze(0)

model.eval()
out = model(test_image).detach()

show_datapoint(test_point)
show_detailed_prediction(test_image,F.softmax(out,dim=1))