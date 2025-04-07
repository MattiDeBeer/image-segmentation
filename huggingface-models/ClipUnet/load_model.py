import torch
from model import ClipUnet
from collections import OrderedDict


def load_model(weights_path: str = "model.pth"):
    model = ClipUnet()
    state_dict = torch.load(weights_path, map_location="cpu", weights_only=True)
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        new_key = k.replace("_orig_mod.", "")  # Remove prefix
        new_state_dict[new_key] = v
    model.load_state_dict(new_state_dict)
    model.eval()
    return model
