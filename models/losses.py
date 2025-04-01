import segmentation_models_pytorch as smp
import torch.nn as nn

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss()

    def forward(self, pred, target):
        return 0.5 * self.dice(pred, target) + 0.5 * self.ce(pred, target)
    
