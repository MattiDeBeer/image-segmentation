import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F

class HybridLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.dice = smp.losses.DiceLoss(mode="multiclass")
        self.ce = nn.CrossEntropyLoss()
        self.combined_confusion_loss = CombinedConfusionLoss(incorrect_penalty=1, confusion_pairs=[(1,2)], confusion_penalty=1)

    def forward(self, pred, target):
        return 0.25 * self.dice(pred, target) + 0.5 * self.ce(pred, target) + 0.25 * self.combined_confusion_loss(pred,target)
    
class IoULoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(IoULoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        preds: (B, C, H, W) - predicted probability maps (softmax applied)
        targets: (B, H, W) - ground truth labels (long tensor)
        """
        preds = nn.functional.softmax(preds,dim=1)
        num_classes = preds.shape[1]
        preds = torch.argmax(preds, dim=1)  # Convert softmax output to class indices
        
        iou_per_class = []
        for c in range(num_classes):
            pred_class = (preds == c).float()
            target_class = (targets == c).float()

            intersection = (pred_class * target_class).sum()
            union = pred_class.sum() + target_class.sum() - intersection
            
            iou = (intersection + self.eps) / (union + self.eps)
            iou_per_class.append(iou)

        return torch.mean(torch.stack(iou_per_class))  # change to 1 - IoU to make it a loss function
    
class DiceLoss(nn.Module):
    def __init__(self, eps=1e-6):
        super(DiceLoss, self).__init__()
        self.eps = eps

    def forward(self, preds, targets):
        """
        preds: (B, C, H, W) - predicted probability maps (softmax applied)
        targets: (B, H, W) - ground truth labels (long tensor)
        """
        num_classes = preds.shape[1]
        preds = torch.nn.functional.softmax(preds, dim=1)  # Ensure softmax probabilities
        targets_onehot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()

        dice_per_class = []
        for c in range(num_classes):
            pred_class = preds[:, c]
            target_class = targets_onehot[:, c]

            intersection = (pred_class * target_class).sum(dim=(1, 2))
            total = (pred_class + target_class).sum(dim=(1, 2))

            dice = (2. * intersection + self.eps) / (total + self.eps)
            dice_per_class.append(dice)

        return torch.mean(torch.stack(dice_per_class))  # convert to 1 - Dice to make it a loss function
    
class PixelAccuracyLoss(nn.Module):
    def __init__(self):
        super(PixelAccuracyLoss, self).__init__()

    def forward(self, preds, targets):
        """
        preds: (B, C, H, W) - predicted probability maps (softmax applied)
        targets: (B, H, W) - ground truth labels (long tensor)
        """
        preds = nn.functional.softmax(preds, dim=1)
        preds = torch.argmax(preds, dim=1)  # Convert softmax output to class indices
        
        num_classes = 3  # Specifically for 3 classes (0,1,2)
        accuracies = []
        
        for c in range(num_classes):
            # Create masks for current class
            class_mask = (targets == c)
            # Calculate accuracy for current class
            if class_mask.sum() > 0:  # Only if class exists in targets
                class_correct = ((preds == targets) & class_mask).float().sum()
                class_total = class_mask.float().sum()
                class_acc = class_correct / class_total
                accuracies.append(class_acc)
        
        return torch.stack(accuracies).mean()  # Average accuracy across all classes
    
class CombinedConfusionLoss(torch.nn.Module):
    def __init__(self, incorrect_penalty=2.0, confusion_pairs=[(1, 2)], confusion_penalty=2.0):
        super().__init__()
        self.incorrect_penalty = incorrect_penalty
        self.confusion_pairs = confusion_pairs
        self.confusion_penalty = confusion_penalty

    def forward(self, pred, target):
        """
        Args:
            pred: (B, C, H, W) - Raw logits from the model
            target: (B, H, W) - Ground truth class labels (one class per image)

        Returns:
            loss: Scaled cross-entropy loss with additional penalties
        """
        # Standard cross-entropy loss
        loss = F.cross_entropy(pred, target, reduction='none')

        # Get predicted class per pixel
        pred_classes = pred.argmax(dim=1)  # (B, H, W)

        # Apply incorrect prediction penalty (general case)
        incorrect_mask = pred_classes != target  # Pixels where prediction is incorrect
        loss[incorrect_mask] *= self.incorrect_penalty

        # Apply extra penalty for specific confusion pairs
        for cls1, cls2 in self.confusion_pairs:
            confusion_mask = ((pred_classes == cls1) & (target == cls2)) | ((pred_classes == cls2) & (target == cls1))
            loss[confusion_mask] *= self.confusion_penalty  # Extra penalty for confusing these pairs

        return loss.mean()