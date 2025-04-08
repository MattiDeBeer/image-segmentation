#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:47:56 2025

@author: matti
"""
import torch
import torch.nn as nn
from torch import cat
import torch.nn.functional as F
from transformers import CLIPProcessor, CLIPModel
import torchvision.models as models
import torchvision.transforms as transforms
import random
import kornia.augmentation as K
import kornia.filters as KF
import kornia as K


class ConvBlock(nn.Module):
    """
    A convolutional block that consists of two convolutional layers, each followed by 
    batch normalization and a ReLU activation function. This block is commonly used 
    in image segmentation and other computer vision tasks to extract features from 
    input images.
    Attributes:
        conv (nn.Sequential): A sequential container of layers that includes two 
            convolutional layers, batch normalization, and ReLU activation.
    Methods:
        forward(x):
            Passes the input tensor through the convolutional block.
    Args:
        in_channels (int): Number of input channels for the first convolutional layer.
        out_channels (int): Number of output channels for both convolutional layers.
        kernel_size (int, optional): Size of the convolutional kernel. Default is 3.
        padding (int, optional): Padding added to all four sides of the input. Default is 1.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ConvBlockDownsample(nn.Module):
    """
    A convolutional block with downsampling functionality.
    This class defines a neural network module that applies a convolutional 
    block followed by a max pooling operation to downsample the spatial 
    dimensions of the input tensor.
    Attributes:
        block (nn.Sequential): A sequential container that combines a 
            ConvBlock for feature extraction and a MaxPool2d layer for 
            downsampling.
    Methods:
        forward(x):
            Applies the convolutional block and max pooling to the input tensor.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class ConvBlockUpsampleSkip(nn.Module):
    """
    A neural network module that performs upsampling with skip connections 
    followed by a convolutional block. This is commonly used in encoder-decoder 
    architectures for tasks like image segmentation.
    Attributes:
        up (nn.ConvTranspose2d): A transposed convolution layer for upsampling 
            the input tensor.
        conv (ConvBlock): A convolutional block applied after concatenating 
            the upsampled tensor with the skip connection.
    Methods:
        forward(x, skip):
            Performs the forward pass of the module. Upsamples the input tensor, 
            aligns it to the spatial dimensions of the skip connection, concatenates 
            the two tensors, and applies the convolutional block.
    Args:
        in_channels (int): Number of input channels for the upsampling layer.
        out_channels (int): Number of output channels for the upsampling layer 
            and the convolutional block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels * 2, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = F.interpolate(x, size=skip.shape[2:], mode="bilinear", align_corners=True)
        x = cat([x, skip], dim=1)
        return self.conv(x)
        
class ConvBlockUpsample(nn.Module):
    """
    A convolutional block with upsampling functionality.
    This class defines a neural network module that performs upsampling 
    using a transposed convolution followed by a convolutional block. 
    It is typically used in encoder-decoder architectures for tasks 
    such as image segmentation or super-resolution.
    Attributes:
        up (nn.ConvTranspose2d): A transposed convolution layer for upsampling.
        conv (ConvBlock): A convolutional block applied after upsampling.
    Methods:
        forward(x):
            Applies the upsampling operation followed by the convolutional block.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels , out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
        

class CustomClipPreprocessor(nn.Module):
    """
    A PyTorch module for preprocessing images by resizing and normalizing them.
    This class is designed to preprocess a batch of images by resizing them to a 
    specified target size and normalizing them using the provided mean and standard 
    deviation values. It is particularly useful for preparing images for input into 
    deep learning models that require a specific input size and normalization.
    Attributes:
        resize (transforms.Resize): A transformation that resizes images to the target size.
        normalize (transforms.Normalize): A transformation that normalizes images using 
            the specified mean and standard deviation.
    Args:
        mean (list or tuple): The mean values for each channel used for normalization.
        std (list or tuple): The standard deviation values for each channel used for normalization.
        target_size (tuple, optional): The target size for resizing the images. Defaults to (224, 224).
    Methods:
        forward(images):
            Applies resizing and normalization transformations to a batch of images.
    """

    def __init__(self, mean, std, target_size=(224, 224)):
        super().__init__()

        self.resize = transforms.Resize(target_size)
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def forward(self, images):

        transformed_images = []
        for img in images:
            img = self.resize(img)
            img = self.normalize(img)
            transformed_images.append(img)

        return torch.stack(transformed_images)
    

class ClipFeatureExtractor(nn.Module):
    """
    A PyTorch module for extracting image features using the CLIP model. This class provides
    functionality to preprocess input images, load a pretrained CLIP model, and extract image
    features. It also allows toggling between training and inference modes for the CLIP model.
    Attributes:
        device (torch.device): The device (CPU or GPU) on which the model and data are loaded.
        custom_preprocessor (CustomClipPreprocessor): A custom preprocessor for resizing and 
            normalizing input images.
        clip_model (CLIPModel): The pretrained CLIP model used for feature extraction.
        train_clip (bool): A flag indicating whether the CLIP model is in training mode.
    Methods:
        __init__(train=False):
            Initializes the ClipFeatureExtractor, sets up the device, preprocessor, and loads
            the pretrained CLIP model.
        set_train(value: bool):
            Sets the training mode for the CLIP model. Enables or disables gradient computation
            for the model's parameters.
        forward(X):
            Processes the input images through the custom preprocessor and extracts image
            features using the CLIP model. Operates in training or inference mode based on
            the `train_clip` attribute.
    """

    def __init__(self, train=False):
        super().__init__()

        # Device setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Initialize the custom preprocessor (for resizing and normalization)
        mean = [0.48145466, 0.4578275, 0.40821073]  # CLIP mean
        std = [0.26862954, 0.26130258, 0.27577711]   # CLIP std
        self.custom_preprocessor = CustomClipPreprocessor(mean=mean, std=std)

        # Load the CLIP model onto the GPU
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)

        # Set training mode
        self.train_clip = train
        self.set_train(train)

    def set_train(self, value: bool):
        assert isinstance(value, bool), "Value must be a boolean"
        for param in self.clip_model.parameters():
            param.requires_grad = value

        self.train_clip = value

    def forward(self, X):
        # Apply the custom preprocessor (resize and normalize images)
        inputs = self.custom_preprocessor(X)

        # Pass the preprocessed inputs to the CLIP model and get the features
        if self.train_clip:
            image_features = self.clip_model.get_image_features(pixel_values=inputs)
        else:
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(pixel_values=inputs)

        return image_features

    
class ResNet34FeatureExtractor(nn.Module):
    """
    A feature extractor based on the ResNet-34 architecture from the torchvision library. 
    This class allows for extracting high-level feature maps from input images by removing 
    the final fully connected layers of the ResNet-34 model. It also provides an option 
    to enable or disable training of the model's parameters.
    Attributes:
        model (nn.Sequential): A sequential model containing all layers of ResNet-34 
            except the final two layers.
        train_res (bool): A flag indicating whether the model is in training mode 
            (parameters require gradients) or not.
    Methods:
        __init__(train=False):
            Initializes the ResNet34FeatureExtractor with the option to enable or 
            disable training of the model's parameters.
        set_train(value: bool):
            Sets the training mode of the model by enabling or disabling gradient 
            computation for its parameters.
        forward(X):
            Performs a forward pass through the feature extractor. If the model is 
            not in training mode, the forward pass is performed without computing 
            gradients.
    """
    def __init__(self,train=False):
        super().__init__()

        resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.model = nn.Sequential(*list(resnet.children())[:-2])

        self.set_train(train)
        self.train_res = train

    def set_train(self,value : bool):

        assert isinstance(value,bool), "Value must be a boolean"

        for param in self.model.parameters():
                param.requires_grad = value
        
        self.train_res = value

    def forward(self,X):

        if self.train_res:
            out = self.model(X)
        else:
            with torch.no_grad():
                out = self.model(X)

        return out
    
class CrossAttentionFusion(nn.Module):
    """
    A PyTorch module that performs cross-attention fusion between ResNet features and CLIP features.
    This module uses a multi-head attention mechanism to combine spatial features from a ResNet backbone
    with global features from a CLIP model. The ResNet features are reshaped and treated as queries, while
    the CLIP features are treated as keys and values in the attention mechanism. The output is a fused
    feature map with the same spatial dimensions as the input ResNet features.
    Args:
        resnet_channels (int): The number of channels in the ResNet feature map.
        num_heads (int, optional): The number of attention heads. Default is 4.
    Methods:
        forward(resnet_feats, clip_feats):
            Performs the forward pass of the cross-attention fusion.
            Args:
                resnet_feats (torch.Tensor): A tensor of shape (B, C, H, W) representing the ResNet feature map,
                                             where B is the batch size, C is the number of channels, and H, W are
                                             the spatial dimensions.
                clip_feats (torch.Tensor): A tensor of shape (B, clip_dim) representing the CLIP feature vectors,
                                           where B is the batch size and clip_dim is the dimensionality of the CLIP features.
            Returns:
                torch.Tensor: A tensor of shape (B, C, H, W) representing the fused feature map.
    """

    def __init__(self, resnet_channels, num_heads=4):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(embed_dim=resnet_channels, num_heads=num_heads)

    def forward(self, resnet_feats, clip_feats):
        B, C, H, W = resnet_feats.shape
        resnet_feats = resnet_feats.flatten(2).permute(2, 0, 1)  # (H*W, B, C)
        clip_feats = clip_feats.unsqueeze(0).expand(H * W, -1, -1)  # Expand to (H*W, B, clip_dim)
        
        attn_output, _ = self.cross_attn(resnet_feats, clip_feats, clip_feats)
        attn_output = attn_output.permute(1, 2, 0).view(B, C, H, W)  # Reshape back

        return attn_output
    
class DataAugmentor(nn.Module):
    """
    A PyTorch module for applying data augmentation to images and their corresponding masks.
    This class is designed to perform both geometric and color-based transformations on batches
    of images and masks. It ensures that the transformations are applied consistently to both
    images and masks, and allows for saving original images and masks at specified intervals
    within the augmented batch.
    Attributes:
        augmentations_per_datapoint (int): The number of augmentations to apply per data point.
        geometric_transforms (torch.nn.Sequential): A sequence of geometric transformations 
            (e.g., random horizontal flip, random rotation).
        colour_transforms (torch.nn.Sequential): A sequence of color-based transformations 
            (e.g., color jitter, Gaussian blur).
    Methods:
        transform_batch(images, masks):
            Applies geometric and color transformations to a batch of images and masks.
        forward(images, masks):
            Applies the transformations to the input batch while preserving original images
            and masks at specified intervals.
    """
    def __init__(self,augmentations_per_datapoint):

        super().__init__()
        
        self.augmentations_per_datapoint = augmentations_per_datapoint

        self.geometric_transforms  = torch.nn.Sequential(
            K.RandomHorizontalFlip(),
            K.RandomRotation(90,resample='nearest', same_on_batch=False),
        )

        self.colour_transforms = torch.nn.Sequential(
            K.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2, same_on_batch=False),
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=1.0, same_on_batch=False)
        )

    def transform_batch(self, images, masks):

        concatenated_batch = torch.cat([images, masks.unsqueeze(1).float()], dim=1)

        concatenated_batch= self.geometric_transforms(concatenated_batch)

        masks = concatenated_batch[:,3,:,:]
        images = concatenated_batch[:,0:3,:,:]

        images = self.colour_transforms(images)


        return images, masks.long()
        
    def forward(self, images, masks):

        saved_images = images[::self.augmentations_per_datapoint+1]
        saved_masks = masks[::self.augmentations_per_datapoint+1]

        transformed_images, transformed_masks = self.transform_batch(images, masks)

        transformed_images[::self.augmentations_per_datapoint+1] = saved_images
        transformed_masks[::self.augmentations_per_datapoint+1] = saved_masks

        return transformed_images, transformed_masks
    
class DataAugmentorPrompt(nn.Module):
    """
    A PyTorch module for applying data augmentation to images, masks, and prompts in a batch. 
    This class is designed to perform both geometric and color transformations on the input data, 
    while ensuring that a subset of the original (non-augmented) data is preserved in the output.
    Attributes:
        augmentations_per_datapoint (int): The number of augmentations to apply per data point.
        geometric_transforms (torch.nn.Sequential): A sequence of geometric transformations 
            (e.g., random horizontal flip, random rotation).
        colour_transforms (torch.nn.Sequential): A sequence of color transformations 
            (e.g., color jitter, Gaussian blur).
    Methods:
        transform_batch(images, masks, prompts):
            Applies geometric and color transformations to the input batch of images, masks, and prompts.
        forward(images, masks, prompts):
            Applies the transformations to the input batch and reinserts a subset of the original 
            (non-augmented) data into the output.
    """

    def __init__(self, augmentations_per_datapoint):
        super().__init__()
        self.augmentations_per_datapoint = augmentations_per_datapoint

        self.geometric_transforms = torch.nn.Sequential(
            K.RandomHorizontalFlip(),
            K.RandomRotation(90, resample='nearest', same_on_batch=False),
        )

        self.colour_transforms = torch.nn.Sequential(
            K.ColorJitter(brightness=0.4, contrast=0.3, saturation=0.2, hue=0.2, same_on_batch=False),
            K.RandomGaussianBlur(kernel_size=(5, 5), sigma=(0.1, 2.0), p=1.0, same_on_batch=False)
        )

    def transform_batch(self, images, masks, prompts):

        if masks.dim() == 3:
            masks = masks.unsqueeze(1)
    
        concatenated_batch = torch.cat([images, masks.float(), prompts.float()], dim=1)  
        concatenated_batch = self.geometric_transforms(concatenated_batch)

        # Split them back:
        images_aug = concatenated_batch[:, :3, :, :]
        masks_aug = concatenated_batch[:, 3:4, :, :]   
        prompts_aug = concatenated_batch[:, 4:5, :, :]   

        # Apply colour transforms only to the images:
        images_aug = self.colour_transforms(images_aug)

        return images_aug, masks_aug.long(), prompts_aug

    def forward(self, images, masks, prompts):
        # Save original unaugmented items based on augmentations_per_datapoint
        saved_images  = images[::self.augmentations_per_datapoint + 1]
        saved_masks   = masks[::self.augmentations_per_datapoint + 1]
        saved_prompts = prompts[::self.augmentations_per_datapoint + 1]

        transformed_images, transformed_masks, transformed_prompts = self.transform_batch(images, masks, prompts)

        # Reinsert the saved (non-augmented) items:
        transformed_images[::self.augmentations_per_datapoint + 1]  = saved_images
        transformed_masks[::self.augmentations_per_datapoint + 1]   = saved_masks.unsqueeze(1)  # ensure channel dimension
        transformed_prompts[::self.augmentations_per_datapoint + 1] = saved_prompts

        # If your downstream expects masks to be [B, H, W], squeeze the channel dimension:
        return transformed_images, transformed_masks.squeeze(1), transformed_prompts


class GaussianPixelNoise(nn.Module):
    """
    A PyTorch module that applies Gaussian noise to an input image. This is useful for 
    data augmentation or simulating noisy environments in image processing tasks.
    Attributes:
        std (float): The standard deviation of the Gaussian noise to be applied. 
                     The value is scaled by dividing by 255.0 to normalize it for image data.
    Methods:
        forward(img):
            Adds Gaussian noise to the input image and clamps the resulting pixel values 
            to the range [0.0, 1.0].
    """

    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, img):
        noise = torch.randn_like(img) * (self.std / 255.0)
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0.0, 1.0)


class RepeatedBlur(nn.Module):
    """
    A PyTorch module that applies a repeated box blur operation to an input image.
    This class uses the Kornia library's `box_blur` function to perform the blurring.
    The number of times the blur is applied is specified during initialization.
    Attributes:
        times (int): The number of times the box blur operation is applied.
    Methods:
        forward(img):
            Applies the box blur operation to the input image the specified number of times.
    """

    def __init__(self, times):
        super().__init__()
        self.times = times

    def forward(self, img):
        for _ in range(self.times):
            img = KF.box_blur(img, (3, 3))
        return img


class ContrastChange(nn.Module):
    """
    A PyTorch module for adjusting the contrast of an image by multiplying 
    its pixel values by a specified factor. The resulting pixel values are 
    clamped to the range [0.0, 1.0].
    Attributes:
        factor (float): The factor by which to adjust the contrast of the image.
    Methods:
        forward(img):
            Adjusts the contrast of the input image by multiplying its pixel 
            values by the specified factor and clamps the result to the range [0.0, 1.0].
    """

    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = img * self.factor
        return torch.clamp(img, 0.0, 1.0)


class BrightnessChange(nn.Module):
    """
    A PyTorch module for adjusting the brightness of an image.
    This module adds a specified offset to the pixel values of an input image
    and ensures that the resulting values are clamped within the valid range [0.0, 1.0].
    Attributes:
        offset (float): The brightness adjustment value, scaled to the range [0.0, 1.0].
    Methods:
        forward(img):
            Applies the brightness adjustment to the input image and clamps the result.
    """

    def __init__(self, offset):
        super().__init__()
        self.offset = offset / 255.0

    def forward(self, img):
        img = img + self.offset
        return torch.clamp(img, 0.0, 1.0)


class Occlusion(nn.Module):
    """
    A PyTorch module that applies a random occlusion (masking) to input images. 
    This is typically used as a data augmentation technique to improve model robustness.
    Attributes:
        size (int): The size of the square occlusion mask to be applied.
    Methods:
        forward(img):
            Applies the occlusion mask to the input image tensor.
    """

    def __init__(self, size):
        super().__init__()
        self.size = size

    def forward(self, img):
        b, c, h, w = img.shape
        for i in range(b):
            x = random.randint(0, w - self.size) if w > self.size else 0
            y = random.randint(0, h - self.size) if h > self.size else 0
            img[i, :, y:y+self.size, x:x+self.size] = 0.0
        return img

class SaltAndPepper(nn.Module):
    """
    A PyTorch module that applies salt-and-pepper noise to an input image tensor. 
    Salt-and-pepper noise randomly sets some pixels to the maximum value (salt) 
    and some to the minimum value (pepper), simulating noise often encountered 
    in image processing tasks.
    Attributes:
        amount (float): The proportion of pixels to be affected by the noise. 
                        Should be a value between 0 and 1.
    Methods:
        forward(img):
            Applies the salt-and-pepper noise to the input image tensor.
    Example:
        salt_and_pepper = SaltAndPepper(amount=0.1)
        noisy_img = salt_and_pepper(img)
    """

    def __init__(self, amount):
        super().__init__()
        self.amount = amount

    def forward(self, img):
        b, c, h, w = img.shape
        noise = torch.rand((b, 1, h, w), device=img.device)
        salt = (noise < self.amount / 2).float()
        pepper = (noise > 1 - self.amount / 2).float()
        mask = 1 - salt - pepper
        return img * mask + salt