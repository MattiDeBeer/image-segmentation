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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

    def forward(self, x):
        return self.block(x)

class ConvBlockUpsampleSkip(nn.Module):
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
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(out_channels , out_channels)

    def forward(self, x):
        x = self.up(x)
        return self.conv(x)
        

class CustomClipPreprocessor(nn.Module):
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
    def __init__(self, std):
        super().__init__()
        self.std = std

    def forward(self, img):
        noise = torch.randn_like(img) * (self.std / 255.0)
        noisy_img = img + noise
        return torch.clamp(noisy_img, 0.0, 1.0)


class RepeatedBlur(nn.Module):
    def __init__(self, times):
        super().__init__()
        self.times = times

    def forward(self, img):
        for _ in range(self.times):
            img = KF.box_blur(img, (3, 3))
        return img


class ContrastChange(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, img):
        img = img * self.factor
        return torch.clamp(img, 0.0, 1.0)


class BrightnessChange(nn.Module):
    def __init__(self, offset):
        super().__init__()
        self.offset = offset / 255.0

    def forward(self, img):
        img = img + self.offset
        return torch.clamp(img, 0.0, 1.0)


class Occlusion(nn.Module):
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