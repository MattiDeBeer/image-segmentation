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
import torch.cuda.nvtx as nvtx

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
        

class ClipFeatureExtractor(nn.Module):
    def __init__(self, train=False):
        super().__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load CLIP Model and Move to GPU
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        
        # Load Preprocessor (No need to move to GPU)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        # Ensure CLIP is Frozen (No Gradients)
        self.freeze_clip()

        # Control whether CLIP should be trainable
        self.train_clip = train
        self.set_train(train)

    def freeze_clip(self):
        """Disables gradient updates for CLIP model to exclude from optimization."""
        for param in self.clip_model.parameters():
            param.requires_grad = False

    def set_train(self, value: bool):
        """Allow fine-tuning of CLIP if explicitly enabled."""
        assert isinstance(value, bool), "Value must be a boolean"
        for param in self.clip_model.parameters():
            param.requires_grad = value

    def forward(self, X):
        X = X.to(self.device)  # Ensure input images are on GPU

        # Preprocess and move tensors to GPU
        inputs = self.clip_processor(images=X, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device, non_blocking=True) for k, v in inputs.items()}

        # Wrap CLIP model inference to exclude from CUDA graph
        nvtx.range_push("Skip_CLIP_CUDA_Graph")  # Start of the region to be excluded
        with torch.no_grad():  # Ensures no gradient computation
            image_features = self.clip_model.get_image_features(**inputs)
        nvtx.range_pop()  # End of the region to be excluded

        return image_features
    
class ResNet34FeatureExtractor(nn.Module):

    def __init__(self,train=False):
        super().__init__()

        resnet = models.resnet34(weights="IMAGENET1K_V1")
        self.model = nn.Sequential(*list(resnet.children())[:-2])

        self.set_train(train)
        self.train = train

    def set_train(self,value : bool):

        assert isinstance(value,bool), "Value must be a boolean"

        for param in self.model.parameters():
                param.requires_grad = value

    def forward(self,X):

        if self.train:
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
