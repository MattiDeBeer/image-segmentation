#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr  9 13:18:31 2025

@author: jamie
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.processing_blocks import *
class ClipUnet(nn.Module):
    """
    A UNet that uses CLIP features in the bottleneck, adapted for
    binary segmentation with an extra prompt channel.

    Expected input shape: [B, 4, H, W]
       - 3 channels = RGB image
       - 1 channel = prompt heatmap
    Output shape: [B, 1, H, W]
       - Logits for binary segmentation
    """

    def __init__(self, 
                 out_channels=1, 
                 in_channels=4,   # 3 (RGB) + 1 (prompt)
                 activation=nn.Identity()):
        super().__init__()
        
        # --- CLIP integration ---
        self.clip_feature_extractor = ClipFeatureExtractor(train=False)
        self.cross_attention_fusion = CrossAttentionFusion(512, num_heads=1)

        # --- UNet encoder/decoder ---
        # The first conv now expects 4 channels
        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)

        # Encoder
        self.enc1 = ConvBlockDownsample(32, 64)    # /2
        self.enc2 = ConvBlockDownsample(64, 128)   # /4
        self.enc3 = ConvBlockDownsample(128, 256)  # /8

        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)      # /8

        # Decoder
        self.dec1 = ConvBlockUpsampleSkip(512, 256)  # /4
        self.dec2 = ConvBlockUpsampleSkip(256, 128)  # /2
        self.dec3 = ConvBlockUpsampleSkip(128, 64)   # /1
        self.dec4 = ConvBlockUpsampleSkip(64, 32)

        # Final layer: 1 channel for binary logits
        self.out = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)
        
        # Optional activation (e.g. nn.Sigmoid), currently Identity for logits
        self.activation = activation

    def forward(self, x):
        """
        x: [B, 4, H, W]
            3 channels = RGB
            1 channel = prompt heatmap
        Returns: [B, 1, H, W] (binary segmentation logits)
        """

        # 1) Split the 4-channel input into:
        #    - clip_image: [B, 3, H, W]
        #    - prompt_heatmap: [B, 1, H, W]
        clip_image = x[:, :3, :, :]       # For CLIP
        prompt_heatmap = x[:, 3:, :, :]   # The extra channel

        # 2) Extract CLIP features from the RGB image only
        clip_features = self.clip_feature_extractor(clip_image)

        # 3) Concatenate prompt back with RGB for UNet
        #    shape => [B, 4, H, W]
        unet_input = torch.cat([clip_image, prompt_heatmap], dim=1)

        # 4) Standard UNet forward pass, but with cross-attention in the bottleneck
        enc0 = self.input(unet_input)    # [B, 32, H, W]
        enc1 = self.enc1(enc0)           # [B, 64, H/2, W/2]
        enc2 = self.enc2(enc1)           # [B, 128, H/4, W/4]
        enc3 = self.enc3(enc2)           # [B, 256, H/8, W/8]

        bottleneck = self.bottleneck(enc3)  # [B, 512, H/8, W/8]

        # 5) Fuse CLIP features at the bottleneck
        attention_output = self.cross_attention_fusion(bottleneck, clip_features)

        # 6) Decode
        dec1 = self.dec1(attention_output, enc3)  # skip connection with enc3
        dec2 = self.dec2(dec1, enc2)              # skip connection with enc2
        dec3 = self.dec3(dec2, enc1)              # skip connection with enc1
        dec4 = self.dec4(dec3, enc0)              # skip connection with enc0

        out = self.out(dec4)              # [B, 1, H, W] logits
        out = self.activation(out)         # e.g., Identity or Sigmoid

        return out
