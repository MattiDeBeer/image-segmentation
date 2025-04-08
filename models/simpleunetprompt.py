#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  8 23:54:03 2025

@author: jamie
"""

import torch
import torch.nn as nn
from models.processing_blocks import (ConvBlockDownsample, ConvBlockUpsampleSkip, ConvBlock)

# A simplified UNet with prompt fused at the input
class SimplePromptUNet(nn.Module):
    def __init__(self, out_channels=1):
        super().__init__()
        
        # Convolution to map from (3+1)=4 channels to 32
        self.input = nn.Conv2d(4, 32, kernel_size=3, padding=1)
        
        # Encoder
        self.enc1 = ConvBlockDownsample(32, 64)   # /2
        self.enc2 = ConvBlockDownsample(64, 128)  # /4
        self.enc3 = ConvBlockDownsample(128, 256) # /8
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)     # /8

        # Decoder 
        self.dec1 = ConvBlockUpsampleSkip(512, 256)  # /4
        self.dec2 = ConvBlockUpsampleSkip(256, 128)  # /2
        self.dec3 = ConvBlockUpsampleSkip(128, 64)   # /1
        self.dec4 = ConvBlockUpsampleSkip(64, 32) 

        # Final output
        self.out = nn.Conv2d(32, out_channels, kernel_size=1)

    def forward(self, rgb_image, prompt_map):
        """
        rgb_image: [B, 3, H, W]
        prompt_map: [B, 1, H, W]
        """

        # Fuse at the input: shape [B, 4, H, W]
        x = torch.cat([rgb_image, prompt_map], dim=1)

        x = self.input(x)          # -> [B, 32, H, W]
        e1 = self.enc1(x)          # -> /2
        e2 = self.enc2(e1)         # -> /4
        e3 = self.enc3(e2)         # -> /8
        bottleneck = self.bottleneck(e3)

        d1 = self.dec1(bottleneck, e3)  # skip with e3
        d2 = self.dec2(d1, e2)         # skip with e2
        d3 = self.dec3(d2, e1)         # skip with e1
        d4 = self.dec4(d3, x)          # skip with x (the earliest features)

        out = self.out(d4)
        return out

if __name__ == "__main__":
    model = SimplePromptUNet(out_channels=1)
    print(model)

    dummy_image = torch.randn(1, 3, 256, 256)
    dummy_prompt = torch.randn(1, 1, 256, 256)
    output = model(dummy_image, dummy_prompt)
    print("Output shape:", output.shape)  # [1, 1, 256, 256]
