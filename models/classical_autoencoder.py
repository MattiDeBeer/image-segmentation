#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  1 14:27:50 2025

@author: jamie
"""

import torch
import torch.nn as nn
from models.processing_blocks import ConvBlockDownsample, ConvBlockUpsampleSkip, ConvBlock, ConvBlockUpsample

class Encoder(nn.Module):
    def __init__(self, in_channels=3):
        super().__init__()
        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
       
        # Encoder blocks
        self.enc1 = ConvBlockDownsample(32, 64)   # /2
        self.enc2 = ConvBlockDownsample(64, 64)   # /4 
        self.enc3 = ConvBlockDownsample(64, 64)   # /8
        self.enc4 = ConvBlockDownsample(64,64)      # /16
       
        # Bottleneck
        self.bottleneck = ConvBlock(64, 64)       # /16
    
    def forward(self, x):
        x0 = self.input(x)
        x1 = self.enc1(x0)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)
        x4 = self.enc4(x3)
       
        bottleneck = self.bottleneck(x4)  # Note lowercase 'b'
        #return bottleneck
        return {"x0": x0, "enc1": x1, "enc2": x2, "enc3": x3, "enc4": x4,  "bottleneck": bottleneck} #for use in segmentation decoder only

       
class Decoder(nn.Module):
    def __init__(self, out_channels=3):
        super().__init__()
        self.dec1 = ConvBlockUpsample(64, 64)    # /4
        self.dec2 = ConvBlockUpsample(64, 64)   
        self.dec3 = ConvBlockUpsample(64, 64)   
        self.dec4 = ConvBlockUpsample(64, 32)     # /1
        self.out = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)

    def forward(self, bottleneck):
        d1 = self.dec1(bottleneck)
        d2 = self.dec2(d1)
        d3 = self.dec3(d2)
        d4 = self.dec4(d3)
        out = self.out(d4)
        return out

class Autoencoder(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # Assign the encoder and decoder to attributes
        self.encoder = Encoder(in_channels)
        self.decoder = Decoder(out_channels)

    def forward(self, x):
        features = self.encoder(x)           # features is a dict of outputs
        bottleneck = features["bottleneck"] 
        out = self.decoder(bottleneck)
        return torch.sigmoid(out)

if __name__ == "__main__":
    model = Autoencoder(in_channels=3, out_channels=3)
    out = model(torch.randn(1, 3, 256, 256)).detach().numpy()
    print(out.shape)