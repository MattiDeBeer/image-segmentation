#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  4 17:34:49 2025

@author: jamie
"""
import torch
import torch.nn as nn
from models.processing_blocks import *
from models.autoencoder import Autoencoder
from models.pre_trained import SegmentationDecoderSkip

    

class PromptEncoder(nn.Module):
    def __init__(self, out_channels=512):
        super().__init__()
        self.enc1 = ConvBlockDownsample(1, 32)   
        self.enc2 = ConvBlockDownsample(32, 64)     
        self.enc3 = ConvBlockDownsample(64, out_channels)    

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(x1)
        x3 = self.enc3(x2)

        return x3  # Now the output is [B, out_channels, 32, 32]
class ClipUnetPrompt(nn.Module):

    def __init__(self, out_channels=1, in_channels=3, activation=nn.Identity()):
        super().__init__()

        self.clip_feature_extractor = ClipFeatureExtractor(train=False)
        self.cross_attention_fusion = CrossAttentionFusion(512, num_heads=1)

        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
       
        # Encoder
        self.enc1 = ConvBlockDownsample(32, 64)   # /2
        self.enc2 = ConvBlockDownsample(64, 128)  # /4
        self.enc3 = ConvBlockDownsample(128, 256) # /8
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)     # /8

        # Prompt encoder (your new encoder for prompts)
        self.prompt_encoder = PromptEncoder(out_channels=512)
        

        self.prompt_fusion = nn.Conv2d(1024, 512, kernel_size=1, padding=0)

        # Decoder 
        self.dec1 = ConvBlockUpsampleSkip(512, 256) # /4
        self.dec2 = ConvBlockUpsampleSkip(256, 128) # /2
        self.dec3 = ConvBlockUpsampleSkip(128, 64)  # /1
        self.dec4 = ConvBlockUpsampleSkip(64, 32) 

        self.out = nn.Conv2d(32, out_channels, kernel_size=1)
        self.activation = activation

    def forward(self, X, prompt_heatmap):

        # Image & CLIP feature extraction
        clip_features = self.clip_feature_extractor(X)
        input = self.input(X)
        
        # UNet encoding
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        bottleneck = self.bottleneck(enc3)

        # Prompt encoding
        prompt_embedding = self.prompt_encoder(prompt_heatmap)

        # Fuse the bottleneck with CLIP features
        attention_output = self.cross_attention_fusion(bottleneck, clip_features)

     
        # Concatenate 
        concat_bottleneck = torch.cat([attention_output, prompt_embedding], dim=1)
        fused_bottleneck = self.prompt_fusion(concat_bottleneck)

        # UNet decoding
        dec1 = self.dec1(fused_bottleneck, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        dec4 = self.dec4(dec3, input)

        out = self.out(dec4)
        return self.activation(out)

    



if __name__ == "__main__":

    # Initialize the model
    model = ClipUnetPrompt()
    # Create dummy inputs:
    # Example image: batch size 1, 3 channels, 256x256 pixels.
    dummy_image = torch.randn(1, 3, 256, 256)
    # Example prompt heatmap: batch size 1, 1 channel, 256x256 pixels.
    dummy_prompt = torch.randn(1, 1, 256, 256)
    
    # Run a forward pass.
    output = model(dummy_image, dummy_prompt)
    
    # Print the output shape.
    print("ClipUnetPrompt output shape:", output.shape)
