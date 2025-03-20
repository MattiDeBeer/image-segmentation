#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 14 18:47:56 2025

@author: matti
"""
import torch.nn as nn

class ConvBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=5, padding=2):
        
        super(ConvBlock,self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,X):
        
        X1 = self.conv(X)
        X2 = self.bn(X1)
        X3 = self.relu(X2)
        
        return X3
    
class ConvBlockDownsample(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(ConvBlockDownsample,self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels,kernel_size=3,padding=1)
        self.bn = nn.BatchNorm2d(out_channels)
        self.downsample = nn.MaxPool2d(2)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,X):
        
        X1 = self.conv(X)
        X2 = self.bn(X1)
        X3 = self.downsample(X2)
        X4 = self.relu(X3)
        
        return X4
        
class ConvBlockUpsample(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        super(ConvBlockUpsample,self).__init__()
        
        self.upsample = nn.ConvTranspose2d(in_channels, 2* in_channels, kernel_size=2,stride=2,padding=2)
        self.conv = nn.Conv2d(2* in_channels, out_channels, kernel_size=3,padding=3)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,X):
        
        X1 = self.upsample(X)
        X2 = self.conv(X1)
        X3 = self.bn(X2)
        X4 = self.relu(X3)
        
        return X4
    
class ConvBlockRes(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        
        ### IMPLEMENT THE RES CONNECTION ###
        raise NotImplementedError("The residual conneciton convolution block is not yet implemented")
        
        super(ConvBlock,self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,padding=padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,X):
        
        X1 = self.conv(X)
        X2 = self.bn(X1)
        X3 = self.relu(X2)
        
        return X3
        