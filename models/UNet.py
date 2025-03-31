import torch 
import torch
import torch.nn as nn
from models.processing_blocks import ConvBlockDownsample, ConvBlockUpsampleSkip, ConvBlock, ConvBlockUpsample


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=2, activation = lambda x : x):
        super().__init__()
        
        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
       
        # Encoder
        self.enc1 = ConvBlockDownsample(32, 64)   # /2
        self.enc2 = ConvBlockDownsample(64, 128)           # /4
        self.enc3 = ConvBlockDownsample(128, 256)         # /8
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)             # /8
        
    
        self.dec1 = ConvBlockUpsampleSkip(512, 256)            # /4
        self.dec2 = ConvBlockUpsampleSkip(256, 128)            # /2
        self.dec3 = ConvBlockUpsampleSkip(128, 64)             # /1
        self.dec4 = ConvBlockUpsampleSkip(64, 32) 

        self.out = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)

        self.activation = activation


    def forward(self, x):
        # Encoder
        input = self.input(x)
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
     
        bottleneck = self.bottleneck(enc3)

        dec1 = self.dec1(bottleneck, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        dec4 = self.dec4(dec3,input)
        out = self.out(dec4)


        return self.activation(out)


if __name__ == "__main__":
    model = UNet(in_channels=3,out_channels=3)
    out = model(torch.randn(1,3,256,256)).detach().numpy()
    print(out.shape)
