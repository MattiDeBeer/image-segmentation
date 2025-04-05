import torch 
import torch
import torch.nn as nn
from models.processing_blocks import ConvBlockDownsample, ConvBlockUpsampleSkip, ConvBlock, ConvBlockUpsample


class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, activation = nn.Identity()):
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

class LargeUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, activation = nn.Identity()):
        super().__init__()
        
        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
       
        # Encoder
        self.enc1 = ConvBlockDownsample(32, 64)   # /2
        self.enc2 = ConvBlockDownsample(64, 128)           # /4
        self.enc3 = ConvBlockDownsample(128, 256)         # /8
        self.enc4 = ConvBlockDownsample(256, 512)         # /16
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)             # /16
        
        self.dec1 = ConvBlockUpsampleSkip(1024, 512)           # /8
        self.dec2 = ConvBlockUpsampleSkip(512, 256)            # /4
        self.dec3 = ConvBlockUpsampleSkip(256, 128)            # /2
        self.dec4 = ConvBlockUpsampleSkip(128, 64)             # /1
        self.dec5 = ConvBlockUpsampleSkip(64, 32) 

        self.out = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)

        self.activation = activation


    def forward(self, x):
        # Encoder
        input = self.input(x)
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
     
        bottleneck = self.bottleneck(enc4)

        dec1 = self.dec1(bottleneck, enc4)
        dec2 = self.dec2(dec1, enc3)
        dec3 = self.dec3(dec2, enc2)
        dec4 = self.dec4(dec3, enc1)
        dec5 = self.dec5(dec4, input)
        out = self.out(dec5)

        return self.activation(out)
    
class SegmentationClassificationUNet(nn.Module):
    def __init__(self, in_channels=3, activation = nn.Identity()):
        super().__init__()
        
        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
       
        # Encoder
        self.enc1 = ConvBlockDownsample(32, 64)   # /2
        self.enc2 = ConvBlockDownsample(64, 128)           # /4
        self.enc3 = ConvBlockDownsample(128, 256)         # /8
        
        # Bottleneck
        self.bottleneck = ConvBlock(256, 512)             # /8
        
        self.downsample = nn.MaxPool2d(2)

        self.dec1 = ConvBlockUpsampleSkip(512, 256)            # /4
        self.dec2 = ConvBlockUpsampleSkip(256, 128)            # /2
        self.dec3 = ConvBlockUpsampleSkip(128, 64)             # /1
        self.dec4 = ConvBlockUpsampleSkip(64, 32) 

        self.out = nn.Conv2d(32, 1, kernel_size=1, padding=0)

        self.classifier_head = nn.Sequential(
            nn.Linear(131072,300),
            nn.ReLU(),
            nn.Linear(300,2)
        )

        self.activation = activation


    def forward(self, x):
        # Encoder
        input = self.input(x)
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
     
        bottleneck = self.bottleneck(enc3)

        bottleneck_downsample = self.downsample(bottleneck)
        classifier_in = bottleneck_downsample.reshape(-1,512*16*16)

        classifier_out = self.classifier_head(classifier_in)

        dec1 = self.dec1(bottleneck, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        dec4 = self.dec4(dec3,input)
        out = self.out(dec4)


        return self.activation(out), classifier_out

if __name__ == "__main__":
    model = LargeUNet()
    out = model(torch.randn(1,3,256,256))

    print(out.size())


    
