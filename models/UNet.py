import torch 
import torch
import torch.nn as nn
from models.processing_blocks import ConvBlockDownsample, ConvBlockUpsample, ConvBlock


class UNet(nn.Module):
    

    def __init__(self,in_channels=3,out_channels=2):
       
        super(UNet, self).__init__()
        
        self.enc1 = ConvBlockDownsample(in_channels,8) #/2
        self.enc2 = ConvBlockDownsample(8, 16) #/4
        self.enc3 = ConvBlockDownsample(16, 32) #/8
        self.enc4 = ConvBlockDownsample(32,64)  #/16
        
        self.bottleneck = ConvBlock(64, 128, kernel_size=3,padding=1) #/16
        
        self.dec1 = ConvBlock(128, 64, kernel_size=3, padding=1) #/16
        self.dec2 = ConvBlockUpsample(128, 32) #/8
        self.dec3 = ConvBlockUpsample(64, 16) #/4
        self.dec4 = ConvBlockUpsample(32, 8) #/2
        self.dec5 = ConvBlockUpsample(16, 8) #/1
        
        self.out = ConvBlock(8,out_channels,kernel_size = 3, padding = 1)
        
        self.out_activation = torch.sigmoid

    def forward(self,X):
        
        enc1 = self.enc1(X)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        
        bottleneck = self.bottleneck(enc4)
        
        dec1 = self.dec1(bottleneck)
        dec2 = self.dec2(torch.cat([dec1,enc4], dim = 1))
        dec3 = self.dec3(torch.cat([dec2,enc3],dim = 1))
        dec4 = self.dec4(torch.cat([dec3,enc2],dim = 1))
        dec5 = self.dec5(torch.cat([dec4,enc1],dim=1))
        
        out = self.out(dec5)
        
        return self.out_activation(out)


if __name__ == "__main__":
    model = UNet(in_channels=3,out_channels=2)
    out = model(torch.randn(1,3,256,256)).detach().numpy()[0]
    print(out.shape)
