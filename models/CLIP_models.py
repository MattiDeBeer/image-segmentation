import torch
import torch.nn as nn
import torchvision.models as models
from models.processing_blocks import *
from models.helperFunctions import *
from customDatasets.datasets import DummyDataset

class ClipResSegmentationModel(nn.Module):

    def __init__(self,out_channels = 3):
        super().__init__()

        self.clip_feature_extractor = ClipFeatureExtractor(train=False)
        self.encoder = ResNet34FeatureExtractor(train=False)
        self.cross_attention_fusion = CrossAttentionFusion(512,num_heads=4)

        self.dec1 = ConvBlockUpsample(512,256)
        self.dec2 = ConvBlockUpsample(256,128)
        self.dec3 = ConvBlockUpsample(128,64)
        self.dec4 = ConvBlockUpsample(64,32)
        self.dec5 = ConvBlockUpsample(32,16)

        self.out = ConvBlock(in_channels=19,out_channels=out_channels)
    
    def forward(self,X):

        clip_features = self.clip_feature_extractor(X)
        resnet_features = self.encoder(X)

        attn = self.cross_attention_fusion(resnet_features,clip_features)

        dec1 = self.dec1(attn)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3)
        dec5 = self.dec5(dec4)
        out = self.out(torch.cat([dec5,X],dim=1))

        return out
    
class ClipUnet(nn.Module):

    def __init__(self,out_channels = 3, in_channels = 3, activation = nn.Identity() ):
        super().__init__()

        self.clip_feature_extractor = ClipFeatureExtractor(train=False)
        self.cross_attention_fusion = CrossAttentionFusion(512,num_heads=1)

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
    
    def forward(self,X):

        clip_features = self.clip_feature_extractor(X)

        input = self.input(X)
        enc1 = self.enc1(input)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
     
        bottleneck = self.bottleneck(enc3)

        attention_output = self.cross_attention_fusion(bottleneck,clip_features)

        dec1 = self.dec1(attention_output, enc3)
        dec2 = self.dec2(dec1, enc2)
        dec3 = self.dec3(dec2, enc1)
        dec4 = self.dec4(dec3,input)
        out = self.out(dec4)

        return out
    
class ClipAutoencoder(nn.Module):

    def __init__(self,out_channels = 3, in_channels = 3, activation = nn.Identity() ):
        super().__init__()

        self.clip_feature_extractor = ClipFeatureExtractor(train=False)

        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)

        self.coupler = nn.Linear(512,16384)
       
        self.dec1 = ConvBlockUpsample(64, 64)
        self.dec2 = ConvBlockUpsample(64, 64)
        self.dec3 = ConvBlockUpsample(64, 32)
        self.dec4 = ConvBlockUpsampleSkip(32, 32) 

        self.out = nn.Conv2d(32, out_channels, kernel_size=1, padding=0)

        self.activation = activation
    
    def forward(self,X):

        clip_features = self.clip_feature_extractor(X)

        input = self.input(X)

        bottleneck = self.coupler(clip_features).reshape(-1,64,16,16)

        dec1 = self.dec1(bottleneck)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3,input)
        out = self.out(dec4)

        return out
    

if __name__ == '__main__':
    dataset_loc = '../../Datasets/Oxford-IIIT-Pet-Augmented'
    train_dataset = DummyDataset(label_channels=3)

    image = train_dataset[0][0].unsqueeze(0)

    print(image.size())

    model = ClipAutoencoder()

    print(model(image).size())