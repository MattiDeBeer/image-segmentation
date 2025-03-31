import torch
import torch.nn as nn
import torchvision.models as models
from processing_blocks import *
from customDatasets.datasets import ImageDataset3Mask, DummyDataset
from models.helperFunctions import *

class ClipSegmentationModel(nn.Module):

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
    


        


if __name__ == '__main__':
    dataset_loc = '../../Datasets/Oxford-IIIT-Pet-Augmented'
    train_dataset = DummyDataset(label_channels=3)

    image = train_dataset[0][0].unsqueeze(0)

    print(image.size())

    model = ClipSegmentationModel()

    print(model(image).size())