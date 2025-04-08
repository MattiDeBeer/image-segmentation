import torch
import torch.nn as nn
import torchvision.models as models
from models.processing_blocks import *
from models.helperFunctions import *
from customDatasets.datasets import DummyDataset

class ClipResSegmentationModel(nn.Module):
    """
    ClipResSegmentationModel is a neural network model designed for image segmentation tasks. 
    It combines features extracted from a CLIP-based feature extractor and a ResNet34-based 
    feature extractor using cross-attention fusion. The model then performs a series of 
    upsampling and convolution operations to generate the final segmentation output.
    Attributes:
        clip_feature_extractor (nn.Module): A CLIP-based feature extractor for obtaining 
            high-level semantic features from the input image.
        encoder (nn.Module): A ResNet34-based feature extractor for obtaining spatially 
            rich features from the input image.
        cross_attention_fusion (nn.Module): A cross-attention mechanism to fuse features 
            from the CLIP and ResNet34 feature extractors.
        dec1, dec2, dec3, dec4, dec5 (nn.Module): Convolutional upsampling blocks used 
            to progressively upsample and refine the fused features.
        out (nn.Module): A final convolutional block that generates the output segmentation 
            map with the specified number of output channels.
    Methods:
        __init__(out_channels=3, in_channels=3, activation=nn.Identity()):
            Initializes the ClipResSegmentationModel with the specified parameters.
        forward(X):
            Performs a forward pass through the model. Takes an input tensor `X` and 
            returns the segmentation output.
    """
    def __init__(self, out_channels = 3, in_channels = 3, activation = nn.Identity() ):
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
    """
    ClipUnet is a U-Net-based architecture designed for image segmentation tasks, 
    incorporating CLIP (Contrastive Language–Image Pretraining) features for enhanced 
    contextual understanding. The model fuses image features with CLIP features using 
    cross-attention mechanisms to improve segmentation performance.
    Attributes:
        clip_feature_extractor (ClipFeatureExtractor): A feature extractor based on CLIP, 
            used to extract high-level semantic features from input images.
        cross_attention_fusion (CrossAttentionFusion): A module that fuses bottleneck 
            features with CLIP features using cross-attention.
        input (nn.Conv2d): A convolutional layer to process the input image.
        enc1, enc2, enc3 (ConvBlockDownsample): Downsampling convolutional blocks for 
            encoding features at different resolutions.
        bottleneck (ConvBlock): A convolutional block at the bottleneck of the U-Net 
            architecture.
        dec1, dec2, dec3, dec4 (ConvBlockUpsampleSkip): Upsampling convolutional blocks 
            with skip connections for decoding features.
        out (nn.Conv2d): A convolutional layer to produce the final output.
        activation (nn.Module): An activation function applied to the output.
    Methods:
        forward(X):
            Performs a forward pass through the network. Takes an input tensor `X`, 
            extracts CLIP features, processes the input through the U-Net encoder-decoder 
            structure, fuses features using cross-attention, and produces the final output.
    """
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
    """
    A segmentation model utilizing CLIP features for image processing.
    This class is named `ClipAutoencoder`, but it is not a traditional autoencoder. 
    Instead, it is a segmentation model that leverages CLIP features as a bottleneck 
    representation to guide the decoding process for segmentation tasks.
    Attributes:
        clip_feature_extractor (ClipFeatureExtractor): A feature extractor based on CLIP, 
            used to generate feature embeddings from input images.
        input (nn.Conv2d): A convolutional layer to process the input image.
        coupler (nn.Linear): A linear layer to map CLIP features to a spatial representation.
        dec1 (ConvBlockUpsample): The first upsampling block in the decoder.
        dec2 (ConvBlockUpsample): The second upsampling block in the decoder.
        dec3 (ConvBlockUpsample): The third upsampling block in the decoder.
        dec4 (ConvBlockUpsampleSkip): The final upsampling block with skip connections.
        out (nn.Conv2d): A convolutional layer to produce the final output.
        activation (nn.Module): An activation function applied to the output.
    Methods:
        forward(X):
            Performs a forward pass through the model. Takes an input tensor `X`, extracts 
            CLIP features, processes the input through the decoder, and produces the 
            segmentation output.
    """
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

        bottleneck = self.coupler(clip_features).view(-1,64,16,16)

        dec1 = self.dec1(bottleneck)
        dec2 = self.dec2(dec1)
        dec3 = self.dec3(dec2)
        dec4 = self.dec4(dec3,input)
        out = self.out(dec4)

        return out
    

if __name__ == '__main__':
    train_dataset = DummyDataset(label_channels=3)
    image = train_dataset[0][0].unsqueeze(0)
    model = ClipUnet()
    mask, label = model(image)
    print(f"mask size:{mask.size()}")
    print(f"class size: {label.size()}")