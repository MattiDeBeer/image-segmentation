import torch 
import torch
import torch.nn as nn
from models.processing_blocks import ConvBlockDownsample, ConvBlockUpsampleSkip, ConvBlock, ConvBlockUpsample


class UNet(nn.Module):
    """
    A U-Net architecture for image segmentation tasks.
    The U-Net is a convolutional neural network designed for biomedical image segmentation. 
    It consists of an encoder-decoder structure with skip connections between corresponding 
    layers in the encoder and decoder. The encoder compresses the spatial dimensions while 
    capturing features, and the decoder reconstructs the spatial dimensions using the features 
    from the encoder.
    Attributes:
        input (nn.Conv2d): Initial convolutional layer to process input channels.
        enc1 (ConvBlockDownsample): First downsampling block in the encoder.
        enc2 (ConvBlockDownsample): Second downsampling block in the encoder.
        enc3 (ConvBlockDownsample): Third downsampling block in the encoder.
        bottleneck (ConvBlock): Bottleneck block at the lowest resolution.
        dec1 (ConvBlockUpsampleSkip): First upsampling block in the decoder with skip connection.
        dec2 (ConvBlockUpsampleSkip): Second upsampling block in the decoder with skip connection.
        dec3 (ConvBlockUpsampleSkip): Third upsampling block in the decoder with skip connection.
        dec4 (ConvBlockUpsampleSkip): Fourth upsampling block in the decoder with skip connection.
        out (nn.Conv2d): Final convolutional layer to produce the output.
        activation (nn.Module): Activation function applied to the output.
    Methods:
        forward(x):
            Defines the forward pass of the U-Net model.
            Args:
                x (torch.Tensor): Input tensor of shape (batch_size, in_channels, height, width).
            Returns:
                torch.Tensor: Output tensor of shape (batch_size, out_channels, height, width).
    """

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
    """
    LargeUNet is a deep convolutional neural network architecture for image segmentation tasks.
    It follows the U-Net design pattern with an encoder-decoder structure, including skip connections
    between corresponding encoder and decoder layers to preserve spatial information.
    Attributes:
        input (nn.Conv2d): Initial convolutional layer to process input channels.
        enc1 (ConvBlockDownsample): First downsampling block in the encoder.
        enc2 (ConvBlockDownsample): Second downsampling block in the encoder.
        enc3 (ConvBlockDownsample): Third downsampling block in the encoder.
        enc4 (ConvBlockDownsample): Fourth downsampling block in the encoder.
        bottleneck (ConvBlock): Bottleneck block at the deepest level of the network.
        dec1 (ConvBlockUpsampleSkip): First upsampling block in the decoder with skip connection.
        dec2 (ConvBlockUpsampleSkip): Second upsampling block in the decoder with skip connection.
        dec3 (ConvBlockUpsampleSkip): Third upsampling block in the decoder with skip connection.
        dec4 (ConvBlockUpsampleSkip): Fourth upsampling block in the decoder with skip connection.
        dec5 (ConvBlockUpsampleSkip): Final upsampling block in the decoder with skip connection.
        out (nn.Conv2d): Final convolutional layer to produce the output segmentation map.
        activation (nn.Module): Activation function applied to the output (default: nn.Identity).
    Methods:
        forward(x):
            Defines the forward pass of the network. Takes an input tensor `x` and returns the
            segmentation map after passing through the encoder, bottleneck, and decoder.
    """

    def __init__(self, in_channels=3, out_channels=3, activation = nn.Identity()):
        super().__init__()
        
        self.input = nn.Conv2d(in_channels, 32, kernel_size=1, padding=0)
       
        # Encoder
        self.enc1 = ConvBlockDownsample(32, 64) 
        self.enc2 = ConvBlockDownsample(64, 128)
        self.enc3 = ConvBlockDownsample(128, 256)
        self.enc4 = ConvBlockDownsample(256, 512)
        
        # Bottleneck
        self.bottleneck = ConvBlock(512, 1024)
        
        #decoder
        self.dec1 = ConvBlockUpsampleSkip(1024, 512)
        self.dec2 = ConvBlockUpsampleSkip(512, 256)
        self.dec3 = ConvBlockUpsampleSkip(256, 128)
        self.dec4 = ConvBlockUpsampleSkip(128, 64)
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
     
        # Bottleneck
        bottleneck = self.bottleneck(enc4)

        # Decoder
        dec1 = self.dec1(bottleneck, enc4)
        dec2 = self.dec2(dec1, enc3)
        dec3 = self.dec3(dec2, enc2)
        dec4 = self.dec4(dec3, enc1)
        dec5 = self.dec5(dec4, input)
        out = self.out(dec5)

        return self.activation(out)
    

if __name__ == "__main__":
    model = LargeUNet()
    out = model(torch.randn(1,3,256,256))
    print(out.size())


    
