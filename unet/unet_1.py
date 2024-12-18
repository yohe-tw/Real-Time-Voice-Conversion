import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNN(nn.Module):
    def __init__(self, input_channels=256):
        super(SimpleCNN, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, stride=2, padding=1),  # Downsampling via stride
            nn.ReLU(inplace=True)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # Downsampling via stride
            nn.ReLU(inplace=True)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),  # Downsampling via stride
            nn.ReLU(inplace=True)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),  # No downsampling
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        feat1 = self.block1(x)  # Shape: [batch_size, 64, height//2, width//2]
        feat2 = self.block2(feat1)  # Shape: [batch_size, 128, height//4, width//4]
        feat3 = self.block3(feat2)  # Shape: [batch_size, 256, height//8, width//8]
        feat4 = self.block4(feat3)  # Shape: [batch_size, 512, height//8, width//8]
        return feat1, feat2, feat3, feat4


# +
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=32):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, latent_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2 = self.conv_block(latent_dim, 2 * latent_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = self.conv_block(2 * latent_dim, 4 * latent_dim)

        # Decoder
        self.upconv2 = nn.ConvTranspose2d(4 * latent_dim, 2 * latent_dim, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(4 * latent_dim, 2 * latent_dim)
        
        self.upconv1 = nn.ConvTranspose2d(2 * latent_dim, latent_dim, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(2 * latent_dim, latent_dim)

        # Final Convolution
        self.final_conv = nn.Conv2d(latent_dim, out_channels, kernel_size=1)

        # Tanh Activation for Output
        self.tanh = nn.Tanh()

    def conv_block(self, in_channels, out_channels):
        """Simple Convolutional Block: Conv -> ReLU -> Conv -> ReLU"""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))

        # Decoder
        dec2 = self.upconv2(bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.upconv1(dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # Final Output
        out = self.final_conv(dec1)
        out = self.tanh(out)  # Apply Tanh Activation
        return out
