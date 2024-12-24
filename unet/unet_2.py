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

class EmbedFC(nn.Module):
    def __init__(self, input_dim, emb_dim):
        super(EmbedFC, self).__init__()
        '''
        generic one layer FC NN for embedding things  
        '''
        self.input_dim = input_dim
        layers = [
            nn.Linear(input_dim, emb_dim),
            nn.GELU(),
            nn.Linear(emb_dim, emb_dim),
        ]
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        x = x.view(-1, self.input_dim)
        return self.model(x)

# +
import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, latent_dim=32, n_classes=5):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, latent_dim)
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.enc2 = self.conv_block(latent_dim, 2 * latent_dim)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        # Bottleneck
        self.bottleneck = self.conv_block(2 * latent_dim, 4 * latent_dim)

        # contextembed
        self.contextembed1 = EmbedFC(n_classes, 4*latent_dim)
        self.contextembed2 = EmbedFC(n_classes, 2*latent_dim)
        self.latent_dim = latent_dim
        self.n_classes = n_classes

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

    def forward(self, x, c, context_mask_probability=0.1):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool1(enc1))

        # Bottleneck
        bottleneck = self.bottleneck(self.pool2(enc2))

        
        # mask out context if context_mask == 1
        context_mask = torch.zeros_like(c)
        context_mask[torch.rand(c.shape) < 0.1] = 1
        context_mask = context_mask[:, None]
        context_mask = context_mask.repeat(1,self.n_classes)
        context_mask = (-1*(1-context_mask)) # need to flip 0 <-> 1

        
        # convert context to one hot embedding
        c = nn.functional.one_hot(c, num_classes=self.n_classes).type(torch.float)
        c = c * context_mask

        # embed context
        cemb1 = self.contextembed1(c).view(-1, self.latent_dim * 4, 1, 1)
        cemb2 = self.contextembed2(c).view(-1, self.latent_dim * 2, 1, 1)


        # Decoder
        dec2 = self.upconv2(cemb1*bottleneck)
        dec2 = self.dec2(torch.cat([dec2, enc2], dim=1))

        dec1 = self.upconv1(cemb2*dec2)
        dec1 = self.dec1(torch.cat([dec1, enc1], dim=1))

        # Final Output
        out = self.final_conv(dec1)
        out = self.tanh(out)  # Apply Tanh Activation
        return out
