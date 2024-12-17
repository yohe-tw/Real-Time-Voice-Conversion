import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import resnet50
from vgg import VGG16


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


class unetUp(nn.Module):
    def __init__(self, in_size_encoder, in_size_decoder, out_size):
        super(unetUp, self).__init__()
        # The first convolution's in_size is the sum of encoder and decoder feature map channels
        self.conv1 = nn.Conv2d(in_size_encoder + in_size_decoder, out_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs1, inputs2):
        """
        Args:
            inputs1: Feature map from the encoder (skip connection)
            inputs2: Upsampled feature map from the previous decoder layer
        """
        # Debug: Log input shapes
        # print(f"[DEBUG] inputs1 shape: {inputs1.shape}")
        # print(f"[DEBUG] inputs2 shape before upsampling: {inputs2.shape}")

        # Upsample inputs2
        upsampled_inputs2 = self.up(inputs2)

        # Debug: Log shape after upsampling
        # print(f"[DEBUG] inputs2 shape after upsampling: {upsampled_inputs2.shape}")

        # Align dimensions
        if upsampled_inputs2.size(2) != inputs1.size(2) or upsampled_inputs2.size(3) != inputs1.size(3):
            diff_h = inputs1.size(2) - upsampled_inputs2.size(2)
            diff_w = inputs1.size(3) - upsampled_inputs2.size(3)
            # print(f"[DEBUG] Dimension differences - Height: {diff_h}, Width: {diff_w}")

            # Pad or crop upsampled_inputs2
            upsampled_inputs2 = F.pad(
                upsampled_inputs2,
                (diff_w // 2, diff_w - diff_w // 2, diff_h // 2, diff_h - diff_h // 2)
            )

        # Concatenate along the channel dimension
        outputs = torch.cat([inputs1, upsampled_inputs2], 1)

        # Debug: Log shape after concatenation
        # print(f"[DEBUG] outputs shape after concatenation: {outputs.shape}")

        # Apply convolutions
        outputs = self.conv1(outputs)
        outputs = self.relu(outputs)
        outputs = self.conv2(outputs)
        outputs = self.relu(outputs)

        # Debug: Log final output shape
        # print(f"[DEBUG] outputs shape after final conv: {outputs.shape}")
        return outputs

class Unet(nn.Module):
    def __init__(self, input_channels=256, embedding_dim=10):
        super(Unet, self).__init__()

        # Use SimpleCNN as the backbone
        self.backbone = SimpleCNN(input_channels=256)
        in_filters = [64, 128, 256, 512]  # Filters for each layer in the backbone
        out_filters = [64, 128, 256, 512]  # Filters for the U-Net decoder layers

        # Upsampling layers
        self.up_concat4 = unetUp(in_size_encoder=256, in_size_decoder=512, out_size=out_filters[3])  # For feat4
        self.up_concat3 = unetUp(in_size_encoder=128, in_size_decoder=out_filters[3], out_size=out_filters[2])
        self.up_concat2 = unetUp(in_size_encoder=64, in_size_decoder=out_filters[2], out_size=out_filters[1])
        self.up_concat1 = unetUp(in_size_encoder=input_channels, in_size_decoder=out_filters[1], out_size=out_filters[0])

        # Final output layer (1 channel for mel-spectrogram)
        self.final = nn.Conv2d(out_filters[0], 128, kernel_size=1)

        # Technique embedding layer
        self.embedding = nn.Linear(embedding_dim, 128)

    def forward(self, inputs, technique_embedding):
        
        # Process technique embedding
        technique_embedding = self.embedding(technique_embedding)  # Shape: [batch_size, 128]
        technique_embedding = technique_embedding.unsqueeze(-1).unsqueeze(-1)  # Shape: [batch_size, 128, 1, 1]
        
        # print(f"[DEBUG] inputs shape: {inputs.shape}")

        # Debug: Log shape after unsqueezing
        # print(f"[DEBUG] technique_embedding shape after unsqueeze: {technique_embedding.shape}")
        # print(f"technique_embedding: {technique_embedding}")

        # Broadcast technique embedding to match input spatial dimensions
        technique_embedding = technique_embedding.expand(-1, -1, inputs.size(2), inputs.size(3))  # Shape: [batch_size, 128, height, width]

        # Debug: Log shape after expanding
        # print(f"[DEBUG] technique_embedding shape after expand: {technique_embedding.shape}")

        # Combine inputs and technique_embedding along the channel dimension
        inputs = torch.cat([inputs, technique_embedding], dim=1)

        # Forward pass through the backbone
        feat1, feat2, feat3, feat4 = self.backbone(inputs)

        # Decoder with skip connections
        up4 = self.up_concat4(feat3, feat4)
        up3 = self.up_concat3(feat2, up4)
        up2 = self.up_concat2(feat1, up3)
        up1 = self.up_concat1(inputs, up2)

        # Final layer
        final = self.final(up1)
        return final
