#!/usr/bin/env python
# coding: utf-8
# %%
import torch
import torch.nn as nn

class SVE_VAE(nn.Module):
    def __init__(self, latent_dim=64):
        
        """
        Initialize the Shallow CNN + Bi-LSTM Encoder and CNN Decoder for Mel-Spectrogram.
        
        Args:
            latent_dim (int): Dimensionality of the latent space in the encoder.
        """
        super(SVE_VAE, self).__init__()
        
        self.latent_dim = latent_dim

        # Encoder - Shallow CNN + Bi-LSTM
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, stride=1, padding=1),  # 1 channel input
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Downsample to [64, 200, 16]
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # Downsample to [32, 100, 32]
        )
        
        self.encoder_reshape = nn.Sequential(
            nn.Flatten(start_dim=2),  # Flatten: [batch_size, 32, 6400]
            nn.Linear(4096, 1024)     # Project: [batch_size, 32, 1024]
        )
        self.encoder_bilstm1 = nn.LSTM(1024, 128, bidirectional=True, batch_first=True)
        self.encoder_bilstm2 = nn.LSTM(256, latent_dim, bidirectional=True, batch_first=True)

        # Decoder - Bi-LSTM + Upsampling CNN
        self.decoder_repeat = nn.Linear(latent_dim, 64 * 256)  # Expand latent space to [batch_size, 64 * 256]
        self.decoder_bilstm1 = nn.LSTM(256, 128, bidirectional=True, batch_first=True)  # Bi-LSTM processes time steps

        self.decoder = nn.Sequential(
            # Upsample: [batch, 16, 64, 16] -> [batch, 16, 128, 32]
            nn.ConvTranspose2d(16, 16, kernel_size=4, stride=(2, 2), padding=1, output_padding=0),
            nn.ReLU(),

            # Upsample: [batch, 16, 128, 32] -> [batch, 8, 256, 64]
            nn.ConvTranspose2d(16, 8, kernel_size=4, stride=(2, 2), padding=1, output_padding=0),
            nn.ReLU(),

            # Fix ConvTranspose2D3: [batch, 8, 256, 64] -> [batch, 4, 256, 128]
            nn.ConvTranspose2d(8, 4, kernel_size=4, stride=(2, 2), padding=1, output_padding=0),
            nn.ReLU(),

            # Final Output: [batch, 4, 256, 128] -> [batch, 1, 256, 256]
            nn.ConvTranspose2d(4, 1, kernel_size=4, stride=(2, 2), padding=1, output_padding=0),
            nn.Tanh()
        )

    def encode(self, x):
        
        encoder_intermediates = []

        # Encoder Layers
        print(f"Input shape: {x.shape}")
        x = self.encoder[0](x)
        print(f"After Conv2d(1 -> 16): {x.shape}")
        x = self.encoder[2](x)
        print(f"After MaxPool2d: {x.shape}")
        encoder_intermediates.append(x)

        x = self.encoder[3](x)
        print(f"After Conv2d(16 -> 32): {x.shape}")
        x = self.encoder[5](x)
        print(f"After MaxPool2d: {x.shape}")
        encoder_intermediates.append(x)

        # Bi-LSTM Layers
        x = self.encoder_reshape(x)
        print(f"After Flatten: {x.shape}")
        x, _ = self.encoder_bilstm1(x)
        print(f"After Bi-LSTM1: {x.shape}")
        x, _ = self.encoder_bilstm2(x)
        print(f"After Bi-LSTM2: {x.shape}")

        # Mean and log-variance
        mu = x[:, -1, :self.latent_dim]         # First half is mu
        logvar = x[:, -1, self.latent_dim:]     # Second half is logvar
        print(f"Latent mean shape: {mu.shape}, logvar shape: {logvar.shape}")
        return mu, logvar, encoder_intermediates

    def decode(self, latent):
        """
        Decode the latent space to reconstruct the mel-spectrogram.
        Args:
            latent (tensor): Latent vector of shape [batch_size, latent_dim].
        Returns:
            recon (tensor): Reconstructed mel-spectrogram.
            decoder_intermediates (list): Intermediate outputs for multi-scale loss.
        """
        decoder_intermediates = []

        # Step 1: Expand latent space to [batch_size, 64 * 256]
        print(f"Latent input shape: {latent.shape}")
        x = self.decoder_repeat(latent)  # Linear layer expands latent vector
        print(f"After Linear: {x.shape}")

        # Step 2: Reshape to [batch_size, 64, 256]
        x = x.view(-1, 64, 256)
        print(f"After Reshape for Bi-LSTM: {x.shape}")

        # Step 3: Bi-LSTM processing
        x, _ = self.decoder_bilstm1(x)  # Output: [batch_size, 64, 256]
        print(f"After Bi-LSTM: {x.shape}")

        # Step 4: Reshape for CNN input: [batch_size, channels=16, height=64, width=16]
        batch_size, time_steps, features = x.size()
        x = x.contiguous().view(batch_size, 16, time_steps, features // 16)
        print(f"After Reshape for CNN: {x.shape}")

        # Step 5: CNN Transpose Layers for Upsampling
        x = self.decoder[0](x)
        print(f"After ConvTranspose2D1: {x.shape}")
        decoder_intermediates.append(x)

        x = self.decoder[2](x)
        print(f"After ConvTranspose2D2: {x.shape}")
        decoder_intermediates.append(x)

        x = self.decoder[4](x)
        print(f"After ConvTranspose2D3: {x.shape}")
        decoder_intermediates.append(x)

        x = self.decoder[6](x)
        print(f"After Final Conv2D: {x.shape}")

        return x, decoder_intermediates

    def forward(self, x):
        
        mu, logvar, encoder_intermediates = self.encode(x)
        z = self.reparameterize(mu, logvar)       # Sample latent vector
        recon, decoder_intermediates = self.decode(z)  # Pass z to decoder
        return recon, mu, logvar, encoder_intermediates, decoder_intermediates
    
    def reparameterize(self, mu, logvar):
        """
        Reparameterization trick: Sample z from N(mu, sigma^2).
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def multi_scale_loss(self, input, recon, scales, criterion):
        """
        Compute multi-scale reconstruction loss by downsampling input and reconstructed outputs.

        Args:
            input (tensor): Ground truth mel-spectrogram of shape [batch_size, 1, height, width].
            recon (tensor): Reconstructed mel-spectrogram of shape [batch_size, 1, height, width].
            scales (list of int): List of downsampling factors.
            criterion (loss function): Reconstruction loss criterion (e.g., MSELoss).

        Returns:
            loss (tensor): Combined multi-scale reconstruction loss.
        """
        loss = criterion(recon, input)  # Loss at original scale

        # Compute losses at multiple scales
        for scale in scales:
            if scale > 1:
                # Downsample input and recon to smaller scales
                downsampled_input = torch.nn.functional.avg_pool2d(input, kernel_size=scale)
                downsampled_recon = torch.nn.functional.avg_pool2d(recon, kernel_size=scale)

                # Add scaled reconstruction loss
                scale_loss = criterion(downsampled_recon, downsampled_input)
                loss += scale_loss

        return loss
    
    def multi_scale_loss_with_intermediates(self, input, recon, input_intermediates, recon_intermediates, scales, criterion):
        """
        Compute the multi-scale loss, including intermediate outputs.
        """
        # Loss on final output
        loss = self.multi_scale_loss(input, recon, scales, criterion)
        
        # Add losses for intermediate outputs
        for inp_intermediate, rec_intermediate in zip(input_intermediates, recon_intermediates):
            loss += criterion(rec_intermediate, inp_intermediate)

        return loss

    def vae_loss(self, input, recon, mu, logvar, input_intermediates, recon_intermediates, scales, criterion):
        """
        VAE Loss: Multi-scale reconstruction loss + KL divergence.
        """
        # Multi-scale reconstruction loss
        recon_loss = self.multi_scale_loss_with_intermediates(input, recon, input_intermediates, recon_intermediates, scales, criterion)

        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

        # Combine losses
        return recon_loss + kl_loss

