import torch
import torchaudio
import sys
import os
from utils import compute_mel_spectrogram

from hifigan.models import Generator  # Import the generator class from HiFi-GAN repo

# Initialize HiFi-GAN Generator
hifi_gan = Generator(None)
checkpoint_path = 'generator_v3'

# Load pre-trained weights
checkpoint = torch.load(checkpoint_path, map_location='cpu')
hifi_gan.load_state_dict(checkpoint['generator'])
hifi_gan.eval()  # Set the model to evaluation mode

# Generate waveform from mel-spectrogram
mel_spectrogram = compute_mel_spectrogram("../../../Chinese/ZH-Alto-1/Pharyngeal/你就不要想起我/Pharyngeal_Group/0002.wav")
waveform = hifi_gan(mel_spectrogram)
torchaudio.save("output_test.wav", waveform.cpu(), sample_rate=48000)
