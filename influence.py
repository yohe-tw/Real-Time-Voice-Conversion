import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms
import torchaudio.functional
import torch.optim as optim
import torch
import os
import random
import sys
from tqdm import tqdm
from torchsummary import summary
from utils.utils import parse_wav_files, denormalize_mel, plot_mel_spectrogram_direct, convert_mel_to_wav, compute_mel_spectrogram
from utils.data import VocalTechniqueDataset
from unet.unet_2 import UNet
import argparse

# Ensure device is set
device = "cuda" if torch.cuda.is_available() else "cpu"


#################
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_path', type=str, default="./generated_wavs")
    parser.add_argument('--testing_path', type=str, default="../Chinese/ZH-Alto-1/Pharyngeal/你就不要想起我/Control_Group/0002.wav")
    parser.add_argument('--model_path', type=str, default="outputmodel/cond.pt")
    parser.add_argument('--condition', type=int, default=1)
    args = parser.parse_args()
    return args

args = parse_args()


#################


# Compute global min and max
# global_mel_min, global_mel_max = compute_global_min_max(wav_dict)
# Min: -100.0, Max: 53.63450622558594
global_mel_min = -100.0
global_mel_max = 54
print(f"Global Mel-Spectrogram Min: {global_mel_min}, Max: {global_mel_max}")

model = UNet(latent_dim=32).to(device)  # Move to GPU if available

model.load_state_dict(torch.load(args.model_path, map_location=device))
model.eval()  # Set the model to evaluation mode



control_mel = compute_mel_spectrogram(args.testing_path).unsqueeze(0)
control_mel = control_mel[:, :, :, 10:266]

model_output = None

# Run inference
with torch.no_grad():  # Disable gradient computation for inference
    print(control_mel.shape)
    output = model(control_mel.to(device), torch.tensor([args.condition]).to(device), 0)
    model_output = denormalize_mel(output, global_min=global_mel_min, global_max=global_mel_max)
    # plot_mel_spectrogram_direct(model_output, path="output.png")
    # print(model_output) 

    
# List of denormalized mel-spectrograms (e.g., model outputs)
mel_spectrograms = model_output  # Replace with actual tensors
mel_spectrograms = model_output.squeeze(1)

# Directory to save generated .wav files
output_dir = args.output_path
os.makedirs(output_dir, exist_ok=True)

# Generate audio files
output_path = os.path.join(output_dir, f"output_nice.wav")
convert_mel_to_wav(mel_spectrograms, output_path)

