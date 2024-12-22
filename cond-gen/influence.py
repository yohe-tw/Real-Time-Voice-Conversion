import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
import torchaudio.transforms
import torchaudio.functional
import torch.optim as optim
import torch
import os
from torch.utils.data import Dataset, DataLoader, random_split
from collections import OrderedDict
import psutil  # For monitoring memory usage
from torch.nn.functional import cosine_similarity
import random
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import sys
from tqdm import tqdm
from torchsummary import summary
from utils import parse_wav_files, denormalize_mel, plot_mel_spectrogram_direct, convert_mel_to_wav
from data import VocalTechniqueDataset
from cond_unet import ContextUnet

sys.path.append('..')
from unet.unet_1 import UNet
print("Switched back to:", os.getcwd())

# Ensure device is set
device = "cuda" if torch.cuda.is_available() else "cpu"


#################
base_directory = "../.."  # Replace with the actual path
wav_dict = parse_wav_files(base_directory)

# Remove file paths with "Paired_Speech_Group"
for key, value in wav_dict.items():
    value['file_path'] = [
        path for path in value['file_path']
        if "Paired_Speech_Group" not in path
    ]
#################


# Compute global min and max
# global_mel_min, global_mel_max = compute_global_min_max(wav_dict)
# Min: -100.0, Max: 53.63450622558594
global_mel_min = -100.0
global_mel_max = 54
print(f"Global Mel-Spectrogram Min: {global_mel_min}, Max: {global_mel_max}")


# create dataset
dataset = VocalTechniqueDataset(wav_dict, global_mel_min=global_mel_min, global_mel_max=global_mel_max)
dataset.validate_pairs()

model = ContextUnet(in_channels=1, n_classes=5).to(device)  # Move to GPU if available



model.load_state_dict(torch.load("cond.pt", map_location=device))
model.eval()  # Set the model to evaluation mode

data = dataset[0]  # Replace with the desired sample index
control_mel = data["control_mel"].unsqueeze(0).to(device)
technique_mel = data["technique_mel"].unsqueeze(0).to(device)
label = torch.tensor(data["label"]).unsqueeze(0).to(device)

zero_context_mask = torch.full((1, ), 0).to(device)


print(control_mel.shape)

model_output = None

# Run inference
with torch.no_grad():  # Disable gradient computation for inference
    
    output = model(control_mel, label, zero_context_mask)
    model_output = denormalize_mel(output, global_min=global_mel_min, global_max=global_mel_max)
    plot_mel_spectrogram_direct(model_output, path="output.png")
    # print(model_output) 

    
# List of denormalized mel-spectrograms (e.g., model outputs)
mel_spectrograms = model_output  # Replace with actual tensors
mel_spectrograms = model_output.squeeze(1)
mel_spectrograms = mel_spectrograms[:, 40:120, :]  # Select the first 80 mel bins

# Directory to save generated .wav files
output_dir = "./generated_wavs"
os.makedirs(output_dir, exist_ok=True)

# Generate audio files
output_path = os.path.join(output_dir, f"output_nice.wav")
convert_mel_to_wav(mel_spectrograms, output_path)

plot_mel_spectrogram_direct(technique_mel, path="tech.png")
print(mel_spectrograms.shape)
# technique_mel = technique_mel.squeeze(1)
print(technique_mel.shape)
technique_mel = F.interpolate(technique_mel, size=(80, technique_mel.size(-1)), mode="bilinear", align_corners=False)
print(technique_mel.shape)
convert_mel_to_wav(technique_mel[0], os.path.join(output_dir, f"output_teq.wav"))